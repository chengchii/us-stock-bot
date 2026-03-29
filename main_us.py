import os
import discord
from discord.ext import commands, tasks
import pandas as pd
import asyncio
import json
import requests
import yfinance as yf
from datetime import datetime, time, timezone
from threading import Thread
from flask import Flask

# --- 偽裝成瀏覽器的 Session ---
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
})

# --- Render 存活檢查 ---
app = Flask('')
@app.route('/')
def home(): return "US Quant Bot: Master Level Engine Online!"
def run(): app.run(host='0.0.0.0', port=10000)
def keep_alive(): Thread(target=run).start()

# ==========================================
# 1. 系統與環境變數設定
# ==========================================
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
CHANNEL_ID = os.environ.get("CHANNEL_ID")
TWELVE_API_KEY = os.environ.get("TWELVEDATA_API_KEY") 
INVEST_AMOUNT = 1000  

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# ==========================================
# 2. 自動抓取 NASDAQ 100 最新成分股名單
# ==========================================
def get_nasdaq_100_tickers():
    try:
        print("🌐 正在下載 NASDAQ 100 成分股...")
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        html_data = session.get(url, timeout=10).text 
        from io import StringIO
        tables = pd.read_html(StringIO(html_data))
        for table in tables:
            if 'Ticker' in table.columns:
                df = table
                break
        tickers = df['Ticker'].tolist()
        names_dict = {}
        for idx, row in df.iterrows():
            sym = row['Ticker']
            name = str(row['Company']).split()[0].replace(',', '')
            names_dict[sym] = name
        return tickers, names_dict
    except Exception as e:
        print(f"❌ 抓取名單失敗: {e}")
        backup_list = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "NFLX", "AVGO"]
        return backup_list, {s: s for s in backup_list}

WATCHLIST, STOCK_NAMES = get_nasdaq_100_tickers()
print(f"✅ 成功載入 {len(WATCHLIST)} 檔 NASDAQ 100 股票！")

# ==========================================
# 3. 虛擬帳本系統
# ==========================================
PORTFOLIO_FILE = "portfolio_us.json" 

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f: return json.load(f)
        except: pass
    return {"cash": 0.0, "holdings": {}, "last_month": ""}

def save_portfolio(p):
    with open(PORTFOLIO_FILE, "w") as f: json.dump(p, f, indent=4)

# ==========================================
# 4. 完美數據引擎 (Twelve Data API)
# ==========================================
async def fetch_multi_timeframe_data(symbol):
    if not TWELVE_API_KEY:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    loop = asyncio.get_running_loop()
    def get_twelve_data(interval, outputsize):
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={TWELVE_API_KEY}"
        try:
            response = requests.get(url, timeout=10).json()
            if 'values' in response:
                df = pd.DataFrame(response['values'])
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                df = df.sort_index(ascending=True)
                df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
                return df
            return pd.DataFrame()
        except:
            return pd.DataFrame()
    df_1h = await loop.run_in_executor(None, lambda: get_twelve_data("1h", 1000))
    df_1d = await loop.run_in_executor(None, lambda: get_twelve_data("1day", 500))
    df_1wk = await loop.run_in_executor(None, lambda: get_twelve_data("1week", 260))
    return df_1h, df_1d, df_1wk

# ==========================================
# 5. 量化技術指標計算大腦
# ==========================================
def calculate_indicators(df):
    if df.empty or len(df) < 200: return df
    df['EMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_25'] = df['Close'].ewm(span=25, adjust=False).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_75'] = df['Close'].ewm(span=75, adjust=False).mean()
    df['EMA_140'] = df['Close'].ewm(span=140, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    macd = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    df['MACD'], df['MACD_SIG'] = macd, sig
    delta = df['Close'].diff()
    up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
    df['RSI_75'] = 100 - (100 / (1 + up.ewm(com=74, adjust=False).mean() / down.ewm(com=74, adjust=False).mean()))
    low_min = df['Low'].rolling(window=14).min()
    high_max = df['High'].rolling(window=14).max()
    k_fast = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['STOCH_K'] = k_fast.rolling(window=3).mean()
    df['STOCH_D'] = df['STOCH_K'].rolling(window=3).mean()
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(window=14).mean()
    return df

# ==========================================
# 6. 單股多維度全息掃描 (華爾街 5 大模塊)
# ==========================================
async def process_stock_query(channel, symbol):
    initial_msg = await channel.send(f"🔍 正在對 `{symbol}` 啟動全息多維度掃描...")
    try:
        df_1h, df_1d, df_1wk = await fetch_multi_timeframe_data(symbol)
        if df_1d.empty or len(df_1d) < 200:
            await initial_msg.edit(content=f"❌ `{symbol}` 數據不足。")
            return
            
        df_1h = calculate_indicators(df_1h)
        df_1d = calculate_indicators(df_1d)
        df_1wk = calculate_indicators(df_1wk)

        c_price = df_1d['Close'].iloc[-1]
        atr = df_1d['ATR_14'].iloc[-1]
        ema200 = df_1d['EMA_200'].iloc[-1]
        
        vol_ma20 = df_1d['Volume'].rolling(20).mean().iloc[-1]
        vol_ratio = df_1d['Volume'].iloc[-1] / vol_ma20 if vol_ma20 > 0 else 0
        bias_200 = ((c_price - ema200) / ema200) * 100
        bias_str = f"{bias_200:+.1f}%"
        if bias_200 > 30: bias_str += " ⚠️高乖離"
        upper_bb = df_1d['MA_20'].iloc[-1] + (2 * df_1d['Close'].rolling(20).std().iloc[-1])

        trend_status = "🟢 長線多頭 (順勢做多)" if c_price > ema200 else "🔴 長線空頭 (順勢做空)"
        
        h_macd, h_sig = df_1h['MACD'].iloc[-1], df_1h['MACD_SIG'].iloc[-1]
        h_ema14, h_ema50 = df_1h['EMA_14'].iloc[-1], df_1h['EMA_50'].iloc[-1]
        short_signal = "觀望 (動能不足)"
        if c_price > ema200 and (h_macd > h_sig) and (h_macd < 0) and (h_ema14 > h_ema50):
            short_signal = "🚀【極限狙擊】MACD 零軸下金叉且均線共振"
        
        d_ema25, d_ema75, d_ema140 = df_1d['EMA_25'].iloc[-1], df_1d['EMA_75'].iloc[-1], df_1d['EMA_140'].iloc[-1]
        if (d_ema25 > d_ema75 > d_ema140) and (df_1d['RSI_75'].iloc[-1] > 50):
            d_k, d_d = df_1d['STOCH_K'].iloc[-1], df_1d['STOCH_D'].iloc[-1]
            mid_signal = "🎯【黃金買點】隨機指標超賣區金叉" if (d_k > d_d and d_k < 30) else "⏳【耐心等待】多頭排列強勢，等待 KD 落入 20 以下"
        else:
            mid_signal = "觀望 (結構未明)"

        long_signal = "📈【持股續抱】" if df_1wk['MA_20'].iloc[-1] > df_1wk['MA_50'].iloc[-1] else "觀望 (趨勢未成)"

        atr_multiplier = 1.5
        stop_loss = c_price - (atr_multiplier * atr)
        risk = c_price - stop_loss
        target_price = c_price + (risk * 2) 

        msg = f"📊 **【{symbol} 華爾街量化全息診斷】**\n"
        msg += f"> 💵 現價: **`${c_price:.2f}`** | ⚡ 成交量: `{vol_ratio:.1f}x` 均量\n"
        msg += f"> 📏 200EMA 乖離: `{bias_str}` | 🌋 短壓 (布林): `${upper_bb:.2f}`\n"
        msg += "========================================\n"
        msg += f"🛡️ **模塊一 (大局視角)**: {trend_status}\n"
        msg += f"⏱️ **短線策略 (1H)**: {short_signal}\n"
        msg += f"📅 **中線波段 (1D)**: {mid_signal}\n"
        msg += f"🔭 **長線跟蹤 (1W)**: {long_signal}\n"
        msg += "========================================\n"
        msg += f"⚖️ **模塊五 (ATR 動態風險管理)**\n"
        if c_price > ema200:
            msg += f"🔴 **初始停損**: 跌破 `${stop_loss:.2f}` 立即認賠\n"
            msg += f"🟢 **進攻目標**: 上看 `${target_price:.2f}` (盈虧比 1:2)\n"
            msg += f"💡 **ATR 移動停利**: 獲利後請將出場點調高至 `波段最高價 - ${(atr_multiplier * atr):.2f}`\n"
        else:
            msg += "⚠️ 目前處於長線空頭，系統拒絕給予做多計畫。"

        await initial_msg.edit(content=msg)
    except Exception as e:
        await initial_msg.edit(content=f"❌ 系統錯誤！")


# ==========================================
# 7. 華爾街雙軌決策 AI 掃描引擎 (趨勢順勢 vs 恐慌抄底)
# ==========================================
async def perform_scan(force_send=False):
    channel = bot.get_channel(int(CHANNEL_ID))
    if not channel: return

    p = load_portfolio()
    msg_lines = []
    
    curr_month = datetime.now().strftime("%Y-%m")
    if p.get("last_month", "") != curr_month:
        p["cash"] = p.get("cash", 0.0) + INVEST_AMOUNT
        p["last_month"] = curr_month
        msg_lines.append(f"🏦 **美股入金**：已存入 ${INVEST_AMOUNT} USD，現金餘額：`${p['cash']:.2f}`")

    # 🛡️ 總經雷達：判定當前市場環境
    try:
        vix_data = yf.download("^VIX", period="5d", progress=False)
        current_vix = float(vix_data['Close'].iloc[-1].iloc[0]) if not vix_data.empty else 20.0
    except:
        current_vix = 20.0

    is_bull_market = False
    is_panic_market = False

    if current_vix > 30:
        is_panic_market = True
        msg_lines.append(f"🚨 **【總經核彈警報】** VIX 恐慌指數飆升至 `{current_vix:.2f}`！")
        msg_lines.append("> 🛡️ **右側策略建議**: 趨勢已遭破壞，建議保守投資者空手觀望，保護本金。")
        msg_lines.append("> 🩸 **左側策略建議**: 巴菲特貪婪時刻！啟動「極度超賣 + 基本面護城河」抄底掃描...")
    else:
        _, df_market, _ = await fetch_multi_timeframe_data("SPY")
        if df_market.empty:
            msg_lines.append("⚠️ 警告：無法取得 SPY 數據，暫停買進。")
        else:
            ma60_market = df_market['Close'].tail(60).mean()
            is_bull_market = df_market['Close'].iloc[-1] > ma60_market
            if is_bull_market:
                msg_lines.append(f"🦅 大盤安全 (VIX: `{current_vix:.1f}`)。啟動「大師級順勢動能」掃描...")
            else:
                msg_lines.append("🛑 **【美股警報】** S&P 500 (SPY) 跌破季線。盤整偏空，暫停右側突破買進。")

    results = []
    panic_buy_results = []
    
    # 只要是大牛市(順勢)，或是大恐慌(抄底)，我們都啟動掃描
    if is_bull_market or is_panic_market:
        scan_msg = await channel.send("機房全力運算中，正在掃描 NASDAQ 100 強 (約需 3~5 分鐘)...")
        
        for s in WATCHLIST:
            _, df, _ = await fetch_multi_timeframe_data(s)
            if df.empty or len(df) < 200: continue
            
            try:
                close = df['Close']
                vol = df['Volume']
                ema200 = close.ewm(span=200, adjust=False).mean()
                macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
                sig = macd.ewm(span=9, adjust=False).mean()
                
                delta = close.diff()
                up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
                rsi = 100 - (100 / (1 + up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean()))
                
                ma20 = close.rolling(window=20).mean()
                std20 = close.rolling(window=20).std()
                upper_bb = ma20 + (2 * std20)
                vol_ma20 = vol.rolling(window=20).mean()

                high_low = df['High'] - df['Low']
                high_close = (df['High'] - df['Close'].shift()).abs()
                low_close = (df['Low'] - df['Close'].shift()).abs()
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = tr.rolling(window=14).mean().iloc[-1]

                c_price = close.iloc[-1]
                c_macd = macd.iloc[-1]
                c_sig = sig.iloc[-1]
                c_rsi = rsi.iloc[-1]
                c_vol = vol.iloc[-1]
                c_vol_ma20 = vol_ma20.iloc[-1]

                score = 0
                badges = []

                if c_price > ema200.iloc[-1]: score += 15
                if c_macd > c_sig: score += 10
                if c_macd > 0: score += 10 

                vol_ratio = c_vol / c_vol_ma20 if c_vol_ma20 > 0 else 0
                vol_score = min(30, int(vol_ratio * 10)) 
                score += vol_score
                if vol_score >= 20: badges.append("🔥機構爆量")

                if c_price > upper_bb.iloc[-1]:
                    score += 20; badges.append("🌋突破壓力")
                elif c_price > ma20.iloc[-1]:
                    score += 10 

                if 55 <= c_rsi <= 65:
                    score += 15; badges.append("🎯完美時機")
                elif 50 < c_rsi < 55 or 65 < c_rsi <= 70:
                    score += 10
                elif c_rsi > 70:
                    score += 5; badges.append("⚠️留意追高")

                # 軌道 1：右側順勢名單 (大牛市才會觸發)
                if is_bull_market and score >= 60 and c_macd > c_sig:
                    results.append({
                        'symbol': s, 'name': STOCK_NAMES.get(s, s), 'price': c_price,
                        'score': score, 'vol_ratio': vol_ratio, 'badges': badges, 'atr': atr
                    })
                
                # 軌道 2：左側抄底名單 (大恐慌才會觸發，條件是 RSI 跌破 30 極度超賣)
                if is_panic_market and c_rsi < 30:
                    panic_buy_results.append({
                        'symbol': s, 'name': STOCK_NAMES.get(s, s), 'price': c_price,
                        'rsi': c_rsi, 'vol_ratio': vol_ratio, 'badges': ["🩸極度超賣"], 'atr': atr
                    })
            except:
                continue
            await asyncio.sleep(0.3)
            
        try: await scan_msg.delete()
        except: pass

        # --- 輸出報告與基本面過濾 ---
        final_buy_targets = []
        
        # 情境 A：順勢多頭報告
        if is_bull_market and results:
            results.sort(key=lambda x: (x['score'], x['vol_ratio']), reverse=True)
            msg_lines.append(f"\n📈 **【右側順勢：動能強勢股名單】**")
            for idx, r in enumerate(results[:5]):
                try:
                    info = yf.Ticker(r['symbol']).info
                    if info.get('trailingEps', 1) is not None and info.get('trailingEps', 1) < 0:
                        r['badges'].append("☠️虧損企業(剔除)")
                        r['score'] -= 50 
                except: pass
                
                badge_str = " ".join(r['badges'])
                msg_lines.append(f"🔸 **{r['symbol']} ({r['name']})** | 評分: `{r['score']}` | 標籤: {badge_str}")
                if r['score'] >= 60 and "☠️虧損企業(剔除)" not in r['badges']: final_buy_targets.append(r)

        # 情境 B：左側抄底報告
        if is_panic_market and panic_buy_results:
            panic_buy_results.sort(key=lambda x: x['rsi']) # RSI 越低(越超賣)排越前面
            msg_lines.append(f"\n📉 **【左側抄底：價值浮現巨頭名單】** (已過濾基本面)")
            for idx, r in enumerate(panic_buy_results[:5]):
                # 抄底極度危險，嚴格檢查基本面，虧損公司絕對不碰
                try:
                    info = yf.Ticker(r['symbol']).info
                    eps = info.get('trailingEps', 1)
                    if eps is not None and eps < 0: continue # 直接跳過不顯示
                except: pass
                
                msg_lines.append(f"🩸 **{r['symbol']} ({r['name']})** | 現價: `${r['price']:.2f}` | RSI: `{r['rsi']:.1f}`")
                final_buy_targets.append(r) # 將優質抄底目標納入買進名單
                
        if not final_buy_targets and (is_bull_market or is_panic_market):
            msg_lines.append("\n🔎 AI 巡邏完畢，目前無符合雙軌策略之標的。")

    # 💼 持股停利損檢查 (ATR 動態追蹤)
    for sym, data_p in list(p.get("holdings", {}).items()):
        _, df, _ = await fetch_multi_timeframe_data(sym)
        if df.empty: continue
        curr_p = df['Close'].iloc[-1]
        
        tr = pd.concat([df['High'] - df['Low'], (df['High'] - df['Close'].shift()).abs(), (df['Low'] - df['Close'].shift()).abs()], axis=1).max(axis=1)
        curr_atr = tr.rolling(window=14).mean().iloc[-1]
        
        if curr_p > data_p["high_price"]: data_p["high_price"] = curr_p
        
        trailing_stop = data_p["high_price"] - (1.5 * curr_atr)
        ema200 = df['EMA_200'].iloc[-1] if 'EMA_200' in df else curr_p * 0.5
        
        if curr_p < trailing_stop or curr_p < ema200:
            sell_val = data_p["shares"] * curr_p
            profit_pct = ((curr_p - data_p["avg_cost"]) / data_p["avg_cost"]) * 100
            p["cash"] += sell_val
            name = STOCK_NAMES.get(sym, sym)
            msg_lines.append(f"🚨 **ATR 動態平倉**：{name} 賣出價 `${curr_p:.2f}` (總報酬 `{profit_pct:.1f}%`)")
            del p["holdings"][sym]

    # 🛒 橋水注碼法：波動率平價分配 (無論是順勢還是抄底，都適用嚴格風控)
    if final_buy_targets and p.get("cash", 0) > 10: 
        buy_targets = final_buy_targets[:5] 
        msg_lines.append(f"\n🛒 **【機構級自動注碼 (ATR 平價)】**")
        total_risk_capital = p.get("cash", 0) * 0.02 
        
        for target in buy_targets:
            sym = target['symbol']
            if sym not in p.get("holdings", {}):
                risk_per_share = 1.5 * target['atr']
                if risk_per_share <= 0: risk_per_share = target['price'] * 0.05 
                
                shares = round(total_risk_capital / risk_per_share, 4)
                max_investment = p["cash"] * 0.25
                cost = shares * target['price']
                if cost > max_investment:
                    shares = round(max_investment / target['price'], 4)
                    cost = shares * target['price']
                
                if shares > 0.0001 and p["cash"] >= cost:
                    p["cash"] -= cost
                    if "holdings" not in p: p["holdings"] = {}
                    p["holdings"][sym] = {"shares": shares, "avg_cost": target['price'], "high_price": target['price']}
                    msg_lines.append(f"└ 買入 {sym} `{shares:.4f}` 股 (投入 `${cost:.2f}`) | ATR:`{target['atr']:.2f}`")

    save_portfolio(p)
    if force_send or msg_lines:
        final_msg = "\n".join(msg_lines)
        await channel.send(final_msg[:1990])
# ==========================================
# 8. 智慧指令路由
# ==========================================
@bot.event
async def on_message(message):
    if message.author == bot.user: return
    
    raw_content = message.content.strip()
    upper_content = raw_content.upper()

    if raw_content in ["美股庫存", "庫存", "帳本"]:
        await show_portfolio(message.channel)
        return
    if raw_content in ["美股掃描", "全面掃描", "大盤"]:
        await perform_scan(force_send=True)
        return

    if upper_content in STOCK_NAMES:
        await process_stock_query(message.channel, upper_content)
        return
    elif upper_content.startswith('$') and upper_content[1:].isalpha() and len(upper_content) <= 6:
        await process_stock_query(message.channel, upper_content[1:])
        return

    await bot.process_commands(message)

# ==========================================
# 🌟 每日自動掃描鬧鐘
# ==========================================
scan_time = time(hour=22, minute=0, tzinfo=timezone.utc)

@tasks.loop(time=scan_time)
async def daily_scan_task():
    today_weekday = datetime.now().weekday()
    if 1 <= today_weekday <= 5:
        channel = bot.get_channel(int(CHANNEL_ID))
        if channel:
            await channel.send("🌅 **【早安華爾街】** 美股已收盤，系統自動啟動大師級晨間掃描...")
        await perform_scan(force_send=True)

@bot.event
async def on_ready():
    print(f"🦅 US Bot Online: {bot.user}")
    keep_alive() 
    if not daily_scan_task.is_running():
        daily_scan_task.start()
        print("⏰ 每日晨間掃描鬧鐘已啟動 (排定於台灣時間 06:00)")

if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        print("❌ 找不到 Discord Bot Token")
    else:
        bot.run(DISCORD_BOT_TOKEN)
