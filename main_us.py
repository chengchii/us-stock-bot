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
from io import StringIO

# --- 偽裝成瀏覽器的 Session ---
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
})

# --- Render 存活檢查 ---
app = Flask('')
@app.route('/')
def home(): return "US Quant Bot: Alpaca Ultimate Engine Online!"
def run(): app.run(host='0.0.0.0', port=10000)
def keep_alive(): Thread(target=run).start()

# ==========================================
# 1. 系統與環境變數設定
# ==========================================
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")       # 🆕 Alpaca Key
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY") # 🆕 Alpaca Secret
INVEST_AMOUNT = 1000  

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# ==========================================
# 2. 自動抓取 NASDAQ 100 最新成分股名單
# ==========================================
def get_nasdaq_100_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        html_data = session.get(url, timeout=10).text 
        tables = pd.read_html(StringIO(html_data))
        for table in tables:
            if 'Ticker' in table.columns:
                df = table
                break
        tickers = df['Ticker'].tolist()
        names_dict = {}
        for idx, row in df.iterrows():
            sym = row['Ticker']
            names_dict[sym] = str(row['Company']).split()[0].replace(',', '')
        return tickers, names_dict
    except:
        backup_list = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "NFLX", "AVGO"]
        return backup_list, {s: s for s in backup_list}

WATCHLIST, STOCK_NAMES = get_nasdaq_100_tickers()

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
# 4. 🚀 Alpaca API 終極數據引擎 (高速、穩定、精準)
# ==========================================
async def fetch_multi_timeframe_data(symbol, daily_only=False):
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("❌ 找不到 Alpaca API Keys！")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    loop = asyncio.get_running_loop()
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "accept": "application/json"
    }
    
    # Alpaca 歷史數據端點
    base_url = "https://data.alpaca.markets/v2/stocks/bars"

    def get_alpaca_data(timeframe, limit):
        # timeframe 格式: '1Hour', '1Day', '1Week'
        # 由於是免費方案，資料會有 15 分鐘延遲 (sip feed 需要付費，免費版用 iex)
        url = f"{base_url}?symbols={symbol}&timeframe={timeframe}&limit={limit}&feed=iex"
        try:
            res = requests.get(url, headers=headers, timeout=10).json()
            if 'bars' in res and symbol in res['bars']:
                # Alpaca 原始欄位: t(time), o(open), h(high), l(low), c(close), v(volume)
                df = pd.DataFrame(res['bars'][symbol])
                df['datetime'] = pd.to_datetime(df['t'])
                df.set_index('datetime', inplace=True)
                df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
                return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            return pd.DataFrame()
        except Exception as e:
            return pd.DataFrame()

    # 高速併發抓取！不再需要 sleep 降速！
    if daily_only:
        df_1d = await loop.run_in_executor(None, lambda: get_alpaca_data("1Day", 500))
        return pd.DataFrame(), df_1d, pd.DataFrame()
    else:
        df_1h = await loop.run_in_executor(None, lambda: get_alpaca_data("1Hour", 1000))
        df_1d = await loop.run_in_executor(None, lambda: get_alpaca_data("1Day", 500))
        df_1wk = await loop.run_in_executor(None, lambda: get_alpaca_data("1Week", 260))
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
# 6. 單股多維度全息掃描
# ==========================================
async def process_stock_query(channel, symbol):
    initial_msg = await channel.send(f"🔍 正在對 `{symbol}` 啟動 Alpaca 高頻引擎掃描...")
    loop = asyncio.get_running_loop()
    try:
        df_1h, df_1d, df_1wk = await fetch_multi_timeframe_data(symbol, daily_only=False)
        if df_1d.empty or len(df_1d) < 200:
            await initial_msg.edit(content=f"❌ `{symbol}` 數據不足。請確認代號正確。")
            return
            
        df_1h = calculate_indicators(df_1h)
        df_1d = calculate_indicators(df_1d)
        df_1wk = calculate_indicators(df_1wk)

        c_price = df_1d['Close'].iloc[-1]
        atr = df_1d['ATR_14'].iloc[-1]
        ema200 = df_1d['EMA_200'].iloc[-1]
        
        # 外資共識還是借用 yfinance 的輕量 info 抓取
        def fetch_info():
            try: return yf.Ticker(symbol).info
            except: return {}
        info = await loop.run_in_executor(None, fetch_info)
        target_price_wallst = info.get('targetMeanPrice', 0)
        eps = info.get('trailingEps', 0)
        
        upside_str = ""
        if target_price_wallst > 0:
            upside = ((target_price_wallst - c_price) / c_price) * 100
            upside_str = f" | 🎯目標價: `${target_price_wallst:.2f}` (`{upside:+.1f}%`)"
        eps_str = f" | 💼 EPS: `{eps}`" + (" ⚠️虧損" if eps and eps < 0 else " ✅獲利")

        vol_ma20 = df_1d['Volume'].rolling(20).mean().iloc[-1]
        vol_ratio = df_1d['Volume'].iloc[-1] / vol_ma20 if vol_ma20 > 0 else 0
        bias_200 = ((c_price - ema200) / ema200) * 100
        bias_str = f"{bias_200:+.1f}%"
        if bias_200 > 30: bias_str += " ⚠️高乖離"
        upper_bb = df_1d['MA_20'].iloc[-1] + (2 * df_1d['Close'].rolling(20).std().iloc[-1])

        trend_status = "🟢 長線多頭" if c_price > ema200 else "🔴 長線空頭"
        
        h_macd, h_sig = df_1h['MACD'].iloc[-1], df_1h['MACD_SIG'].iloc[-1]
        h_ema14, h_ema50 = df_1h['EMA_14'].iloc[-1], df_1h['EMA_50'].iloc[-1]
        short_signal = "觀望"
        if c_price > ema200 and (h_macd > h_sig) and (h_macd < 0) and (h_ema14 > h_ema50):
            short_signal = "🚀【極限狙擊】MACD 金叉且均線共振"
        
        d_ema25, d_ema75, d_ema140 = df_1d['EMA_25'].iloc[-1], df_1d['EMA_75'].iloc[-1], df_1d['EMA_140'].iloc[-1]
        if (d_ema25 > d_ema75 > d_ema140) and (df_1d['RSI_75'].iloc[-1] > 50):
            d_k, d_d = df_1d['STOCH_K'].iloc[-1], df_1d['STOCH_D'].iloc[-1]
            mid_signal = "🎯【黃金買點】KD 跌落超賣區金叉" if (d_k > d_d and d_k < 30) else "⏳【耐心等待】多頭強勢，等 KD 落入 20"
        else:
            mid_signal = "觀望"

        long_signal = "📈【持股續抱】" if df_1wk['MA_20'].iloc[-1] > df_1wk['MA_50'].iloc[-1] else "觀望"

        atr_multiplier = 1.5
        stop_loss = c_price - (atr_multiplier * atr)
        risk = c_price - stop_loss
        target_price = c_price + (risk * 2) 

        msg = f"📊 **【{symbol} 華爾街量化全息診斷】**\n"
        msg += f"> 💵 現價: **`${c_price:.2f}`** | ⚡ 成交量: `{vol_ratio:.1f}x` 均量\n"
        msg += f"> 📏 200EMA 乖離: `{bias_str}` | 🌋 短壓 (布林): `${upper_bb:.2f}`\n"
        msg += f"> 🏦 **機構共識**:{upside_str}{eps_str}\n"
        msg += "========================================\n"
        msg += f"🛡️ **模塊一 (大局視角)**: {trend_status}\n"
        msg += f"⏱️ **短線策略 (1H)**: {short_signal}\n"
        msg += f"📅 **中線波段 (1D)**: {mid_signal}\n"
        msg += f"🔭 **長線跟蹤 (1W)**: {long_signal}\n"
        msg += "========================================\n"
        msg += f"⚖️ **模塊五 (ATR 動態風險管理)**\n"
        if c_price > ema200:
            msg += f"🔴 **初始停損**: 跌破 `${stop_loss:.2f}` 立即認賠\n"
            msg += f"🟢 **進攻目標**: 上看 `${target_price:.2f}`\n"
            msg += f"💡 **ATR 移動停利**: 獲利後請將出場點調高至 `最高價 - ${(atr_multiplier * atr):.2f}`\n"
        else:
            msg += "⚠️ 目前處於長線空頭，系統拒絕給予做多計畫。"

        await initial_msg.edit(content=msg)
    except Exception as e:
        await initial_msg.edit(content=f"❌ 系統錯誤！{e}")

# ==========================================
# 7. 終極 AI 掃描引擎 (Alpaca 高速版)
# ==========================================
async def perform_scan(channel=None, force_send=False):
    if channel is None:
        try: channel = bot.get_channel(int(CHANNEL_ID))
        except: return 
    if not channel: return
    
    loop = asyncio.get_running_loop()
    p = load_portfolio()
    msg_lines = []
    
    curr_month = datetime.now().strftime("%Y-%m")
    if p.get("last_month", "") != curr_month:
        p["cash"] = p.get("cash", 0.0) + INVEST_AMOUNT
        p["last_month"] = curr_month
        msg_lines.append(f"🏦 **美股入金**：已存入 ${INVEST_AMOUNT} USD，現金餘額：`${p['cash']:.2f}`")

    try:
        def get_vix(): return yf.download("^VIX", period="5d", progress=False)
        vix_data = await loop.run_in_executor(None, get_vix)
        current_vix = float(vix_data['Close'].iloc[-1].iloc[0]) if not vix_data.empty else 20.0
    except: current_vix = 20.0

    is_bull_market = False
    is_panic_market = False
    spy_mom20 = 0.0 

    if current_vix > 30:
        is_panic_market = True
        msg_lines.append(f"🚨 **【總經核彈警報】** VIX 恐慌指數飆升至 `{current_vix:.2f}`！")
        msg_lines.append("> 🩸 啟動「極度超賣 + 基本面護城河」抄底掃描...")
    else:
        _, df_market, _ = await fetch_multi_timeframe_data("SPY", daily_only=True)
        if df_market.empty:
            msg_lines.append("⚠️ 警告：無法取得 SPY 數據，暫停買進。")
        else:
            spy_close = df_market['Close']
            ma60_market = spy_close.tail(60).mean()
            is_bull_market = spy_close.iloc[-1] > ma60_market
            if len(spy_close) >= 20:
                spy_mom20 = (spy_close.iloc[-1] - spy_close.iloc[-20]) / spy_close.iloc[-20]
            
            if is_bull_market:
                msg_lines.append(f"🦅 大盤安全 (VIX: `{current_vix:.1f}`)。啟動「大師級順勢動能」掃描...")
            else:
                msg_lines.append("🛑 **【美股警報】** S&P 500 跌破季線。暫停右側突破買進。")

    results = []
    panic_buy_results = []
    
    if is_bull_market or is_panic_market:
        for s in WATCHLIST:
            _, df, _ = await fetch_multi_timeframe_data(s, daily_only=True)
            if df.empty or len(df) < 200: continue
            
            try:
                df = calculate_indicators(df)
                close = df['Close']
                c_price = close.iloc[-1]
                c_macd = df['MACD'].iloc[-1]
                c_sig = df['MACD_SIG'].iloc[-1]
                c_rsi = df['RSI_75'].iloc[-1]
                ema200 = df['EMA_200'].iloc[-1]
                atr = df['ATR_14'].iloc[-1]
                ma20 = df['MA_20'].iloc[-1]
                upper_bb = ma20 + (2 * close.rolling(window=20).std().iloc[-1])
                vol_ma20 = df['Volume'].rolling(window=20).mean().iloc[-1]

                stock_mom20 = (c_price - close.iloc[-20]) / close.iloc[-20] if len(close) >= 20 else 0
                rs_score = stock_mom20 - spy_mom20 

                score = 0
                badges = []

                if c_price > ema200: score += 15
                if c_macd > c_sig: score += 10
                if c_macd > 0: score += 10 
                if rs_score > 0: 
                    score += 10
                    if rs_score > 0.05: badges.append("🌟抗跌領漲")

                vol_ratio = df['Volume'].iloc[-1] / vol_ma20 if vol_ma20 > 0 else 0
                vol_score = min(30, int(vol_ratio * 10)) 
                score += vol_score
                if vol_score >= 20: badges.append("🔥機構爆量")

                if c_price > upper_bb:
                    score += 20; badges.append("🌋突破壓力")
                elif c_price > ma20: score += 10 

                if 55 <= c_rsi <= 65: score += 15; badges.append("🎯完美時機")
                elif 50 < c_rsi < 55 or 65 < c_rsi <= 70: score += 10
                elif c_rsi > 70: score += 5; badges.append("⚠️留意追高")

                if is_bull_market and score >= 60 and c_macd > c_sig:
                    results.append({'symbol': s, 'name': STOCK_NAMES.get(s, s), 'price': c_price, 'score': score, 'vol_ratio': vol_ratio, 'badges': badges, 'atr': atr, 'rs': rs_score})
                
                if is_panic_market and c_rsi < 30:
                    panic_buy_results.append({'symbol': s, 'name': STOCK_NAMES.get(s, s), 'price': c_price, 'rsi': c_rsi, 'vol_ratio': vol_ratio, 'badges': ["🩸極度超賣"], 'atr': atr, 'rs': rs_score})
            except: pass

        final_buy_targets = []
        def check_fundamentals(sym):
            try: return yf.Ticker(sym).info
            except: return {}

        if is_bull_market and results:
            results.sort(key=lambda x: (x['score'], x['rs']), reverse=True) 
            msg_lines.append(f"\n📈 **【右側順勢：高 RS 領頭羊名單】**")
            for idx, r in enumerate(results[:5]):
                info = await loop.run_in_executor(None, check_fundamentals, r['symbol'])
                if info.get('trailingEps', 1) is not None and info.get('trailingEps', 1) < 0:
                    r['badges'].append("☠️虧損剔除")
                    r['score'] -= 50 
                
                target_p = info.get('targetMeanPrice', 0)
                upside_str = ""
                if target_p > r['price']:
                    upside_str = f" | 🎯潛在空間 `+{((target_p - r['price'])/r['price'])*100:.1f}%`"
                elif target_p > 0 and target_p < r['price']:
                    r['badges'].append("⚠️估值偏高")
                
                badge_str = " ".join(r['badges'])
                msg_lines.append(f"🔸 **{r['symbol']} ({r['name']})** | 評分:`{r['score']}` | RS:`{r['rs']*100:+.1f}%`{upside_str}")
                msg_lines.append(f"> 🏷️ 標籤: {badge_str}")
                if r['score'] >= 60 and "☠️虧損剔除" not in r['badges'] and "⚠️估值偏高" not in r['badges']: 
                    final_buy_targets.append(r)

        if is_panic_market and panic_buy_results:
            panic_buy_results.sort(key=lambda x: x['rsi']) 
            msg_lines.append(f"\n📉 **【左側抄底：價值浮現巨頭名單】**")
            for idx, r in enumerate(panic_buy_results[:5]):
                info = await loop.run_in_executor(None, check_fundamentals, r['symbol'])
                eps = info.get('trailingEps', 1)
                if eps is not None and eps < 0: continue 
                msg_lines.append(f"🩸 **{r['symbol']} ({r['name']})** | 現價: `${r['price']:.2f}` | RSI: `{r['rsi']:.1f}`")
                final_buy_targets.append(r) 

    # 💼 持股與注碼邏輯...
    if final_buy_targets and p.get("cash", 0) > 10: 
        msg_lines.append(f"\n🛒 **【機構級自動注碼 (ATR 平價)】**")
        total_risk_capital = p.get("cash", 0) * 0.02 
        for target in final_buy_targets[:5]:
            sym = target['symbol']
            if sym not in p.get("holdings", {}):
                risk_per_share = 1.5 * target['atr']
                if risk_per_share <= 0: risk_per_share = target['price'] * 0.05 
                shares = round(total_risk_capital / risk_per_share, 4)
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

async def show_portfolio(channel):
    p = load_portfolio()
    msg = f"💼 **美股機構帳戶總覽**\n💵 **可用現金：** `${p.get('cash', 0):.2f}` USD\n"
    if not p.get("holdings"): msg += "📭 目前空手觀望中。"
    else:
        msg += "📦 **當前持股 (ATR 動態保護中)：**\n"
        for sym, d in p["holdings"].items():
            msg += f"🔸 **{sym}** ({STOCK_NAMES.get(sym, sym)}): `{d['shares']:.4f}`股 (平均成本:`${d['avg_cost']:.2f}`)\n"
    await channel.send(msg)

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
        await message.channel.send("⚡ 指令已接收！Alpaca 引擎高速掃描中，請稍候約 1 分鐘...")
        asyncio.create_task(perform_scan(channel=message.channel, force_send=True))
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
    if 1 <= datetime.now().weekday() <= 5:
        await perform_scan(channel=None, force_send=True) 

@bot.event
async def on_ready():
    print(f"🦅 US Bot Online: {bot.user}")
    keep_alive() 
    if not daily_scan_task.is_running(): daily_scan_task.start()

if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN: print("❌ 找不到 Discord Bot Token")
    else: bot.run(DISCORD_BOT_TOKEN)
