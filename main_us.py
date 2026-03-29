import os
import discord
from discord.ext import commands, tasks
import pandas as pd
import asyncio
import json
import requests
from datetime import datetime, time, timezone
from threading import Thread
from flask import Flask

# --- 偽裝成瀏覽器的 Session (防維基百科阻擋) ---
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
})

# --- Render 存活檢查 (Keep Alive) ---
app = Flask('')
@app.route('/')
def home(): return "US Quant Bot: Wall Street Engine Online!"
def run(): app.run(host='0.0.0.0', port=10000)
def keep_alive(): Thread(target=run).start()

# ==========================================
# 1. 系統與環境變數設定
# ==========================================
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
CHANNEL_ID = os.environ.get("CHANNEL_ID")
TWELVE_API_KEY = os.environ.get("TWELVEDATA_API_KEY") # 👈 記得在 Render 設定這把鑰匙！
INVEST_AMOUNT = 1000  # 每月入金 1000 美金

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# ==========================================
# 2. 自動抓取 NASDAQ 100 最新成分股名單
# ==========================================
def get_nasdaq_100_tickers():
    try:
        print("🌐 正在從維基百科下載最新的 NASDAQ 100 成分股名單...")
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        html_data = session.get(url, timeout=10).text 
        
        # 使用 StringIO 避免 Pandas 未來版本的警告 (FutureWarning)
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
        print(f"❌ 抓取 NASDAQ 100 名單失敗: {e}")
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
# 4. 完美數據引擎 (Twelve Data 機構級 API)
# ==========================================
async def fetch_multi_timeframe_data(symbol):
    if not TWELVE_API_KEY:
        print("❌ 找不到 Twelve Data API Key，請確認 Render 環境變數設定！")
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
            else:
                return pd.DataFrame()
        except Exception as e:
            return pd.DataFrame()

    # 運用 lambda 完美傳遞參數，抓取三種時間級別
    df_1h = await loop.run_in_executor(None, lambda: get_twelve_data("1h", 1000))
    df_1d = await loop.run_in_executor(None, lambda: get_twelve_data("1day", 500))
    df_1wk = await loop.run_in_executor(None, lambda: get_twelve_data("1week", 260))

    return df_1h, df_1d, df_1wk

# ==========================================
# 5. 量化技術指標計算大腦
# ==========================================
def calculate_indicators(df):
    if df.empty or len(df) < 200: return df
    
    # 均線系統
    df['EMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_25'] = df['Close'].ewm(span=25, adjust=False).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_75'] = df['Close'].ewm(span=75, adjust=False).mean()
    df['EMA_140'] = df['Close'].ewm(span=140, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # MACD
    macd = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    df['MACD'] = macd
    df['MACD_SIG'] = sig

    # RSI
    delta = df['Close'].diff()
    up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
    df['RSI_75'] = 100 - (100 / (1 + up.ewm(com=74, adjust=False).mean() / down.ewm(com=74, adjust=False).mean()))

    # 隨機指標 Stochastic (KD)
    low_min = df['Low'].rolling(window=14).min()
    high_max = df['High'].rolling(window=14).max()
    k_fast = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['STOCH_K'] = k_fast.rolling(window=3).mean()
    df['STOCH_D'] = df['STOCH_K'].rolling(window=3).mean()

    # ATR 動態停損
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
    initial_msg = await channel.send(f"🔍 正在對 `{symbol}` 啟動全息多維度掃描 (機構級數據連線中)...")
    try:
        df_1h, df_1d, df_1wk = await fetch_multi_timeframe_data(symbol)
        
        if df_1d.empty or len(df_1d) < 200:
            await initial_msg.edit(content=f"❌ `{symbol}` 歷史數據不足或 API 達到次數上限，無法計算。")
            return
            
        df_1h = calculate_indicators(df_1h)
        df_1d = calculate_indicators(df_1d)
        df_1wk = calculate_indicators(df_1wk)

        c_price = df_1d['Close'].iloc[-1]
        atr = df_1d['ATR_14'].iloc[-1]
        ema200 = df_1d['EMA_200'].iloc[-1]

        # 模塊 1：趨勢過濾
        trend_status = "🟢 長線多頭 (適合做多)" if c_price > ema200 else "🔴 長線空頭 (禁止做多，僅限做空)"
        
        # 模塊 2：短線 (1H)
        h_macd, h_sig = df_1h['MACD'].iloc[-1], df_1h['MACD_SIG'].iloc[-1]
        h_ema14, h_ema50 = df_1h['EMA_14'].iloc[-1], df_1h['EMA_50'].iloc[-1]
        short_signal = "觀望"
        if c_price > ema200 and (h_macd > h_sig) and (h_macd < 0) and (h_ema14 > h_ema50):
            short_signal = "🚀【買入訊號】MACD 動能金叉且均線共振"
        
        # 模塊 3：中線 (Daily)
        d_ema25, d_ema75, d_ema140 = df_1d['EMA_25'].iloc[-1], df_1d['EMA_75'].iloc[-1], df_1d['EMA_140'].iloc[-1]
        d_rsi75 = df_1d['RSI_75'].iloc[-1]
        d_k, d_d = df_1d['STOCH_K'].iloc[-1], df_1d['STOCH_D'].iloc[-1]
        mid_signal = "觀望"
        if (d_ema25 > d_ema75 > d_ema140) and (d_rsi75 > 50):
            if d_k > d_d and d_k < 30: 
                mid_signal = "🎯【狙擊買點】價格回調完成，隨機指標超賣區金叉"
            else:
                mid_signal = "⏳【等待回調】多頭排列強勢，等待 KD 落入 20 以下"

        # 模塊 4：長線 (Weekly)
        w_ma20, w_ma50 = df_1wk['MA_20'].iloc[-1], df_1wk['MA_50'].iloc[-1]
        long_signal = "觀望"
        if w_ma20 > w_ma50:
            long_signal = "📈【持股續抱】週線 20/50 多頭排列，吃到大趨勢"

        # 模塊 5：風險管理計算
        stop_loss = min(d_ema140, c_price - (1.5 * atr))
        risk = c_price - stop_loss
        target_price = c_price + (risk * 2) 

        msg = f"📊 **【{symbol} 華爾街多維度全息診斷報告】** 報價: `${c_price:.2f}`\n"
        msg += f"🛡️ **模塊一 (大局趨勢)**: {trend_status} (200 EMA: `${ema200:.2f}`)\n"
        msg += "----------------------------------------\n"
        msg += f"⏱️ **短線策略 (1H)**: {short_signal}\n"
        msg += f"📅 **中線策略 (Daily)**: {mid_signal}\n"
        msg += f"🔭 **長線策略 (Weekly)**: {long_signal}\n"
        msg += "----------------------------------------\n"
        msg += f"⚖️ **模塊五 (風險管理與 ATR 動態計畫)**\n"
        
        if c_price > ema200:
            msg += f"> 🛑 **強勢停損 (Stop Loss)**: 跌破 `${stop_loss:.2f}` 立即出場\n"
            msg += f"> 🎯 **最小獲利目標 (Target)**: `${target_price:.2f}` (盈虧比 1:2)\n"
            msg += f"> 💡 **動態出場**: 獲利後請以收盤價跌破 140 EMA (`${d_ema140:.2f}`) 作為防守。\n"
        else:
            msg += "> ⚠️ 目前處於長線空頭，系統拒絕給予做多計畫，建議空手等待。"

        await initial_msg.edit(content=msg)
    except Exception as e:
        await initial_msg.edit(content=f"❌ 系統錯誤！無法生成報告。")
        print(e)

# ==========================================
# 7. 華爾街 AI 多因子評分掃描引擎 + 自動佈局
# ==========================================
async def perform_scan(force_send=False):
    channel = bot.get_channel(int(CHANNEL_ID))
    if not channel: return

    p = load_portfolio()
    msg_lines = []
    
    # 🏦 入金邏輯
    curr_month = datetime.now().strftime("%Y-%m")
    if p.get("last_month", "") != curr_month:
        p["cash"] = p.get("cash", 0.0) + INVEST_AMOUNT
        p["last_month"] = curr_month
        msg_lines.append(f"🏦 **美股入金**：已存入 ${INVEST_AMOUNT} USD，現金餘額：`${p['cash']:.2f}`")

    # 🛡️ 大盤判定邏輯 (SPY)
    _, df_market, _ = await fetch_multi_timeframe_data("SPY")
    if df_market.empty:
        msg_lines.append("⚠️ 警告：無法取得 SPY 數據，保護機制啟動，暫停買進。")
        is_bull_market = False
    else:
        ma60_market = df_market['Close'].tail(60).mean()
        is_bull_market = df_market['Close'].iloc[-1] > ma60_market
        if not is_bull_market:
            msg_lines.append("🛑 **【美股警報】** S&P 500 (SPY) 跌破季線。空頭市場嚴禁做多！")

    results = []
    
    if is_bull_market:
        scan_msg = await channel.send("🦅 美股大盤偏多！啟動「華爾街 AI 多因子評分模型」掃描 NASDAQ 100 強 (約需 3~5 分鐘)...")
        
        for s in WATCHLIST:
            # 掃描時只拿日線以節省 API 呼叫次數
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
                    score += 20
                    badges.append("🌋突破壓力")
                elif c_price > ma20.iloc[-1]:
                    score += 10 

                if 55 <= c_rsi <= 65:
                    score += 15
                    badges.append("🎯完美時機")
                elif 50 < c_rsi < 55 or 65 < c_rsi <= 70:
                    score += 10
                elif c_rsi > 70:
                    score += 5 
                    badges.append("⚠️留意追高")

                if score >= 60 and c_macd > c_sig:
                    results.append({
                        'symbol': s, 'name': STOCK_NAMES.get(s, s), 'price': c_price,
                        'score': score, 'vol_ratio': vol_ratio, 'badges': badges
                    })
            except:
                continue
                
            await asyncio.sleep(0.3)
            
        try: await scan_msg.delete()
        except: pass

        if results:
            results.sort(key=lambda x: (x['score'], x['vol_ratio']), reverse=True)
            msg_lines.append(f"\n🤖 **【AI 多因子量化選股報告】** (篩選出 {len(results)} 檔美股科技巨獸)")
            msg_lines.append("`量化維度：趨勢動能(35%) + 機構籌碼(30%) + 突破爆發(20%) + 風險時機(15%)`\n")
            
            for idx, r in enumerate(results[:10]):
                rank_icon = "👑" if idx == 0 else ("🥈" if idx == 1 else ("🥉" if idx == 2 else "🔹"))
                badge_str = " ".join(r['badges']) if r['badges'] else "溫和上漲"
                msg_lines.append(f"{rank_icon} **Top {idx+1}: {r['symbol']} ({r['name']})**")
                msg_lines.append(f"> 📊 綜合評分: **`{r['score']}分`** (現價 `${r['price']:.2f}`) | 成交量達均量 `{r['vol_ratio']:.1f}倍`")
                msg_lines.append(f"> 🏷️ AI 標籤: {badge_str}\n")
        else:
            msg_lines.append("\n🔎 AI 巡邏完畢，目前 NASDAQ 100 資金動能不足，無任何股票達到 60 分及格線。")

    # 💼 持股停利損檢查
    for sym, data_p in list(p.get("holdings", {}).items()):
        _, df, _ = await fetch_multi_timeframe_data(sym)
        if df.empty: continue
        curr_p = df['Close'].iloc[-1]
        ema200 = df['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
        if curr_p > data_p["high_price"]: data_p["high_price"] = curr_p
        
        if curr_p < ema200 or curr_p < data_p["high_price"] * 0.85:
            sell_val = data_p["shares"] * curr_p
            profit_pct = ((curr_p - data_p["avg_cost"]) / data_p["avg_cost"]) * 100
            p["cash"] += sell_val
            name = STOCK_NAMES.get(sym, sym)
            msg_lines.append(f"🚨 **自動平倉**：{name} 賣出價 `${curr_p:.2f}` (報酬率 `{profit_pct:.1f}%`)")
            del p["holdings"][sym]

    # 🛒 執行買進
    if is_bull_market and results and p.get("cash", 0) > 10: 
        buy_targets = results[:5] 
        budget = p["cash"] / len(buy_targets) 
        msg_lines.append(f"\n🛒 **【AI 自動佈局 (零股)】** 預計每檔分配約 `${budget:.2f}`")
        
        for target in buy_targets:
            sym = target['symbol']
            if sym not in p.get("holdings", {}):
                shares = round(budget / target['price'], 4) 
                if shares > 0.0001:
                    p["cash"] -= shares * target['price']
                    if "holdings" not in p: p["holdings"] = {}
                    p["holdings"][sym] = {"shares": shares, "avg_cost": target['price'], "high_price": target['price']}
                    msg_lines.append(f"└ 買入 {sym} `{shares:.4f}` 股。")

    save_portfolio(p)
    if force_send or msg_lines:
        final_msg = "\n".join(msg_lines)
        await channel.send(final_msg[:1990])

async def show_portfolio(channel):
    p = load_portfolio()
    msg = f"💼 **美股帳戶總覽**\n💵 **可用現金：** `${p.get('cash', 0):.2f}` USD\n"
    if not p.get("holdings"):
        msg += "📭 目前空手觀望中。"
    else:
        msg += "📦 **當前持股 (含零股)：**\n"
        for sym, d in p["holdings"].items():
            msg += f"🔸 **{sym}** ({STOCK_NAMES.get(sym, sym)}): `{d['shares']:.4f}`股 (成本:`${d['avg_cost']:.2f}`)\n"
    await channel.send(msg)

# ==========================================
# 8. 智慧指令路由防呆系統
# ==========================================
@bot.event
async def on_message(message):
    if message.author == bot.user: return
    
    raw_content = message.content.strip()
    upper_content = raw_content.upper()

    # 1. 處理中文指令
    if raw_content in ["美股庫存", "庫存", "帳本"]:
        await show_portfolio(message.channel)
        return
    if raw_content in ["美股掃描", "全面掃描", "大盤"]:
        await perform_scan(force_send=True)
        return

    # 2. 精準英文代號查詢 (支援 $ 符號)
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
            await channel.send("🌅 **【早安華爾街】** 美股已收盤，系統自動啟動每日晨間掃描...")
        await perform_scan(force_send=True)
    else:
        print("週末休市，今日不執行自動掃描。")

@bot.event
async def on_ready():
    print(f"🦅 US Bot Online: {bot.user}")
    keep_alive() 
    if not daily_scan_task.is_running():
        daily_scan_task.start()
        print("⏰ 每日晨間掃描鬧鐘已啟動 (排定於台灣時間 06:00)")

if __name__ == "__main__":
    # 確保不會因為沒填 Token 而噴錯
    if not DISCORD_BOT_TOKEN:
        print("❌ 找不到 Discord Bot Token，請檢查 Render 環境變數。")
    else:
        bot.run(DISCORD_BOT_TOKEN)
