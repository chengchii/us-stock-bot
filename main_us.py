import os
import discord
from discord.ext import commands,tasks
import yfinance as yf
import pandas as pd
import asyncio
import json
import requests
from datetime import datetime,timedelta,time,timezone
import traceback
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from flask import Flask
from threading import Thread

# --- 偽裝成瀏覽器的 Session (防 Yahoo 阻擋) ---
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
})

# --- Render 存活檢查 ---
app = Flask('')
@app.route('/')
def home(): return "US Quant Bot: Wall Street Engine Online!"
def run(): app.run(host='0.0.0.0', port=10000)
def keep_alive(): Thread(target=run).start()

# ==========================================
# 1. 系統設定 (美金計價)
# ==========================================
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
CHANNEL_ID = os.environ.get("CHANNEL_ID")
INVEST_AMOUNT = 1000  # 每月入金 1000 美金
RUN_MODE = os.environ.get("RUN_MODE", "listen")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# ==========================================
# 2. 美股百大旗艦觀察名單
# ==========================================
# ==========================================
# 2. 自動抓取 NASDAQ 100 最新成分股名單
# ==========================================
def get_nasdaq_100_tickers():
    try:
        print("🌐 正在從維基百科下載最新的 NASDAQ 100 成分股名單...")
        # 維基百科的 NASDAQ 100 頁面
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        # pandas 內建的超強爬蟲，可以直接抓網頁裡的表格
        tables = pd.read_html(url)
        
        # NASDAQ 100 的成分股通常在維基百科頁面的第 4 個表格 (索引是 4，不過有時候會變)
        # 為了保險，我們寫個迴圈找一下哪個表格有 'Ticker' 欄位
        for table in tables:
            if 'Ticker' in table.columns:
                df = table
                break
        
        # 把 Ticker 欄位抽出來變成名單
        tickers = df['Ticker'].tolist()
        
        # 建立一個簡單的字典，因為維基百科上只有公司英文全名，有點長，我們這裡就簡單用代號當名稱
        names_dict = {}
        for idx, row in df.iterrows():
            sym = row['Ticker']
            # 取公司名稱的前面幾個字就好，以免太長
            name = str(row['Company']).split()[0].replace(',', '')
            names_dict[sym] = name
            
        return tickers, names_dict
    except Exception as e:
        print(f"❌ 抓取 NASDAQ 100 名單失敗: {e}")
        # 如果爬蟲失敗（例如維基百科改版），給一個超精簡的備用名單保底
        backup_list = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
        return backup_list, {s: s for s in backup_list}

# 在程式啟動時，自動執行一次抓取
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
# 4. YFinance 數據引擎
# ==========================================
async def fetch_single_stock_data(symbol, retries=2):
    loop = asyncio.get_running_loop()
    for attempt in range(retries):
        try:
            df = await loop.run_in_executor(None, lambda: yf.Ticker(symbol, session=session).history(period="1y"))
            if not df.empty and len(df) > 0:
                return symbol, df
        except Exception as e:
            pass
        await asyncio.sleep(1)
    return symbol, pd.DataFrame()

# ==========================================
# 5. 專業圖表引擎
# ==========================================
def generate_advanced_chart(df, symbol):
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.family'] = 'DejaVu Sans' # 美股不需要中文字體，用預設的即可
    
    ax1 = plt.subplot(2, 1, 1)
    ema200 = df['Close'].ewm(span=200, adjust=False).mean()
    plt.plot(df.index, df['Close'], label='Price', color='blue', alpha=0.6)
    plt.plot(df.index, ema200, label='200 EMA', color='red', linewidth=2)
    plt.title(f"Wall Street Tech Analysis: {symbol}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    macd = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    hist = macd - sig
    
    plt.plot(df.index, macd, label='MACD', color='blue')
    plt.plot(df.index, sig, label='Signal', color='orange')
    plt.bar(df.index, hist, label='Histogram', color='gray', alpha=0.3)
    plt.axhline(0, color='black', linewidth=1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    chart_path = f"analysis_{symbol}.png"
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()
    return chart_path

# ==========================================
# 6. 單股分析與零股倉位健檢
# ==========================================
async def process_stock_query(channel, symbol):
    initial_msg = await channel.send(f"🔍 正在對 `{symbol}` 進行美股策略分析與倉位精算...")
    try:
        _, df = await fetch_single_stock_data(symbol)
        if df.empty or len(df) < 200:
            await initial_msg.edit(content=f"❌ `{symbol}` 歷史數據不足或 API 無回應。")
            return
            
        p = load_portfolio()
        cash = p.get("cash", 0.0)
        holdings = p.get("holdings", {})
            
        close = df['Close']
        ema200 = close.ewm(span=200, adjust=False).mean()
        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        sig = macd.ewm(span=9, adjust=False).mean()
        
        curr_p = close.iloc[-1]
        curr_ema200 = ema200.iloc[-1]
        curr_macd = macd.iloc[-1]
        curr_sig = sig.iloc[-1]

        trend = "上升趨勢" if curr_p > curr_ema200 else "下降趨勢"
        is_long = (curr_p > curr_ema200) and (macd.iloc[-2] < sig.iloc[-2]) and (curr_macd > curr_sig) and (curr_macd < 0)
        
        chart_path = generate_advanced_chart(df, symbol)
        
        msg = (
            f"📊 **{STOCK_NAMES.get(symbol, symbol)} ({symbol}) 策略報告**\n"
            f"> 1. 【趨勢】：目前處於 **{trend}** (現價 `${curr_p:.2f}` vs 200 EMA `${curr_ema200:.2f}`)\n"
            f"> 2. 【MACD】：快線 `{curr_macd:.2f}` / 慢線 `{curr_sig:.2f}`\n"
            f"> 3. 【行動】：🚀 **{'強烈建議做多' if is_long else '未達進場標準，建議觀望'}**\n"
        )

        msg += "\n💼 **【帳戶與零股健檢】**\n"
        if symbol in holdings:
            data_p = holdings[symbol]
            shares = data_p["shares"]
            avg_cost = data_p["avg_cost"]
            profit_pct = ((curr_p - avg_cost) / avg_cost) * 100
            msg += f"> 📦 持有：`{shares:.4f}` 股 | 均價 `${avg_cost:.2f}` | 報酬 **`{profit_pct:.1f}%`**\n"
        else:
            if is_long:
                # 🌟 零股計算：計算到小數點後 4 位
                max_shares = round(cash / curr_p, 4) if cash > 0 else 0
                msg += f"> 💰 資金：可用現金 `${cash:.2f}`，最多可買入 **`{max_shares:.4f}` 股** (零股)。\n"
            else:
                 msg += "> 📭 狀態：目前未持有此檔股票。\n"

        await initial_msg.delete()
        with open(chart_path, 'rb') as f:
            await channel.send(content=msg, file=discord.File(f))
        os.remove(chart_path)
    except Exception as e:
        await initial_msg.edit(content=f"❌ 系統錯誤！\n```python\n{traceback.format_exc()}\n```")

# ==========================================
# 7. 美股自動掃描引擎 (SPY 大盤 + 零股均分 + 全列表顯示)
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
        msg_lines.append(f"🏦 **美股入金**：已存入 ${INVEST_AMOUNT} USD，現金：`${p['cash']:.2f}`")

    # 判定大盤 (SPY)
    _, df_market = await fetch_single_stock_data("SPY")
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
        scan_msg = await channel.send("🦅 美股大盤偏多！正在掃描 50+ 檔旗艦股，這大約需要 1~2 分鐘...")
        for s in WATCHLIST:
            _, df = await fetch_single_stock_data(s)
            if df.empty or len(df) < 200: continue
            
            close = df['Close']
            ema200 = close.ewm(span=200, adjust=False).mean()
            macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
            sig = macd.ewm(span=9, adjust=False).mean()
            
            # 策略：站上 200EMA + 零軸下金叉
            if (close.iloc[-1] > ema200.iloc[-1]) and (macd.iloc[-2] < sig.iloc[-2]) and (macd.iloc[-1] > sig.iloc[-1]) and (macd.iloc[-1] < 0):
                score = macd.iloc[-1] - sig.iloc[-1] # MACD 快慢線差值，越大代表動能越強
                results.append({'symbol': s, 'price': close.iloc[-1], 'score': score})
            await asyncio.sleep(0.5)
            
        try: await scan_msg.delete()
        except: pass

        # 🌟 排序並列出「所有」符合條件的股票
        results.sort(key=lambda x: x['score'], reverse=True)
        
        if results:
            msg_lines.append(f"\n🎯 **【零軸下黃金交叉清單】** (共 {len(results)} 檔)")
            for idx, r in enumerate(results):
                sym = r['symbol']
                name = STOCK_NAMES.get(sym, sym)
                # 前三名給予火炬圖示強烈推薦
                rec_icon = "🔥 強烈推薦" if idx < 3 else "✅ 符合標準"
                msg_lines.append(f"`{idx+1}.` **{sym}** ({name}) - 股價: `${r['price']:.2f}` | 評級: {rec_icon}")
        else:
            msg_lines.append("🔎 巡邏完畢，目前 **無任何一檔** 符合 MACD 買進訊號。")

    # 持股停利損檢查
    for sym, data_p in list(p.get("holdings", {}).items()):
        _, df = await fetch_single_stock_data(sym)
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

    # 🌟 執行買進：將資金均分給「所有」符合條件的股票 (零股交易)
    if results and p.get("cash", 0) > 10: # 只要現金大於 10 美金就繼續買
        budget = p["cash"] / len(results) # 資金均分給所有標的
        msg_lines.append(f"\n🛒 **【自動佈局 (零股)】** 預計每檔分配約 `${budget:.2f}`")
        
        for target in results:
            sym = target['symbol']
            if sym not in p.get("holdings", {}):
                shares = round(budget / target['price'], 4) # 計算零股，保留 4 位小數
                if shares > 0.0001:
                    p["cash"] -= shares * target['price']
                    if "holdings" not in p: p["holdings"] = {}
                    p["holdings"][sym] = {"shares": shares, "avg_cost": target['price'], "high_price": target['price']}
                    msg_lines.append(f"└ 買入 {sym} `{shares:.4f}` 股。")

    save_portfolio(p)
    if force_send or msg_lines:
        final_msg = "🦅 **【美股量化引擎執行報告】**\n" + "\n".join(msg_lines)
        await channel.send(final_msg[:2000])

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
# 8. 智慧指令路由
# ==========================================
@bot.event
async def on_message(message):
    if message.author == bot.user: return
    content = message.content.strip().upper()

    if content in ["美股庫存", "庫存", "帳本"]:
        await show_portfolio(message.channel)
        return
    if content in ["美股掃描", "全面掃描", "大盤"]:
        await perform_scan(force_send=True)
        return

    # 支援查詢代號 (例如 AAPL, TSLA)
    if content in STOCK_NAMES or content.isalpha() and len(content) <= 5: 
        await process_stock_query(message.channel, content)
        return

    await bot.process_commands(message)

# ==========================================
# 🌟 內建每日自動掃描鬧鐘
# ==========================================
# 設定執行時間：UTC 22:00 = 台灣時間早上 06:00
scan_time = time(hour=22, minute=0, tzinfo=timezone.utc)

@tasks.loop(time=scan_time)
async def daily_scan_task():
    # 台灣時間早上 6 點時，檢查今天是不是星期二到星期六 
    # (因為美股週一到週五的收盤，對應到台灣是週二到週六的清晨)
    # weekday() 回傳值: 0=週一, 1=週二 ... 5=週六, 6=週日
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
    keep_alive() # 啟動假網頁讓 UptimeRobot 監控
    
    # 🌟 啟動每日掃描鬧鐘
    if not daily_scan_task.is_running():
        daily_scan_task.start()
        print("⏰ 每日晨間掃描鬧鐘已啟動 (排定於台灣時間 06:00)")

if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
