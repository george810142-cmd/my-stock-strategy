import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import warnings
from datetime import datetime, timedelta
import io
import requests

# 忽略 pandas 的 FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# ⚙️ 設定區
# ==========================================
SHEET_NAME = 'AStock Overnight trading'

# ⚠️⚠️⚠️ 記憶體安全開關 ⚠️⚠️⚠️
# True = 只跑 S&P 500 前 100 大權重股 (建議！雲端免費版才跑得動)
# False = 跑完整 500 檔 (極可能導致 Streamlit Cloud 記憶體不足崩潰)
LIMIT_TOP_100 = True 

# 設定回測時間
BACKTEST_PERIOD = "5y" 

CONFIG = {
    'MIN_PRICE': 2.0, 
    'MAX_PRICE': 500.0, # 放寬價格上限以包含大型股
    'MIN_VOLUME': 800000,
    'MARKET_FILTER_MA': 50, 
    'MARKET_FILTER': True,
    'MIN_RVOL': 2.5, 
    'MIN_RSI': 50,
    'MIN_MOMENTUM': 0.00, 
    'MAX_MOMENTUM': 0.25,
    'USE_MA60_FILTER': True, 
    'REQUIRE_GREEN_CANDLE': True,
    'STRONG_CLOSE_RATIO': 0.70, 
    'USE_VWAP_FILTER': True,
    'STOP_LOSS_PCT': -0.07,
    'HOLDING_COUNT': 3, 
    'HOLDING_DAYS': 5
}

# 預設備用清單 (萬一抓不到 SP500 時使用)
FALLBACK_TICKERS = [
    'AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'LLY', 'AVGO', 
    'V', 'JPM', 'XOM', 'WMT', 'UNH', 'MA', 'PG', 'COST', 'JNJ', 'HD', 'MRK', 'ORCL', 
    'CVX', 'ABBV', 'BAC', 'CRM', 'KO', 'NFLX', 'AMD', 'PEP', 'TMO', 'LIN', 'WFC', 
    'ADBE', 'DIS', 'MCD', 'CSCO', 'ACN', 'ABT', 'QCOM', 'CAT', 'INTU', 'GE', 'AMAT', 
    'TXN', 'DHR', 'VZ', 'IBM', 'PM', 'AMGN', 'NOW', 'ISRG', 'UBER', 'PFE', 'GS'
]

# ==========================================
# 1. 取得 S&P 500 成分股 (改用 CSV 以避免依賴問題)
# ==========================================
@st.cache_data(ttl=3600*24)
def get_sp500_tickers():
    try:
        # 使用 Slickcharts 或 GitHub 的公開 CSV 來源 (這裡使用 GitHub 上的可靠來源)
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        response = requests.get(url)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            tickers = df['Symbol'].tolist()
            # 確保格式正確 (例如 BRK.B -> BRK-B)
            clean_tickers = [t.replace('.', '-') for t in tickers]
            
            if LIMIT_TOP_100:
                return clean_tickers[:100] # 只回傳前 100 檔
            return clean_tickers
        else:
            st.warning("⚠️ 無法下載 S&P 500 清單，使用備用清單。")
            return FALLBACK_TICKERS
    except Exception as e:
        st.error(f"⚠️ 下載成分股清單失敗: {e}，使用備用清單。")
        return FALLBACK_TICKERS

# ==========================================
# 2. Google Sheet 連線
# ==========================================
def connect_to_gsheet():
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        if "gcp_service_account" not in st.secrets:
            st.error("❌ 未偵測到 Secrets，無法連線 Sheet。")
            return None
        key_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(key_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open(SHEET_NAME)
        return sheet
    except Exception as e:
        st.error(f"Google Sheet 連線失敗: {e}")
        return None

def upload_dataframe(sheet, tab_name, df):
    if sheet is None: return
    try:
        try: worksheet = sheet.worksheet(tab_name)
        except: worksheet = sheet.add_worksheet(title=tab_name, rows="5000", cols="20")
        worksheet.clear()
        df_clean = df.fillna('').astype(str)
        data = [df_clean.columns.values.tolist()] + df_clean.values.tolist()
        worksheet.update(range_name='A1', values=data)
        st.success(f"✅ 上傳成功: [{tab_name}] - 共 {len(df)} 筆")
    except Exception as e: st.error(f"❌ 上傳失敗: {e}")

# ==========================================
# 3. 數據獲取 (批次處理以節省記憶體)
# ==========================================
@st.cache_data(ttl=3600*4)
def get_data(tickers):
    with st.spinner(f'📥 正在下載 {len(tickers)} 檔股票的 {BACKTEST_PERIOD} 數據...'):
        # 1. 下載 SPY
        spy = yf.download("SPY", period=BACKTEST_PERIOD, progress=False, auto_adjust=False)
        if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
        
        # 2. 下載個股
        # 使用 threads=True 加速，但這也是記憶體殺手
        data = yf.download(tickers, period=BACKTEST_PERIOD, group_by='ticker', auto_adjust=False, threads=True)
        
        if not data.empty:
             data = data.dropna(axis=1, how='all')
        
        return data, spy

# ==========================================
# 4. 策略邏輯
# ==========================================
def run_strategy(data, spy):
    status_text = st.empty()
    status_text.text("🧠 執行 V60 策略運算中 (這可能需要一點時間)...")
    
    spy_ma = spy['Close'].rolling(CONFIG['MARKET_FILTER_MA']).mean()
    market_signal = (spy['Close'] > spy_ma).to_dict()

    tickers = data.columns.levels[0].tolist()
    all_candidates = []

    progress_bar = st.progress(0)
    
    for i, ticker in enumerate(tickers):
        # 每 10 檔更新一次進度條，減少前端負擔
        if i % 10 == 0: progress_bar.progress((i + 1) / len(tickers))
        
        try:
            df = data[ticker].copy().dropna()
            if len(df) < 60: continue 
            if df.index.tz is not None: df.index = df.index.tz_localize(None)

            # 指標計算
            df['MA60'] = df['Close'].rolling(60).mean()
            df['VolMA20'] = df['Volume'].rolling(20).mean().replace(0, 1)
            df['RVol'] = df['Volume'] / df['VolMA20']
            df['Close_20d'] = df['Close'].shift(20)
            df['Momentum_20d'] = (df['Close'] - df['Close_20d']) / df['Close_20d']
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            df['Range'] = df['High'] - df['Low']
            df['Close_Loc'] = np.where(df['Range'] > 0, (df['Close'] - df['Low']) / df['Range'], 0.5)
            df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['Weekday'] = df.index.dayofweek

            # 篩選條件
            condition = (
                (df['Weekday'] == 0) & 
                (df['Close'] >= CONFIG['MIN_PRICE']) & (df['Close'] <= CONFIG['MAX_PRICE']) & 
                (df['Volume'] > CONFIG['MIN_VOLUME']) &
                (df['Close'] > df['MA60']) & 
                (df['Momentum_20d'] >= CONFIG['MIN_MOMENTUM']) & (df['Momentum_20d'] <= CONFIG['MAX_MOMENTUM']) & 
                (df['RVol'] > CONFIG['MIN_RVOL']) &
                (df['RSI'] > CONFIG['MIN_RSI']) &
                (df['Close'] > df['Open']) & 
                (df['Close_Loc'] > CONFIG['STRONG_CLOSE_RATIO']) & 
                (df['Close'] > df['Typical_Price'])
            )
            
            dates = df.index[condition]
            
            for date in dates:
                if not market_signal.get(date, False): continue
                loc = df.index.get_loc(date)
                monday_open = df.iloc[loc]
                buy_date = monday_open.name
                buy_price = float(monday_open['Open'])
                stop_price = buy_price * (1 + CONFIG['STOP_LOSS_PCT'])
                
                sell_date, sell_price, status = None, 0.0, ""
                
                # A. 歷史交易
                if loc + 5 < len(df):
                    week_data = df.iloc[loc:loc+5]
                    hit_stop = week_data['Low'] <= stop_price
                    if hit_stop.any():
                        status, sell_price = "StopLoss", stop_price
                        sell_date = week_data[hit_stop].index[0]
                    else:
                        status = "Closed"
                        next_monday = df.iloc[loc+5]
                        sell_date = next_monday.name
                        sell_price = float(next_monday['Open'])
                
                # B. 持倉中
                else:
                    days_passed = df.iloc[loc:]
                    hit_stop = days_passed['Low'] <= stop_price
                    if hit_stop.any():
                        status, sell_price = "StopLoss", stop_price
                        sell_date = days_passed[hit_stop].index[0]
                    else:
                        status, sell_date = "HOLD", "HOLDING"
                        sell_price = float(df.iloc[-1]['Close']) 

                pnl = sell_price - buy_price
                ret_pct = pnl / buy_price

                all_candidates.append({
                    'Ticker': ticker, 'Buy_Date': buy_date, 'Buy_Price': round(buy_price, 2),
                    'Sell_Date': sell_date, 'Sell_Price': round(sell_price, 2),
                    'Profit': round(pnl, 2), 'Return_Pct': round(ret_pct * 100, 2),
                    'Status': status, 'RVol': round(monday_open['RVol'], 2)
                })

        except Exception: continue
    
    status_text.empty()
    progress_bar.empty()
        
    if not all_candidates: return pd.DataFrame()
    
    df_all = pd.DataFrame(all_candidates)
    df_history = df_all.sort_values(by=['Buy_Date', 'RVol'], ascending=[True, False]) \
                        .groupby('Buy_Date').head(CONFIG['HOLDING_COUNT']).reset_index(drop=True)
    
    return df_history.sort_values(by='Buy_Date', ascending=False)

def predict_next_week(data, spy):
    candidates = []
    tickers = data.columns.levels[0].tolist()
    
    spy_ma = spy['Close'].rolling(CONFIG['MARKET_FILTER_MA']).mean().iloc[-1]
    if spy['Close'].iloc[-1] < spy_ma:
        st.warning("🛑 大盤紅燈 (SPY < MA50)，策略建議下週空手。")
        return pd.DataFrame()

    for ticker in tickers:
        try:
            df = data[ticker].dropna()
            if df.empty: continue
            curr = df.iloc[-1]
            
            close, volume = curr['Close'], curr['Volume']
            if not (CONFIG['MIN_PRICE'] <= close <= CONFIG['MAX_PRICE']): continue
            if volume <= CONFIG['MIN_VOLUME']: continue
            if close <= df['Close'].rolling(60).mean().iloc[-1]: continue
            
            vol_ma20 = df['Volume'].rolling(20).mean().iloc[-1]
            rvol = volume / vol_ma20 if vol_ma20 > 0 else 0
            if rvol <= CONFIG['MIN_RVOL']: continue
            
            mom = (close - df['Close'].shift(20).iloc[-1]) / df['Close'].shift(20).iloc[-1]
            if not (CONFIG['MIN_MOMENTUM'] <= mom <= CONFIG['MAX_MOMENTUM']): continue
            
            delta = df['Close'].diff()
            rs = (delta.where(delta > 0, 0)).rolling(14).mean().iloc[-1] / (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
            rsi = 100 - (100 / (1 + rs))
            if rsi <= CONFIG['MIN_RSI']: continue
            
            typical = (curr['High'] + curr['Low'] + close) / 3
            rng = curr['High'] - curr['Low']
            loc = (close - curr['Low']) / rng if rng > 0 else 0.5
            
            if close <= curr['Open'] or close <= typical or loc <= CONFIG['STRONG_CLOSE_RATIO']: continue
            
            candidates.append({
                'Ticker': ticker, 'Close': close, 'RVol': round(rvol, 2),
                'RSI': round(rsi, 2), 'Momentum': round(mom*100, 2)
            })
        except: continue
        
    df_next = pd.DataFrame(candidates)
    if not df_next.empty:
        return df_next.sort_values(by='RVol', ascending=False).head(5)
    return pd.DataFrame()

# ==========================================
# 🚀 主頁面
# ==========================================
st.title("📈 V60 美股策略儀表板")
st.caption(f"Universe: {'S&P 100 (Safe Mode)' if LIMIT_TOP_100 else 'S&P 500 (Full)'} | Period: {BACKTEST_PERIOD}")

if st.button("🚀 開始執行策略掃描"):
    
    # 0. 獲取股票清單
    tickers = get_sp500_tickers()
    st.info(f"📋 目標股票池：共 {len(tickers)} 檔")

    # 1. 獲取資料
    data, spy = get_data(tickers)
    st.success(f"資料下載完成！準備進行運算...")

    # 2. 執行策略
    df_history = run_strategy(data, spy)
    
    # 3. 顯示歷史紀錄
    st.subheader("📜 歷史回測紀錄")
    if not df_history.empty:
        st.dataframe(df_history)
        
        # 簡易分析
        total_ret = df_history['Return_Pct'].sum()
        win_rate = (df_history['Profit'] > 0).mean() * 100
        st.metric("歷史總獲利 %", f"{total_ret:.2f}%", delta=f"勝率 {win_rate:.0f}%")
    else:
        st.warning("⚠️ 在這段期間內沒有觸發任何訊號。")

    # 4. 預測下週
    st.subheader("🔮 下週一潛在標的")
    df_next = predict_next_week(data, spy)
    
    if not df_next.empty:
        st.dataframe(df_next)
    else:
        st.info("🔍 目前沒有符合下週進場條件的標的。")

    # 5. 上傳 Google Sheet
    if st.checkbox("📤 上傳結果到 Google Sheet?"):
        sheet = connect_to_gsheet()
        if sheet:
            if not df_history.empty: 
                upload_dataframe(sheet, "V60_History_5Y", df_history)
            
            if not df_next.empty: 
                upload_dataframe(sheet, "V60_Action_List", df_next)
