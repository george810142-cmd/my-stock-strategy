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
import gc

# 忽略 pandas 的 FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# ⚙️ 設定區
# ==========================================
SHEET_NAME = 'AStock Overnight trading'
BATCH_SIZE = 50 
BACKTEST_PERIOD = "5y" 

CONFIG = {
    'MIN_PRICE': 2.0, 
    'MAX_PRICE': 1000.0,
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

FALLBACK_TICKERS = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA']

# ==========================================
# 1. 基礎函數 (Get Tickers, GSheet, Upload)
# ==========================================
@st.cache_data(ttl=3600*24)
def get_sp500_tickers():
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        response = requests.get(url)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            tickers = df['Symbol'].tolist()
            clean_tickers = [t.replace('.', '-') for t in tickers]
            return clean_tickers
        return FALLBACK_TICKERS
    except: return FALLBACK_TICKERS

def connect_to_gsheet():
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        if "gcp_service_account" not in st.secrets:
            st.error("❌ 未偵測到 Secrets。")
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
# 2. 策略核心邏輯
# ==========================================
def process_batch_strategy(data, spy, market_signal):
    batch_candidates = []
    tickers = data.columns.levels[0].tolist()

    for ticker in tickers:
        try:
            df = data[ticker].copy().dropna()
            if len(df) < 60: continue 
            if df.index.tz is not None: df.index = df.index.tz_localize(None)

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
                
                if loc + 5 < len(df):
                    week_data = df.iloc[loc:loc+5]
                    hit_stop = week_data['Low'] <= stop_price
                    if hit_stop.any():
                        status, sell_price, sell_date = "StopLoss", stop_price, week_data[hit_stop].index[0]
                    else:
                        status, sell_price, sell_date = "Closed", float(df.iloc[loc+5]['Open']), df.iloc[loc+5].name
                else:
                    days_passed = df.iloc[loc:]
                    hit_stop = days_passed['Low'] <= stop_price
                    if hit_stop.any():
                        status, sell_price, sell_date = "StopLoss", stop_price, days_passed[hit_stop].index[0]
                    else:
                        status, sell_date, sell_price = "HOLD", "HOLDING", float(df.iloc[-1]['Close']) 

                batch_candidates.append({
                    'Ticker': ticker, 'Buy_Date': buy_date, 'Buy_Price': round(buy_price, 2),
                    'Sell_Date': sell_date, 'Sell_Price': round(sell_price, 2),
                    'Profit': round(sell_price - buy_price, 2), 
                    'Return_Pct': round(((sell_price - buy_price)/buy_price) * 100, 2),
                    'Status': status, 'RVol': round(monday_open['RVol'], 2)
                })
        except Exception: continue
    return batch_candidates

def predict_next_week(tickers, spy):
    candidates = []
    spy_ma = spy['Close'].rolling(CONFIG['MARKET_FILTER_MA']).mean().iloc[-1]
    if spy['Close'].iloc[-1] < spy_ma:
        return pd.DataFrame()

    for i in range(0, len(tickers), BATCH_SIZE * 2):
        chunk = tickers[i:i + BATCH_SIZE * 2]
        try:
            data = yf.download(chunk, period="3mo", group_by='ticker', auto_adjust=False, threads=True, progress=False)
            if data.empty: continue
            if len(chunk) > 1: data = data.dropna(axis=1, how='all')
            current_tickers = data.columns.levels[0].tolist() if isinstance(data.columns, pd.MultiIndex) else chunk

            for ticker in current_tickers:
                try:
                    df = data[ticker].dropna() if isinstance(data.columns, pd.MultiIndex) else data.dropna()
                    if df.empty: continue
                    curr = df.iloc[-1]
                    close, volume = curr['Close'], curr['Volume']
                    if not (CONFIG['MIN_PRICE'] <= close <= CONFIG['MAX_PRICE']): continue
                    if volume <= CONFIG['MIN_VOLUME']: continue
                    
                    vol_ma20 = df['Volume'].rolling(20).mean().iloc[-1]
                    rvol = volume / vol_ma20 if vol_ma20 > 0 else 0
                    if rvol <= CONFIG['MIN_RVOL']: continue
                    
                    mom = (close - df['Close'].shift(20).iloc[-1]) / df['Close'].shift(20).iloc[-1]
                    if not (CONFIG['MIN_MOMENTUM'] <= mom <= CONFIG['MAX_MOMENTUM']): continue
                    
                    candidates.append({
                        'Ticker': ticker, 'Close': close, 'RVol': round(rvol, 2),
                        'Momentum': round(mom*100, 2)
                    })
                except: continue
            del data
            gc.collect()
        except: continue

    df_next = pd.DataFrame(candidates)
    if not df_next.empty:
        return df_next.sort_values(by='RVol', ascending=False).head(5)
    return pd.DataFrame()

# ==========================================
# 🚀 主頁面 (邏輯修改重點區)
# ==========================================
st.title("📈 V60 美股策略儀表板 (SP500 Pro)")
st.caption(f"Mode: Batch Processing | Period: {BACKTEST_PERIOD}")

# 1. 初始化 Session State (讓資料可以跨越 Rerun 存活)
if 'df_history' not in st.session_state:
    st.session_state['df_history'] = None
if 'df_next' not in st.session_state:
    st.session_state['df_next'] = None

# 2. 只有按下按鈕時，才進行「運算」並更新 Session State
if st.button("🚀 開始執行全市場掃描"):
    
    with st.spinner("📥 下載大盤數據中..."):
        spy = yf.download("SPY", period=BACKTEST_PERIOD, progress=False, auto_adjust=False)
        if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
        spy_ma = spy['Close'].rolling(CONFIG['MARKET_FILTER_MA']).mean()
        market_signal = (spy['Close'] > spy_ma).to_dict()
    
    tickers = get_sp500_tickers()
    st.info(f"📋 鎖定 S&P 500 共 {len(tickers)} 檔股票...")

    all_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_batches = (len(tickers) // BATCH_SIZE) + 1
    
    for i in range(0, len(tickers), BATCH_SIZE):
        chunk = tickers[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        status_text.text(f"🔄 處理中: 批次 {batch_num}/{total_batches}...")
        progress_bar.progress(i / len(tickers))
        
        try:
            batch_data = yf.download(chunk, period=BACKTEST_PERIOD, group_by='ticker', auto_adjust=False, threads=True, progress=False)
            if not batch_data.empty:
                batch_results = process_batch_strategy(batch_data, spy, market_signal)
                all_results.extend(batch_results)
            del batch_data
            gc.collect()
        except: continue

    progress_bar.progress(100)
    status_text.success("✅ 運算完成！")

    # 彙整結果並存入 session_state
    if all_results:
        df_all = pd.DataFrame(all_results)
        df_hist = df_all.sort_values(by=['Buy_Date', 'RVol'], ascending=[True, False]) \
                           .groupby('Buy_Date').head(CONFIG['HOLDING_COUNT']).reset_index(drop=True)
        st.session_state['df_history'] = df_hist.sort_values(by='Buy_Date', ascending=False)
    else:
        st.session_state['df_history'] = pd.DataFrame()

    # 下週預測並存入 session_state
    with st.spinner("正在掃描下週標的..."):
        st.session_state['df_next'] = predict_next_week(tickers, spy)

# ==========================================
# 3. 顯示與上傳區 (獨立於按鈕之外)
# ==========================================
# 這裡會檢查 Session State 裡有沒有資料，如果有就顯示
# 這樣就算你打勾觸發了 Rerun，資料也不會不見

if st.session_state['df_history'] is not None:
    
    df_history = st.session_state['df_history']
    df_next = st.session_state['df_next']

    # --- 顯示歷史 ---
    st.subheader("📜 5年歷史回測紀錄")
    if not df_history.empty:
        st.dataframe(df_history)
        total_ret = df_history['Return_Pct'].sum()
        win_rate = (df_history['Profit'] > 0).mean() * 100
        st.metric("歷史總獲利 %", f"{total_ret:.2f}%", delta=f"勝率 {win_rate:.0f}%")
    else:
        st.warning("⚠️ 無歷史交易訊號。")

    # --- 顯示下週 ---
    st.subheader("🔮 下週一潛在標的")
    if df_next is not None and not df_next.empty:
        st.dataframe(df_next)
    else:
        st.info("無符合標的。")

    # --- 上傳按鈕 (改用 Button 比 Checkbox 更直覺) ---
    st.write("---")
    if st.button("📤 上傳結果到 Google Sheet"):
        sheet = connect_to_gsheet()
        if sheet:
            if not df_history.empty: 
                upload_dataframe(sheet, "V60_SP500_5Y", df_history)
            else:
                st.write("歷史紀錄為空，跳過上傳。")
            
            if df_next is not None and not df_next.empty: 
                upload_dataframe(sheet, "V60_Next_Week", df_next)
            else:
                st.write("下週清單為空，跳過上傳。")
