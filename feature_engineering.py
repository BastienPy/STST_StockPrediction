import os
import numpy as np
import pandas as pd
import pandas_ta as ta
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def process_acl18_csv(
    file_path,
    start_date='2014-01-01',
    end_date='2016-01-01'
):
    """
    Reads a single stock CSV, computes the features described in Table 2
    using pandas_ta, and returns a processed DataFrame. Steps include:
      1) Filtering by date for ACL18 (Jan 1, 2014 - Jan 1, 2016).
      2) Creating Year/Month/Day/Weekday features (scaled).
      3) Computing 18 binary technical signals as in Table 2.
      4) Creating a binary label y_t using thresholds -0.5% and +0.55%.
         Rows in the neutral zone are dropped.
    """

    #Read CSV & sort by date
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    #Filter to ACL18 date range
    mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
    df = df.loc[mask].copy()
    df.reset_index(drop=True, inplace=True)
    
    #Check if the DataFrame is empty after filtering
    if df.empty:
        print(f"Warning: No data in file '{file_path}' for the date range {start_date} to {end_date}.")
        return df

    #Create scaled time-based features
    df['Year']    = df['Date'].dt.year / 3000.0
    df['Month']   = df['Date'].dt.month / 12.0
    df['Day']     = df['Date'].dt.day / 31.0
    df['Weekday'] = df['Date'].dt.weekday / 7.0

    #Compute Technical Indicators (Table 2)
    high = df['High']
    low  = df['Low']
    close = df['Close']
    volume = df['Volume']

    #SMA (10, 30, 50, 200)
    df['SMA_10']   = ta.sma(close, length=10)
    df['SMA_30']   = ta.sma(close, length=30)
    df['SMA_50']   = ta.sma(close, length=50)
    df['SMA_200']  = ta.sma(close, length=200)

    #EMA (10, 30, 50, 200)
    df['EMA_10']   = ta.ema(close, length=10)
    df['EMA_30']   = ta.ema(close, length=30)
    df['EMA_50']   = ta.ema(close, length=50)
    df['EMA_200']  = ta.ema(close, length=200)

    #Momentum (N=10)
    df['MOM_10']   = ta.mom(close, length=10)

    #Stoch RSI
    stochrsi = ta.stochrsi(close, length=14, rsi_length=14, k=3, d=3)
    df['STOCHRSIk'] = stochrsi[f'STOCHRSIk_14_14_3_3']
    df['STOCHRSId'] = stochrsi[f'STOCHRSId_14_14_3_3']

    #Stochastic K, D
    stoch = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)
    df['STOCHk'] = stoch['STOCHk_14_3_3']
    df['STOCHd'] = stoch['STOCHd_14_3_3']

    #MACD
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']      #The main MACD line
    df['MACD_signal'] = macd['MACDs_12_26_9']  #The signal line

    #CCI
    df['CCI_14'] = ta.cci(high, low, close, length=14)

    #MFI_14
    df['MFI_14'] = np.nan
    df['MFI_14'] = np.array(ta.mfi(high, low, close, volume, length=14), dtype='float64').ravel()


    #AD (Accumulation/Distribution)
    df['AD'] = ta.ad(high, low, close, volume)

    #OBV (On Balance Volume)
    df['OBV'] = ta.obv(close, volume)

    #ROC (length=10)
    df['ROC_10'] = ta.roc(close, length=10)

    #Create Binary Signals
    df['SIG_SMA_10']   = (close > df['SMA_10']).astype(int)
    df['SIG_SMA_30']   = (close > df['SMA_30']).astype(int)
    df['SIG_SMA_50']   = (close > df['SMA_50']).astype(int)
    df['SIG_SMA_200']  = (close > df['SMA_200']).astype(int)

    df['SIG_EMA_10']   = (close > df['EMA_10']).astype(int)
    df['SIG_EMA_30']   = (close > df['EMA_30']).astype(int)
    df['SIG_EMA_50']   = (close > df['EMA_50']).astype(int)
    df['SIG_EMA_200']  = (close > df['EMA_200']).astype(int)

    df['SIG_MOM']      = (df['MOM_10'] > 0).astype(int)

    #StochRSI
    srsik = df['STOCHRSIk']
    srsik_shift = srsik.shift(1)
    cond_stochrsi_1 = (srsik <= 25)
    cond_stochrsi_2 = (srsik > srsik_shift) & (srsik < 75)
    df['SIG_STOCHRSI'] = np.where(cond_stochrsi_1 | cond_stochrsi_2, 1, 0)

    #Stochastic K, D
    stoch_k = df['STOCHk']
    stoch_d = df['STOCHd']
    df['SIG_STOCH_K'] = (stoch_k > stoch_k.shift(1)).astype(int)
    df['SIG_STOCH_D'] = (stoch_d > stoch_d.shift(1)).astype(int)

    #MACD
    df['SIG_MACD'] = (df['MACD_signal'] < df['MACD']).astype(int)

    #CCI
    cci = df['CCI_14']
    cci_shift = cci.shift(1)
    cond_cci_1 = (cci <= 100)
    cond_cci_2 = (cci > cci_shift)
    df['SIG_CCI'] = np.where(cond_cci_1 | cond_cci_2, 1, 0)

    #MFI
    mfi = df['MFI_14']
    mfi_shift = mfi.shift(1)
    cond_mfi_1 = (mfi <= 20)
    cond_mfi_2 = (mfi > mfi_shift) & (mfi < 80)
    df['SIG_MFI'] = np.where(cond_mfi_1 | cond_mfi_2, 1, 0)

    #1 if AD(t) >= AD(t-1)
    df['SIG_AD']       = (df['AD'] >= df['AD'].shift(1)).astype(int)

    #1 if OBV(t) >= OBV(t-1)
    df['SIG_OBV']      = (df['OBV'] >= df['OBV'].shift(1)).astype(int)

    #1 if >= 0
    df['SIG_ROC']      = (df['ROC_10'] >= df['ROC_10'].shift(1)).astype(int)

    #Create the label y_t with thresholds:
    df['future_adj_close'] = df['Adj Close'].shift(-1)
    df['return_t'] = (df['future_adj_close'] / df['Adj Close']) - 1.0

    def label_function(x):
        if x <= -0.005:
            return 0
        elif x >= 0.0055:
            return 1
        else:
            return np.nan

    df['label'] = df['return_t'].apply(label_function)

    #Drop early rows with NaNs from indicators
    df.dropna(inplace=True)

    return df


if __name__ == "__main__":
    input_folder = "data/stocknet-dataset"

    for fname in os.listdir(input_folder):
        if fname.lower().endswith('.csv'):
            full_path = os.path.join(input_folder, fname)
            processed_df = process_acl18_csv(full_path)
            out_file_path = os.path.join("data", "stocknet-dataset-processed", fname)
            processed_df.to_csv(out_file_path, index=False)
