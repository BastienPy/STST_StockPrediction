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

    # ---------------------
    # 1. Read CSV & sort by date
    # ---------------------
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ---------------------
    # 2. Filter to ACL18 date range
    # ---------------------
    mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
    df = df.loc[mask].copy()
    df.reset_index(drop=True, inplace=True)
    
    # Check if the DataFrame is empty after filtering
    if df.empty:
        print(f"Warning: No data in file '{file_path}' for the date range {start_date} to {end_date}.")
        return df  # or raise an error if that is preferred

    # ---------------------
    # 3. Create scaled time-based features
    #    According to the paper/Table 2:
    #       Year   = year / 3000
    #       Month  = month / 12
    #       Day    = day_of_month / 31
    #       Weekday= day_of_week / 7
    # ---------------------
    df['Year']    = df['Date'].dt.year / 3000.0
    df['Month']   = df['Date'].dt.month / 12.0
    df['Day']     = df['Date'].dt.day / 31.0
    df['Weekday'] = df['Date'].dt.weekday / 7.0

    # ---------------------
    # 4. Compute Technical Indicators with pandas_ta
    #    We'll base each binary signal on thresholds from Table 2.
    # ---------------------

    # For convenience, define local Series
    high = df['High']
    low  = df['Low']
    close = df['Close']
    volume = df['Volume']

    # 4a) SMA (10, 30, 50, 200)
    df['SMA_10']   = ta.sma(close, length=10)
    df['SMA_30']   = ta.sma(close, length=30)
    df['SMA_50']   = ta.sma(close, length=50)
    df['SMA_200']  = ta.sma(close, length=200)

    # 4b) EMA (10, 30, 50, 200)
    df['EMA_10']   = ta.ema(close, length=10)
    df['EMA_30']   = ta.ema(close, length=30)
    df['EMA_50']   = ta.ema(close, length=50)
    df['EMA_200']  = ta.ema(close, length=200)

    # 4c) Momentum (N=10)
    df['MOM_10']   = ta.mom(close, length=10)

    # 4d) Stoch RSI
    #     "stochrsi" returns multiple columns. We get the K & D lines:
    #     By default: length=14, rsi_length=14, k=3, d=3
    stochrsi = ta.stochrsi(close, length=14, rsi_length=14, k=3, d=3)
    # stochrsi columns: 'STOCHRSIk_14_14_3_3', 'STOCHRSId_14_14_3_3'
    df['STOCHRSIk'] = stochrsi[f'STOCHRSIk_14_14_3_3']
    df['STOCHRSId'] = stochrsi[f'STOCHRSId_14_14_3_3']

    # 4e) Stochastic K, D (classical)
    #     By default: length=14, k=3, d=3
    stoch = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)
    # stoch columns: 'STOCHk_14_3_3', 'STOCHd_14_3_3'
    df['STOCHk'] = stoch['STOCHk_14_3_3']
    df['STOCHd'] = stoch['STOCHd_14_3_3']

    # 4f) MACD (fast=12, slow=26, signal=9)
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    # macd columns: 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'
    df['MACD'] = macd['MACD_12_26_9']      # The main MACD line
    df['MACD_signal'] = macd['MACDs_12_26_9']  # The signal line

    # 4g) CCI (length=14)
    df['CCI_14'] = ta.cci(high, low, close, length=14)

    # Pre-create the column 'MFI_14' as float (by assigning NaN)
    df['MFI_14'] = np.nan

    # 4h) MFI (length=14) – force conversion to a flattened float64 NumPy array
    df['MFI_14'] = np.nan  # This ensures the column is float (np.nan is a float)
    df['MFI_14'] = np.array(ta.mfi(high, low, close, volume, length=14), dtype='float64').ravel()


    # 4i) AD (Accumulation/Distribution)
    #     This is the Chaikin A/D line, cumulative.
    df['AD'] = ta.ad(high, low, close, volume)

    # 4j) OBV (On Balance Volume)
    df['OBV'] = ta.obv(close, volume)

    # 4k) ROC (length=10)
    df['ROC_10'] = ta.roc(close, length=10)

    # ---------------------
    # 5. Create Binary Signals (Table 2 logic)
    #    For those not mentioned, we keep the straightforward threshold:
    #       - SMA/EMA: 1 if Close > SMA/EMA, else 0
    #       - Momentum: 1 if MOM_10 > 0, else 0
    #       - AD: 1 if AD(t) >= AD(t-1)
    #       - OBV: 1 if OBV(t) >= OBV(t-1)
    #       - ROC: 1 if ROC_10 >= 0
    #
    #    For the signals you specifically mentioned (STOCHRSI, STOCHK, STOCHD,
    #    MACD, CCI, MFI), we now apply the textual logic from Table 2:
    #
    #    SIG_STOCHRSI: 1 if [SRSI(i) <= 25] OR [SRSI(i) > SRSI(i−1) AND SRSI(i) < 75]
    #    SIG_STOCH_K:  1 if STOK(i) > STOK(i−1)
    #    SIG_STOCH_D:  1 if STOD(i) > STOD(i−1)
    #    SIG_MACD:     1 if MACD_signal < MACD
    #    SIG_CCI:      1 if [CCI(i) <= 100] OR [CCI(i) > CCI(i−1)]
    #    SIG_MFI:      1 if [MFI(i) <= 20] OR [MFI(i) > MFI(i−1) AND MFI(i) < 80]
    # ---------------------
    df['SIG_SMA_10']   = (close > df['SMA_10']).astype(int)
    df['SIG_SMA_30']   = (close > df['SMA_30']).astype(int)
    df['SIG_SMA_50']   = (close > df['SMA_50']).astype(int)
    df['SIG_SMA_200']  = (close > df['SMA_200']).astype(int)

    df['SIG_EMA_10']   = (close > df['EMA_10']).astype(int)
    df['SIG_EMA_30']   = (close > df['EMA_30']).astype(int)
    df['SIG_EMA_50']   = (close > df['EMA_50']).astype(int)
    df['SIG_EMA_200']  = (close > df['EMA_200']).astype(int)

    df['SIG_MOM']      = (df['MOM_10'] > 0).astype(int)

    # -- StochRSI
    srsik = df['STOCHRSIk']
    srsik_shift = srsik.shift(1)
    cond_stochrsi_1 = (srsik <= 25)
    cond_stochrsi_2 = (srsik > srsik_shift) & (srsik < 75)
    df['SIG_STOCHRSI'] = np.where(cond_stochrsi_1 | cond_stochrsi_2, 1, 0)

    # -- Stochastic K, D
    stoch_k = df['STOCHk']
    stoch_d = df['STOCHd']
    df['SIG_STOCH_K'] = (stoch_k > stoch_k.shift(1)).astype(int)
    df['SIG_STOCH_D'] = (stoch_d > stoch_d.shift(1)).astype(int)

    # -- MACD
    df['SIG_MACD'] = (df['MACD_signal'] < df['MACD']).astype(int)

    # -- CCI
    cci = df['CCI_14']
    cci_shift = cci.shift(1)
    cond_cci_1 = (cci <= 100)
    cond_cci_2 = (cci > cci_shift)
    df['SIG_CCI'] = np.where(cond_cci_1 | cond_cci_2, 1, 0)

    # -- MFI
    mfi = df['MFI_14']
    mfi_shift = mfi.shift(1)
    cond_mfi_1 = (mfi <= 20)
    cond_mfi_2 = (mfi > mfi_shift) & (mfi < 80)
    df['SIG_MFI'] = np.where(cond_mfi_1 | cond_mfi_2, 1, 0)

    # AD: 1 if AD(t) >= AD(t-1)
    df['SIG_AD']       = (df['AD'] >= df['AD'].shift(1)).astype(int)

    # OBV: 1 if OBV(t) >= OBV(t-1)
    df['SIG_OBV']      = (df['OBV'] >= df['OBV'].shift(1)).astype(int)

    # ROC: 1 if >= 0
    df['SIG_ROC']      = (df['ROC_10'] >= df['ROC_10'].shift(1)).astype(int)

    # ---------------------
    # 6. Create the label y_t with thresholds:
    #    y_t = 0 if (AdjClose_{t+1} / AdjClose_t - 1) <= -0.5%
    #    y_t = 1 if (AdjClose_{t+1} / AdjClose_t - 1) >= 0.55%
    #    ignore (drop) if in between
    # ---------------------
    df['future_adj_close'] = df['Adj Close'].shift(-1)
    df['return_t'] = (df['future_adj_close'] / df['Adj Close']) - 1.0

    def label_function(x):
        if x <= -0.005:   # -0.5%
            return 0
        elif x >= 0.0055: # 0.55%
            return 1
        else:
            return np.nan  # neutral region

    df['label'] = df['return_t'].apply(label_function)

    # Drop rows where label is NaN (neutral zone) or where future_adj_close is NaN
    df.dropna(subset=['label'], inplace=True)

    # ---------------------
    # 7. (Optional) Drop early rows with NaNs from indicators
    #    e.g. from the large 200-day SMA.  It's up to you.
    # ---------------------
    df.dropna(inplace=True)

    return df


if __name__ == "__main__":
    # Example usage: process all CSVs in data/stocknet-dataset
    input_folder = "data/stocknet-dataset"

    for fname in os.listdir(input_folder):
        if fname.lower().endswith('.csv'):
            full_path = os.path.join(input_folder, fname)
            processed_df = process_acl18_csv(full_path)
            # Save the processed DataFrame to a new CSV in data\stocknet-dataset-processed
            out_file_path = os.path.join("data", "stocknet-dataset-processed", fname)
            processed_df.to_csv(out_file_path, index=False)

    # 'all_processed' is a list of (filename, processed DataFrame).
    # You can further combine them or feed them into your model.
