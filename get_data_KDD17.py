import yfinance as yf
import pandas as pd
import os

# liste des 50 tickers selon l'article (KDD17 - Zhang et al., 2017)
tickers = [
    "BHP", "DOW", "RIO", "SYT", "VALE",       
    "AMZN", "CMCSA", "DIS", "HD", "TM",       
    "CVX", "PTR", "RDS-B", "TOT", "XOM",      
    "BAC", "BRK-B", "JPM", "SPY", "WFC",     
    "JNJ", "MRK", "NVS", "PFE", "UNH",       
    "BA", "GE", "MA", "MMM", "UPS",           
    "KO", "MO", "PEP", "PG", "WMT",           
    "AAPL", "GOOGL", "INTC", "MSFT", "ORCL",  
    "CHL", "DCM", "NTT", "T", "VZ",           
    "D", "DUK", "EXC", "NGG", "SO"           
]

start_date = "2007-01-01"
end_date = "2016-12-31"
output_dir = "KDD"
os.makedirs(output_dir, exist_ok=True)

for ticker in tickers:
    
    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
        df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        df = df.dropna()
        df.reset_index(inplace=True)
        filename = f"{ticker.replace('^','').replace('/','-')}.csv"
        df.to_csv(os.path.join(output_dir, filename), index=False)
        print(f"Sauvegardé: {filename}")
        
    except Exception as e:
        print(f"Erreur pour {ticker} : {e}")
