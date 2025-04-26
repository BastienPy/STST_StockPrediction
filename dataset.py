import os
import glob
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore", category=torch.serialization.SourceChangeWarning)

# ----------------------------------------------------------------
#  1) Import your pretrained Date2Vec model
# ----------------------------------------------------------------
from Date2Vec.Model import Date2VecConvert

# ----------------------------------------------------------------
#  2) SpatiotemporalEmbed module replicates Figure 2 exactly
# ----------------------------------------------------------------
class SpatiotemporalEmbed(nn.Module):
    """
    Implements the 'spatiotemporal embedding' from Figure 2 of the paper.
    """
    def __init__(self, window_size: int, f: int, time_emb_dim: int, d: int):
        super().__init__()
        self.window_size = window_size
        self.f = f
        self.time_emb_dim = time_emb_dim
        self.d = d
        # Now expect only one timestep's f features + time embedding
        self.dense = nn.Linear(f + time_emb_dim, d)

    def forward(self, X_f: torch.Tensor, X_t: torch.Tensor, date2vec):
        N, f = X_f.shape
        assert N == self.window_size and f == self.f

        E_list = []
        for i in range(N):
            # per-timestep spatial features
            f_i = X_f[i]                       # shape: (f,)

            # get that timestep's time embedding
            time_row = X_t[i].unsqueeze(0)     # shape: (1, time_input_dim)
            with torch.no_grad():
                t_embed = date2vec(time_row)   # shape: (1, time_emb_dim)
            t_embed = t_embed.view(-1)        # shape: (time_emb_dim,)

            # concatenate and project
            X_c_i = torch.cat([f_i, t_embed], dim=0)  # shape: (f + time_emb_dim,)
            E_list.append(self.dense(X_c_i))          # shape: (d,)

        E_X = torch.stack(E_list, dim=0)  # shape: (N, d)
        return E_X


# ----------------------------------------------------------------
#  3) Define a Dataset that supports caching of preprocessed sequences.
# ----------------------------------------------------------------
class StockDataset(Dataset):
    """
    Loads processed CSVs and builds rolling windows of size N.
    Windows are split by the last date in the window into:
      - Train:      [2014-01-01, 2015-08-08)
      - Validation: [2015-08-08, 2015-10-01)
      - Test:       [2015-10-01,  2016-01-01)
    
    If a cache file is provided and exists, the preprocessed sequences
    are loaded from disk rather than re-computed.
    """
    def __init__(
        self,
        folder_processed_csv: str,
        window_size: int,
        date2vec_model_path: str,
        non_time_feature_cols: list,
        time_embed_dim: int = 64,
        out_embed_dim: int = 128,
        split: str = "train",  # "train", "val", or "test"
        use_all_stocks: bool = True,
        cache_file: str = None
    ):
        super().__init__()
        self.window_size = window_size
        self.non_time_cols = non_time_feature_cols
        self.f = len(non_time_feature_cols)
        self.time_embed_dim = time_embed_dim
        self.out_embed_dim = out_embed_dim
        self.split = split.lower().strip()  # "train", "val", or "test"
        self.cache_file = cache_file

        # Date boundaries for splits
        self.train_end_date = pd.to_datetime("2015-08-08")
        self.val_end_date   = pd.to_datetime("2015-10-01")
        self.test_end_date  = pd.to_datetime("2016-01-01")

        # Load the pretrained Date2Vec model
        self.d2v = Date2VecConvert(model_path=date2vec_model_path)

        # Instantiate the spatiotemporal embed module
        self.st_embed = SpatiotemporalEmbed(
            window_size=self.window_size,
            f=self.f,
            time_emb_dim=self.time_embed_dim,
            d=self.out_embed_dim
        )

        # If cache_file is provided and exists, load the sequences.
        if cache_file is not None and os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            with open(cache_file, "rb") as f:
                self.sequences = pickle.load(f)
        else:
            # Otherwise, process CSVs to build the sequences.
            print(f"No cached data")
            csv_paths = glob.glob(os.path.join(folder_processed_csv, "*.csv"))
            if not csv_paths:
                raise FileNotFoundError(f"No CSV found in {folder_processed_csv}.")
            self.data = []
            for path in csv_paths:
                if use_all_stocks:
                    df = pd.read_csv(path, parse_dates=["Date"])
                    if not df.empty:
                        df["StockSymbol"] = os.path.splitext(os.path.basename(path))[0]
                        self.data.append(df)
                else:
                    # Adapt for a subset if needed.
                    pass
            self.data = pd.concat(self.data, ignore_index=True)
            self.data.sort_values(["StockSymbol", "Date"], inplace=True)
            self.data.reset_index(drop=True, inplace=True)

            all_sequences = self.build_sequences(self.data)
            # Filter sequences by the split based on last date
            self.sequences = []
            for seq in all_sequences:
                last_date = seq[2]
                if self.belongs_to_split(last_date):
                    self.sequences.append((seq[0], seq[1]))
            if cache_file is not None:
                with open(cache_file, "wb") as f:
                    pickle.dump(self.sequences, f)
                print(f"Cached preprocessed data to {cache_file}")

    def belongs_to_split(self, date: pd.Timestamp) -> bool:
        if self.split == "train":
            return date < self.train_end_date
        elif self.split == "val":
            return (date >= self.train_end_date) and (date < self.val_end_date)
        elif self.split == "test":
            return (date >= self.val_end_date) and (date < self.test_end_date)
        else:
            raise ValueError(f"Unknown split={self.split}")

    def build_sequences(self, df: pd.DataFrame):
        sequences = []
        for symbol, grp in df.groupby("StockSymbol"):
            grp = grp.sort_values("Date").reset_index(drop=True)
            for i in range(len(grp) - self.window_size + 1):
                window_df = grp.iloc[i : i + self.window_size]
                y = window_df["label"].iloc[-1]
                last_date = window_df["Date"].iloc[-1]
                X_f_list = []
                X_t_list = []
                for row_i in range(self.window_size):
                    row_data = window_df.iloc[row_i]
                    # spatial features (unchanged)
                    fvals = [float(row_data[col]) for col in self.non_time_cols]
                    X_f_list.append(fvals)
                    # use the scaled time features you already computed in feature_engineering.py
                    # new: pad up to 6 dims so Date2Vec’s fc1 can multiply
                    X_t_list.append([
                        float(row_data["Year"]),    # 1: year/3000
                        float(row_data["Month"]),   # 2: month/12
                        float(row_data["Day"]),     # 3: day/31
                        float(row_data["Weekday"]), # 4: weekday/7
                        0.0,                        # 5: dummy
                        0.0                         # 6: dummy
                    ])
                X_f_np = np.array(X_f_list, dtype=np.float32)
                X_t_np = np.array(X_t_list, dtype=np.float32)

                # ─── Normalisation ───
                # on centre-réduit chaque colonne de X_f_np
                mean = X_f_np.mean(axis=0, keepdims=True)    # shape (1, f)
                std  = X_f_np.std(axis=0,  keepdims=True)    # shape (1, f)
                X_f_np = (X_f_np - mean) / (std + 1e-6)       # évite la div. par zéro
                
                X_f_torch = torch.from_numpy(X_f_np)
                X_t_torch = torch.from_numpy(X_t_np)
                with torch.no_grad():
                    E_X = self.st_embed(X_f_torch, X_t_torch, self.d2v)
                sequences.append((E_X.cpu().numpy(), y, last_date))
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        E_X_np, y = self.sequences[idx][0], self.sequences[idx][1]
        E_X_torch = torch.tensor(E_X_np, dtype=torch.float)
        y_torch   = torch.tensor(y, dtype=torch.long)
        return E_X_torch, y_torch

# ----------------------------------------------------------------
#  4) Simple collate function
# ----------------------------------------------------------------
def collate_fn(batch):
    E_list, y_list = zip(*batch)
    E_tensor = torch.stack(E_list, dim=0)
    y_tensor = torch.stack(y_list, dim=0)
    return E_tensor, y_tensor

# ----------------------------------------------------------------
#  5) Example usage
# ----------------------------------------------------------------
if __name__ == "__main__":
    folder_processed = "data/stocknet-dataset-processed"
    date2vec_path = "Date2Vec/d2v_model/d2v_98291_17.169918439404636.pth"
    window_size = 32
    non_time_feature_cols = [
    "Open", "High", "Low", "Close", "Adj Close", "Volume",
    *[f"SIG_SMA_{i}" for i in (10, 30, 50, 200)],
    *[f"SIG_EMA_{i}" for i in (10, 30, 50, 200)],
    "SIG_MOM", "SIG_STOCHRSI", "SIG_STOCH_K", "SIG_STOCH_D",
    "SIG_MACD", "SIG_CCI", "SIG_MFI", "SIG_AD", "SIG_OBV", "SIG_ROC"]
    out_embed_dim = 64

    # Example: load the training split using a cache file.
    train_dataset = StockDataset(
        folder_processed_csv=folder_processed,
        window_size=window_size,
        date2vec_model_path=date2vec_path,
        non_time_feature_cols=non_time_feature_cols,
        time_embed_dim=64,
        out_embed_dim=out_embed_dim,
        split="train",
        use_all_stocks=True,
        cache_file="cache_train.pkl"
    )
    
    print("Training samples:", len(train_dataset))
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    for E_batch, y_batch in train_loader:
        print("E_batch shape:", E_batch.shape)
        print("y_batch shape:", y_batch.shape)
        break
