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
from Date2Vec.Model import Date2VecConvert

class SpatiotemporalEmbed(nn.Module):
    def __init__(self, window_size: int, f: int, time_emb_dim: int, d: int):
        super().__init__()
        self.window_size = window_size
        self.f = f
        self.time_emb_dim = time_emb_dim
        self.d = d
        self.dense = nn.Linear(f + time_emb_dim, d)

    def forward(self, X_f: torch.Tensor, X_t: torch.Tensor, date2vec):
        N, f = X_f.shape
        assert N == self.window_size and f == self.f

        E_list = []
        for i in range(N):
            # caractéristiques spatiales pour chaque pas de temps
            f_i = X_f[i]                      
            # obtenir l'encodage temporel pour ce pas de temps
            time_row = X_t[i].unsqueeze(0)     
            with torch.no_grad():
                t_embed = date2vec(time_row)  
            t_embed = t_embed.view(-1)       
            # concaténer et projeter
            X_c_i = torch.cat([f_i, t_embed], dim=0) 
            E_list.append(self.dense(X_c_i))         

        E_X = torch.stack(E_list, dim=0)  
        return E_X

class StockDataset(Dataset):
    def __init__(
        self,
        folder_processed_csv: str,
        window_size: int,
        date2vec_model_path: str,
        non_time_feature_cols: list,
        time_embed_dim: int = 64,
        out_embed_dim: int = 128,
        split: str = "train",  
        use_all_stocks: bool = True,
        cache_file: str = None
    ):
        super().__init__()
        self.window_size = window_size
        self.non_time_cols = non_time_feature_cols
        self.f = len(non_time_feature_cols)
        self.time_embed_dim = time_embed_dim
        self.out_embed_dim = out_embed_dim
        self.split = split.lower().strip()  
        self.cache_file = cache_file

        #limites de date
        self.train_end_date = pd.to_datetime("2015-08-08")
        self.val_end_date   = pd.to_datetime("2015-10-01")
        self.test_end_date  = pd.to_datetime("2016-01-01")

        # charge Date2Vec pre train
        self.d2v = Date2VecConvert(model_path=date2vec_model_path)

        #encodage spatio-temporel
        self.st_embed = SpatiotemporalEmbed(
            window_size=self.window_size,
            f=self.f,
            time_emb_dim=self.time_embed_dim,
            d=self.out_embed_dim
        )

        #si un fichier de cache existe, charger les séquences.
        if cache_file is not None and os.path.exists(cache_file):
            print(f"Chargement des données en cache depuis {cache_file}")
            with open(cache_file, "rb") as f:
                self.sequences = pickle.load(f)
        else:
            print(f"Aucune donnée en cache")
            csv_paths = glob.glob(os.path.join(folder_processed_csv, "*.csv"))
            if not csv_paths:
                raise FileNotFoundError(f"Aucun CSV trouvé dans {folder_processed_csv}.")
            self.data = []
            for path in csv_paths:
                if use_all_stocks:
                    df = pd.read_csv(path, parse_dates=["Date"])
                    if not df.empty:
                        df["StockSymbol"] = os.path.splitext(os.path.basename(path))[0]
                        self.data.append(df)
                else:
                    pass
            self.data = pd.concat(self.data, ignore_index=True)
            self.data.sort_values(["StockSymbol", "Date"], inplace=True)
            self.data.reset_index(drop=True, inplace=True)

            all_sequences = self.build_sequences(self.data)
            
            #filtrer les séquences par division en fonction de la dernière date
            self.sequences = []
            for seq in all_sequences:
                last_date = seq[2]
                if self.belongs_to_split(last_date):
                    self.sequences.append((seq[0], seq[1]))
            if cache_file is not None:
                with open(cache_file, "wb") as f:
                    pickle.dump(self.sequences, f)
                print(f"Données prétraitées mises en cache dans {cache_file}")

    #filtre les séquences selon la division
    def belongs_to_split(self, date: pd.Timestamp) -> bool:
        if self.split == "train":
            return date < self.train_end_date
        elif self.split == "val":
            return (date >= self.train_end_date) and (date < self.val_end_date)
        elif self.split == "test":
            return (date >= self.val_end_date) and (date < self.test_end_date)
        else:
            raise ValueError(f"Division inconnue : split={self.split}")

    ## construit les séquences à partir du DataFrame
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
                    fvals = [float(row_data[col]) for col in self.non_time_cols]
                    X_f_list.append(fvals)
                    X_t_list.append([
                        float(row_data["Year"]),    # année/3000
                        float(row_data["Month"]),   # mois/12
                        float(row_data["Day"]),     # jour/31
                        float(row_data["Weekday"]), # jour semaine/7
                        0.0,                        # pad
                        0.0                         # pad
                    ])
                X_f_np = np.array(X_f_list, dtype=np.float32)
                X_t_np = np.array(X_t_list, dtype=np.float32)
                # centrer-réduire chaque colonne de X_f_np
                mean = X_f_np.mean(axis=0, keepdims=True)   
                std  = X_f_np.std(axis=0,  keepdims=True)    
                X_f_np = (X_f_np - mean) / (std + 1e-6)       # éviter la div par zéro
                
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

# regroupement
def collate_fn(batch):
    E_list, y_list = zip(*batch)
    E_tensor = torch.stack(E_list, dim=0)
    y_tensor = torch.stack(y_list, dim=0)
    return E_tensor, y_tensor