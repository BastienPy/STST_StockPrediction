import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
import copy
from time import time
import random
import csv
import matplotlib.pyplot as plt

from powernorm.mask_powernorm import MaskPowerNorm as PowerNorm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(314)

#Transformer Encoder Block with multi-head, self-attention and residual connexions and post-normalization (PowerNorm)
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, attn_dropout=0.1, ff_dropout=0.3):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=attn_dropout, batch_first=True
        )
        self.norm1 = PowerNorm(d_model)
        self.norm2 = PowerNorm(d_model)
        self.dropout_attn = nn.Dropout(attn_dropout)
        self.dropout_ff   = nn.Dropout(ff_dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim_feedforward, d_model),
        )
    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout_attn(attn_out)
        x = x.permute(1, 0, 2)
        x = self.norm1(x)
        x = x.permute(1, 0, 2)
        ff_out = self.ffn(x)
        x = x + self.dropout_ff(ff_out)
        x = x.permute(1, 0, 2)
        x = self.norm2(x)
        x = x.permute(1, 0, 2)
        return x

#Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward,
                 attn_dropout=0.1, ff_dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

#STST Model with LSTM and PowerNorm in the Transformer.
class STSTModel(nn.Module):
    def __init__(self, d_model, num_layers, nhead, dim_feedforward,
                 attn_dropout, ff_dropout, n_lstm_layers, d_lstm_hidden,
                 classifier_hidden):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=d_lstm_hidden,
            num_layers=n_lstm_layers, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_lstm_hidden, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(classifier_hidden, 1),
        )

    def forward(self, x):
        x0 = x
        x_enc = self.encoder(x0)
        x = F.gelu(x_enc + x0)
        lstm_out, (h_n, c_n) = self.lstm(x)
        final_h = h_n[-1]
        logits  = self.classifier(final_h).squeeze(-1)
        return logits

# Warmup Scheduler for the learning rate
class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            scale = float(self.last_epoch + 1) / float(self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            return [base_lr for base_lr in self.base_lrs]

# Training function
def train_model(
    model, train_loader, val_loader,
    num_epochs, optimizer, criterion,
    warmup_steps, device, save_path="best.pth",
    log_path=None
):
    model.to(device)
    scheduler = WarmupScheduler(optimizer, warmup_steps)

    writer = None
    if log_path:
        f = open(log_path, "w", newline="")
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "batch_idx", "mean_prob", "count_0", "count_1"
        ])
        writer.writeheader()

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    train_props0, train_props1 = [], []
    val_props0,   val_props1   = [], []
    train_lrs = []

    best_val_loss = float('inf')
    patience, triggers = 10, 0
    best_wts = model.state_dict()

    for epoch in range(1, num_epochs+1):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()

        train_lrs.append(optimizer.param_groups[0]['lr'])

        # Evaluation on train and val sets
        model.eval()
        t_loss = t_corr = t_tot = 0
        t_c0 = t_c1 = 0
        with torch.no_grad():
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device).float()
                logits = model(Xb)
                loss   = criterion(logits, yb)
                probs  = torch.sigmoid(logits)
                preds  = (probs >= 0.5).long()

                t_loss    += loss.item() * Xb.size(0)
                t_corr    += (preds == yb.long()).sum().item()
                t_tot     += Xb.size(0)
                t_c0      += (preds == 0).sum().item()
                t_c1      += (preds == 1).sum().item()

        train_loss = t_loss / t_tot
        train_acc  = t_corr / t_tot
        prop0_tr, prop1_tr = t_c0 / t_tot, t_c1 / t_tot

        v_loss = v_corr = v_tot = 0
        v_c0 = v_c1 = 0
        with torch.no_grad():
            for batch_idx, (Xv, yv) in enumerate(val_loader):
                Xv, yv = Xv.to(device), yv.to(device).float()
                logits = model(Xv)
                loss   = criterion(logits, yv)
                probs  = torch.sigmoid(logits)
                preds  = (probs >= 0.5).long()

                v_loss    += loss.item() * Xv.size(0)
                v_corr    += (preds == yv.long()).sum().item()
                v_tot     += Xv.size(0)
                v_c0      += (preds == 0).sum().item()
                v_c1      += (preds == 1).sum().item()

                if writer:
                    writer.writerow({
                        "epoch":     epoch,
                        "batch_idx": batch_idx,
                        "mean_prob": float(probs.mean().cpu()),
                        "count_0":   int((preds == 0).sum().cpu()),
                        "count_1":   int((preds == 1).sum().cpu()),
                    })

        val_loss = v_loss / v_tot
        val_acc  = v_corr / v_tot
        prop0_val, prop1_val = v_c0 / v_tot, v_c1 / v_tot

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_props0.append(prop0_tr)
        train_props1.append(prop1_tr)
        val_props0.append(prop0_val)
        val_props1.append(prop1_val)

        print(f"Epoch {epoch}  TRAIN Loss={train_loss:.4f}|Acc={train_acc:.4f}|0→{prop0_tr:.2%},1→{prop1_tr:.2%}")
        print(f"             VAL   Loss={val_loss:.4f}|Acc={val_acc:.4f}|0→{prop0_val:.2%},1→{prop1_val:.2%}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            triggers = 0
            best_wts = model.state_dict()
            torch.save(best_wts, save_path)
            print(f"  ↳ Nouvelle meilleure val_loss={val_loss:.4f}, modèle sauvegardé")
        else:
            triggers += 1
            if triggers >= patience:
                print("Early stopping")
                break

    if writer:
        f.close()

    model.load_state_dict(best_wts)
    return (
        model,
        train_losses, val_losses,
        train_accs,   val_accs,
        train_props0, train_props1,
        val_props0,   val_props1,
        train_lrs
    )

# Whole script to run the model
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset import StockDataset, collate_fn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    processed_data_folder = "data/stocknet-dataset-processed"
    date2vec_model_path   = "Date2Vec/d2v_model/d2v_98291_17.169918439404636.pth"
    window_size    = 32
    batch_size     = 32
    lr             = 5e-6
    warmup_steps   = 2500
    num_epochs     = 50
    num_transformer_layers = 6
    nhead          = 4
    d_model         = 64
    dim_ff          = 1024
    attn_dropout    = 0.1
    ff_dropout      = 0.3
    n_lstm_layers  = 2
    d_lstm_hidden  = 256
    classifier_hidden = 256

    #Non-time features to use
    non_time_feature_cols = [
        "Open","High","Low","Close","Adj Close","Volume",
        *[f"SIG_SMA_{i}" for i in (10,30,50,200)],
        *[f"SIG_EMA_{i}" for i in (10,30,50,200)],
        "SIG_MOM","SIG_STOCHRSI","SIG_STOCH_K","SIG_STOCH_D",
        "SIG_MACD","SIG_CCI","SIG_MFI","SIG_AD","SIG_OBV","SIG_ROC"
    ]

    print("Creating datasets...")
    t0 = time()
    train_dataset = StockDataset(
        folder_processed_csv=processed_data_folder,
        window_size=window_size,
        date2vec_model_path=date2vec_model_path,
        non_time_feature_cols=non_time_feature_cols,
        time_embed_dim=64,
        out_embed_dim=d_model,
        split="train",
        use_all_stocks=True,
        cache_file="cache_train.pkl"
    )
    val_dataset = StockDataset(
        folder_processed_csv=processed_data_folder,
        window_size=window_size,
        date2vec_model_path=date2vec_model_path,
        non_time_feature_cols=non_time_feature_cols,
        time_embed_dim=64,
        out_embed_dim=d_model,
        split="val",
        use_all_stocks=True,
        cache_file="cache_val.pkl"
    )
    test_dataset = StockDataset(
        folder_processed_csv=processed_data_folder,
        window_size=window_size,
        date2vec_model_path=date2vec_model_path,
        non_time_feature_cols=non_time_feature_cols,
        time_embed_dim=64,
        out_embed_dim=d_model,
        split="test",
        use_all_stocks=True,
        cache_file="cache_test.pkl"
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"Datasets built in {time()-t0:.2f}s")

    print("Instantiating model...")
    model = STSTModel(
        d_model=d_model,
        num_layers=num_transformer_layers,
        nhead=nhead,
        dim_feedforward=dim_ff,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
        n_lstm_layers=n_lstm_layers,
        d_lstm_hidden=d_lstm_hidden,
        classifier_hidden=classifier_hidden
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    #Training
    model, train_losses, val_losses, \
        train_accs, val_accs, \
        train_p0, train_p1, val_p0, val_p1, \
        train_lrs = train_model(
            model, train_loader, val_loader,
            num_epochs, optimizer, criterion,
            warmup_steps, device,
            save_path="best.pth", log_path="val_debug.csv"
        )

    #Test set evaluation
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            logits = model(X_test)
            probs  = torch.sigmoid(logits)
            preds  = (probs >= 0.5).long().detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_test.detach().cpu().numpy())

    #AUC
    test_probs = []
    with torch.no_grad():
        for X_test, _ in test_loader:
            X_test = X_test.to(device)
            probs  = torch.sigmoid(model(X_test))
            test_probs.extend(probs.detach().cpu().numpy())
    auc = roc_auc_score(all_targets, test_probs)

    acc = accuracy_score(all_targets, all_preds)
    mcc = matthews_corrcoef(all_targets, all_preds)
    print(f"Test preds 0→{all_preds.count(0)/len(all_preds):.2%}, 1→{all_preds.count(1)/len(all_preds):.2%}")
    print(f"Test Accuracy: {acc:.4f}  |  Test MCC: {mcc:.4f}  |  Test AUC: {auc:.4f}")

    #Plotting training and validation metrics
    epochs = np.arange(1, len(train_losses)+1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses,   label="Val Loss")
    ax1.set(title="Loss", xlabel="Epoch", ylabel="Loss"); ax1.legend(); ax1.grid()

    ax2.plot(epochs, train_accs, label="Train Acc")
    ax2.plot(epochs, val_accs,   label="Val Acc")
    ax2.set(title="Accuracy", xlabel="Epoch", ylabel="Acc"); ax2.legend(); ax2.grid()

    ax3.plot(epochs, train_p0, label="Train 0"); ax3.plot(epochs, train_p1, label="Train 1")
    ax3.plot(epochs, val_p0, '--', label="Val 0");   ax3.plot(epochs, val_p1, '--', label="Val 1")
    ax3.set(title="Pred Proportions", xlabel="Epoch", ylabel="Prop"); ax3.legend(); ax3.grid()

    ax4.plot(epochs, train_lrs, label="LR")
    ax4.set(title="Learning Rate", xlabel="Epoch", ylabel="LR"); ax4.grid()

    plt.tight_layout()
    plt.savefig("train_val_metrics_with_lr.png")
    plt.close(fig)
