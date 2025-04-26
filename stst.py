import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_curve, f1_score
from collections import Counter

import copy
from time import time
import random
import csv
import matplotlib.pyplot as plt

from powernorm.mask_powernorm import MaskPowerNorm as PowerNorm

# Set deterministic seeds

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(24)


# -----------------------------------------------------------
# 1. Transformer Encoder Block using multi-head self-attention,
#    residual connections, and post-normalization (PowerNorm).
# -----------------------------------------------------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, attn_dropout=0.1, ff_dropout=0.3):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=attn_dropout, batch_first=True
        )
        self.norm1 = PowerNorm(d_model)
        self.norm2 = PowerNorm(d_model)
        #from torch.nn import LayerNorm
        #self.norm1 = LayerNorm(d_model)
        #self.norm2 = LayerNorm(d_model)
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
        x = self.norm1(x)
        ff_out = self.ffn(x)
        x = x + self.dropout_ff(ff_out)
        x = self.norm2(x)
        return x


# -----------------------------------------------------------
# 2. Transformer Encoder as a stack of encoder blocks.
# -----------------------------------------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward,
                 attn_dropout=0.1, ff_dropout=0.3):
        super(TransformerEncoder, self).__init__()
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


# -----------------------------------------------------------
# 3. STST Model with LSTM and PowerNorm in Transformer.
# -----------------------------------------------------------
class STSTModel(nn.Module):
    def __init__(self, d_model, num_layers, nhead, dim_feedforward,
                 attn_dropout, ff_dropout, n_lstm_layers, d_lstm_hidden,
                 classifier_hidden):
        super().__init__()
        # Transformer
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )
        # LSTM
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=d_lstm_hidden,
            num_layers=n_lstm_layers, batch_first=True
        )
        # Classifier sans sigmoid
        self.classifier = nn.Sequential(
            nn.Linear(d_lstm_hidden, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(classifier_hidden, 1),
        )

    def forward(self, x):
        x0 = x             # (batch, seq, d_model)
        x_enc = self.encoder(x0)      # <-- ici on utilise TransformerEncoder.forward

        # Résidu
        x = F.gelu(x_enc + x0)

        # LSTM + MLP
        lstm_out, (h_n, c_n) = self.lstm(x)
        final_h = h_n[-1]
        logits  = self.classifier(final_h).squeeze(-1)
        return logits

# -----------------------------------------------------------
# 4. Learning Rate Warmup Scheduler (linear warmup).
# -----------------------------------------------------------
class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        # Linear warmup for first 'warmup_steps' steps
        if self.last_epoch < self.warmup_steps:
            scale = float(self.last_epoch + 1) / float(self.warmup_steps)
            warmed_lrs = [base_lr * scale for base_lr in self.base_lrs]
            #print(f"LRs warmed up to: {warmed_lrs} due to the following epoch: {self.last_epoch}")
            return warmed_lrs
        else:
            warmed_lrs = [base_lr for base_lr in self.base_lrs]
            #print(f"LRs not warmed up: {warmed_lrs}")
            return warmed_lrs

# -----------------------------------------------------------
# 5. Training Loop with Early Stopping.
# -----------------------------------------------------------
def train_model(
    model, train_loader, val_loader,
    num_epochs, optimizer, criterion,
    warmup_steps, device, save_path="best.pth",
    log_path=None
):
    model.to(device)
    scheduler = WarmupScheduler(optimizer, warmup_steps)

    # CSV writer si besoin
    writer = None
    if log_path:
        f = open(log_path, "w", newline="")
        writer = csv.DictWriter(f, fieldnames=["epoch","batch_idx","mean_prob","count_0","count_1"])
        writer.writeheader()

    # listes pour plot
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    train_props0, train_props1 = [], []
    val_props0,   val_props1   = [], []

    best_val_loss = float('inf')
    patience, triggers = 50, 0
    best_wts = model.state_dict()

    for epoch in range(1, num_epochs+1):
        # — entraînement —
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # — éval sur TRAIN —
        model.eval()
        train_loss = train_correct = train_total = 0
        train_c0 = train_c1 = 0
        with torch.no_grad():
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device).float()
                logits = model(Xb)
                loss   = criterion(logits, yb)
                probs  = torch.sigmoid(logits)
                preds  = (probs >= 0.5).long()

                train_loss    += loss.item() * Xb.size(0)
                train_correct += (preds == yb.long()).sum().item()
                train_total   += Xb.size(0)
                train_c0      += (preds == 0).sum().item()
                train_c1      += (preds == 1).sum().item()

        train_loss /= train_total
        train_acc  = train_correct / train_total
        prop0_tr = train_c0 / train_total
        prop1_tr = train_c1 / train_total

        # — éval sur VAL —
        val_loss = val_correct = val_total = 0
        val_c0 = val_c1 = 0
        with torch.no_grad():
            for batch_idx, (Xv, yv) in enumerate(val_loader):
                Xv, yv = Xv.to(device), yv.to(device).float()
                logits = model(Xv)
                loss   = criterion(logits, yv)
                probs  = torch.sigmoid(logits)
                preds  = (probs >= 0.5).long()

                val_loss    += loss.item() * Xv.size(0)
                val_correct += (preds == yv.long()).sum().item()
                val_total   += Xv.size(0)
                val_c0      += (preds == 0).sum().item()
                val_c1      += (preds == 1).sum().item()

                if writer:
                    writer.writerow({
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "mean_prob": probs.mean().item(),
                        "count_0": int((preds==0).sum().item()),
                        "count_1": int((preds==1).sum().item()),
                    })

        val_loss /= val_total
        val_acc  = val_correct / val_total
        prop0_val = val_c0  / val_total
        prop1_val = val_c1  / val_total

        # stocker pour le plot
        train_props0.append(prop0_tr)
        train_props1.append(prop1_tr)
        val_props0.append(prop0_val)
        val_props1.append(prop1_val)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # affichage complet
        print(f"Epoch {epoch} TRAIN Loss={train_loss:.4f} | Acc={train_acc:.4f} "
              f"| preds 0→{prop0_tr:.2%}, 1→{prop1_tr:.2%}")
        print(f"           VAL   Loss={val_loss:.4f} | Acc={val_acc:.4f} "
              f"| preds 0→{prop0_val:.2%}, 1→{prop1_val:.2%}")

        # early stopping + sauvegarde best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            triggers = 0
            best_wts = model.state_dict()
            torch.save(best_wts, save_path)
            print(f"  ↳ Nouvelle meilleure val_loss={val_loss:.4f}, modèle sauvé→ `{save_path}`")
        else:
            triggers += 1
            if triggers >= patience:
                print("Early stopping")
                break

    if writer:
        f.close()

    # recharger le meilleur modèle
    model.load_state_dict(best_wts)
    return model, train_losses, val_losses, train_accs, val_accs, train_props0, train_props1, val_props0, val_props1



# -----------------------------------------------------------
# 6. Example Usage: Training Script
# -----------------------------------------------------------
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset import StockDataset, collate_fn  # Make sure dataset.py is in the same folder
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters from your paper's best settings
    processed_data_folder = "data/stocknet-dataset-processed"
    date2vec_model_path = r"Date2Vec/d2v_model/d2v_98291_17.169918439404636.pth"
    
    window_size = 32
    batch_size = 32
    lr = 5.35e-6
    warmup_steps = 25000
    num_epochs = 96
    
    # Transformer parameters
    num_transformer_layers = 4
    nhead = 4
    d_model = 64
    dim_feedforward = 2048
    attn_dropout = 0.1
    ff_dropout = 0.3
    
    # LSTM parameters
    n_lstm_layers = 2
    d_lstm_hidden = 256
    
    # Classifier hidden dimension (the MLP after LSTM)
    classifier_hidden = 128
    
    # Columns
    non_time_feature_cols = [
    "Open", "High", "Low", "Close", "Adj Close", "Volume",
    *[f"SIG_SMA_{i}" for i in (10, 30, 50, 200)],
    *[f"SIG_EMA_{i}" for i in (10, 30, 50, 200)],
    "SIG_MOM", "SIG_STOCHRSI", "SIG_STOCH_K", "SIG_STOCH_D",
    "SIG_MACD", "SIG_CCI", "SIG_MFI", "SIG_AD", "SIG_OBV", "SIG_ROC"]
    
    print("Creating datasets...")
    dataset_create_start = time()
    
    # Train/Val/Test split
    train_dataset = StockDataset(
        folder_processed_csv=processed_data_folder,
        window_size=window_size,
        date2vec_model_path=date2vec_model_path,
        non_time_feature_cols=non_time_feature_cols,
        time_embed_dim=64,
        out_embed_dim=d_model,
        split="train",
        use_all_stocks=True,
        cache_file="cache_train.pkl"  # Caching enabled
    )


    val_dataset = StockDataset(
        folder_processed_csv=processed_data_folder,
        window_size=window_size,
        date2vec_model_path=date2vec_model_path,
        non_time_feature_cols=non_time_feature_cols,
        time_embed_dim=64,
        out_embed_dim=d_model,
        split="val",    # <--- key
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
        split="test",   # <--- key
        use_all_stocks=True,
        cache_file="cache_test.pkl"
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  collate_fn=collate_fn)
    # right after you build train_loader
    #warmup_steps = len(train_loader)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, collate_fn=collate_fn)
    print(f"Dataset creation took {time() - dataset_create_start:.2f} seconds.")
    
    print("Instantiating model...")
    instantation_start = time()
    # Instantiate STST Model with LSTM
    model = STSTModel(
        d_model=d_model,
        num_layers=num_transformer_layers,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
        n_lstm_layers=n_lstm_layers,
        d_lstm_hidden=d_lstm_hidden,
        classifier_hidden=classifier_hidden
    )

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 1) Récupère tous les labels de train
    labels = [y for (_, y) in train_dataset.sequences]  # sequences = [(E_X, y), ...]
    cnt    = Counter(labels)
    n0, n1 = cnt[0], cnt[1]

    # 2) Construis le pos_weight
    pos_weight = torch.tensor([n0 / n1], dtype=torch.float, device=device)
    print(f"  → pos_weight = {pos_weight.item():.3f} (n0={n0}, n1={n1})")

    # 3) Instancie ta loss pondérée
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    #criterion = nn.BCEWithLogitsLoss()  # Binary classification
    print(f"Model instantiation took {time() - instantation_start:.2f} seconds.")
    
    print("Starting training...")
    # juste avant d’entrer dans train_model()
    log_path = "val_debug.csv"
    with open(log_path, "w", newline="") as csvfile:
        fieldnames = ["epoch", "batch_idx", "mean_prob", "count_0", "count_1"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    training_start = time()
    model, train_losses, val_losses, train_accs, val_accs, train_props0, train_props1, val_props0, val_props1 = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        criterion=criterion,
        warmup_steps=warmup_steps,
        device=device,
        log_path="val_debug.csv"
    )
    print(f"Training took {time() - training_start:.2f} seconds.")
    
    # After evaluating over the test loader, collect all predictions and true labels:
    all_preds = []
    all_targets = []

    start_test = time()

    # --- calcul du seuil optimal sur la VAL ---
    model.eval()
    all_val_probs   = []
    all_val_targets = []

    with torch.no_grad():
        for Xv, yv in val_loader:
            Xv = Xv.to(device)
            yv = yv.to(device).float()
            logits = model(Xv)
            probs  = torch.sigmoid(logits)
            all_val_probs.extend(probs.cpu().numpy())
            all_val_targets.extend(yv.cpu().numpy())

    # ROC pour récupérer thresholds
    #fpr, tpr, thresholds = roc_curve(all_val_targets, all_val_probs)
    # maximise Youden’s J = tpr − fpr
    #j_scores      = tpr - fpr
    #ix_opt        = np.argmax(j_scores)
    #opt_threshold = thresholds[ix_opt]
    #print(f"Seuil optimal sur VAL  = {opt_threshold:.3f} (J={j_scores[ix_opt]:.3f})")

    threshs = np.linspace(np.min(all_val_probs), np.max(all_val_probs), 100)
    f1s = [f1_score(all_val_targets, np.array(all_val_probs)>=t) for t in threshs]
    best_idx = np.argmax(f1s)
    opt_threshold = threshs[best_idx]            # ← on l’appelle opt_threshold
    print(f"Seuil F1-optimal = {opt_threshold:.3f} (F1={f1s[best_idx]:.3f})")

    model.eval()
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device).float()
            
            outputs = model(X_test)            # (batch,)
            preds = (torch.sigmoid(outputs) >= opt_threshold).long().cpu().numpy()
            #preds = (outputs >= 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_test.cpu().numpy())

    #Vérfier la distribution de logits
    test_probs = []
    with torch.no_grad():
        for X_test, _ in test_loader:
            X_test = X_test.to(device)
            logits = model(X_test)
            test_probs.extend(torch.sigmoid(logits).cpu().numpy())

    print(f"Test probs  min={np.min(test_probs):.3f}, mean={np.mean(test_probs):.3f},  max={np.max(test_probs):.3f}")

    # Compute Accuracy:
    acc = accuracy_score(all_targets, all_preds)

    # Compute MCC:
    mcc = matthews_corrcoef(all_targets, all_preds)

    n0 = all_preds.count(0)
    n1 = all_preds.count(1)
    total = n0 + n1
    print(f"Test preds 0→{n0/total:.2%}, 1→{n1/total:.2%}")
    print(f"Test Accuracy: {acc:.4f}  |  Test MCC: {mcc:.4f}")
    print(f"Test MCC: {mcc:.4f}")
    print(f"Testing took {time() - start_test:.2f} seconds.")

    epochs = list(range(1, len(train_losses)+1))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,5))

    # 1) Loss
    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses,   label="Val Loss")
    ax1.set_title("Loss");     ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid()

    # 2) Accuracy
    ax2.plot(epochs, train_accs, label="Train Acc")
    ax2.plot(epochs, val_accs,   label="Val Acc")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.legend(); ax2.grid()

    # 3) Proportions de classes
    ax3.plot(epochs, train_props0, label="Train ⌀0")
    ax3.plot(epochs, train_props1, label="Train ⌀1")
    ax3.plot(epochs,   val_props0, label="Val ⌀0", linestyle="--")
    ax3.plot(epochs,   val_props1, label="Val ⌀1", linestyle="--")
    ax3.set_title("Proportions prédictions")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Proportion")
    ax3.legend(); ax3.grid()

    plt.tight_layout()
    plt.savefig("train_val_metrics_with_props.png")
    plt.close(fig)                # ← closes the figure to free memory
