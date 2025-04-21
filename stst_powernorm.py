# stst_model.py
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef

import copy
from time import time
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


class PowerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(PowerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        mean = x.mean(dim=[1], keepdim=True)
        var = ((x - mean) ** 2).mean(dim=[1], keepdim=True)
        std = torch.sqrt(var + self.eps)
        normed = (x - mean) / std
        return self.gamma * normed + self.beta



# ---------------------------------------------------------
# 1. Transformer Encoder Block using multi-head self-attention,
#    residual connections, and post-normalization (LayerNorm).
# -----------------------------------------------------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, attn_dropout=0.1, ff_dropout=0.3):
        """
        d_model: hidden size of the transformer
        nhead: number of attention heads
        dim_feedforward: dimension of the feed-forward sublayer
        attn_dropout: dropout rate for multi-head attention
        ff_dropout: dropout rate for feed-forward layers
        """
        super(TransformerEncoderBlock, self).__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=attn_dropout,
            batch_first=True
        )
        
        # Post-norm style
        self.norm1 = PowerNorm(d_model)
        self.norm2 = PowerNorm(d_model)
        
        # Separate dropout for attention vs. feed-forward
        self.dropout_attn = nn.Dropout(attn_dropout)
        self.dropout_ff = nn.Dropout(ff_dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x)  # (batch, seq_len, d_model)
        x = x + self.dropout_attn(attn_output)    # Residual connection
        x = self.norm1(x)
        
        # Feed-forward
        ffn_output = self.ffn(x)                  # (batch, seq_len, d_model)
        x = x + self.dropout_ff(ffn_output)       # Residual connection
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
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        for layer in self.layers:
            x = layer(x)
        return x

# -----------------------------------------------------------
# 3. STST Model with LSTM:
#    - The transformer encoder processes the spatiotemporal embeddings.
#    - The sequence output from the encoder is fed into a multi-layer LSTM.
#    - We then use a feed-forward classifier to get a single sigmoid output.
# -----------------------------------------------------------
class STSTModel(nn.Module):
    def __init__(
        self,
        d_model,
        num_layers,
        nhead,
        dim_feedforward,
        attn_dropout,
        ff_dropout,
        n_lstm_layers=2,
        d_lstm_hidden=256,
        classifier_hidden=64
    ):
        """
        d_model: dimension of tokens from spatiotemporal embedding
        num_layers: number of transformer encoder blocks
        nhead: number of attention heads
        dim_feedforward: hidden dimension in transformer blocks (feed-forward)
        attn_dropout: attention dropout
        ff_dropout: feed-forward dropout
        n_lstm_layers: number of stacked LSTM layers
        d_lstm_hidden: hidden size for LSTM
        classifier_hidden: hidden dimension for final MLP
        """
        super(STSTModel, self).__init__()
        
        # 1) Transformer Encoder
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )
        
        # 2) Multi-layer LSTM
        #    The LSTM input size must match d_model from the Transformer output
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_lstm_hidden,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=0.0,  # You can add LSTM dropout if desired
            bidirectional=False
        )
        
        # 3) Classifier (Feed-forward)
        #    We'll do a small MLP: [d_lstm_hidden -> classifier_hidden -> 1]
        self.classifier = nn.Sequential(
            nn.Linear(d_lstm_hidden, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(classifier_hidden, 1)
        )
        
    def forward(self, x):
        """
        x: (batch, seq_len, d_model) from the spatiotemporal embedding.
        Returns: (batch,) probability in [0,1].
        """
        # (A) Transformer encoder
        encoded = self.encoder(x)  # (batch, seq_len, d_model)
        
        # (B) LSTM
        #     The LSTM returns (lstm_out, (h_n, c_n))
        #     - lstm_out: (batch, seq_len, d_lstm_hidden)
        #     - h_n: (n_lstm_layers, batch, d_lstm_hidden)
        lstm_out, (h_n, c_n) = self.lstm(encoded)
        
        # We'll use the final hidden state from the top LSTM layer as our context vector
        # h_n is shape (n_lstm_layers, batch, d_lstm_hidden)
        final_h = h_n[-1]  # (batch, d_lstm_hidden)
        
        # (C) Classifier
        logits = self.classifier(final_h)   # (batch, 1)
        prob = torch.sigmoid(logits).squeeze(-1)  # (batch,)
        
        return prob

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
def train_model(model, train_loader, val_loader, num_epochs, optimizer, criterion, warmup_steps, device, save_path="best_model.pth"):
    model.to(device)
    scheduler = WarmupScheduler(optimizer, warmup_steps=warmup_steps)
    
    best_val_loss = float('inf')
    patience = 100
    trigger_times = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)  # (batch, seq_len, d_model)
            y_batch = y_batch.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(X_batch)      # (batch,)
            #print(outputs.min(), outputs.max())
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item() * X_batch.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device).float()
                
                outputs = model(X_val)    # (batch,)
                loss = criterion(outputs, y_val)
                val_loss += loss.item() * X_val.size(0)
                
                preds = (outputs >= 0.5).long()
                correct += (preds == y_val.long()).sum().item()
                total += y_val.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss={epoch_loss:.4f} | "
              f"Val Loss={val_loss:.4f} | "
              f"Val Acc={val_acc:.4f}")
        #Print the numbers of steps and the learning rate
        print(f"Step: {scheduler.last_epoch}, LR: {scheduler.get_lr()}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, save_path)  # Save the best model
            print(f"Best model saved at epoch {epoch+1}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping!")
                break
    
    # Load best weights
    model.load_state_dict(best_model_wts)
    return model

# -----------------------------------------------------------
# 6. Example Usage: Training Script
# -----------------------------------------------------------
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset import StockDataset, collate_fn  # Make sure dataset.py is in the same folder
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters from your paper's best settings
    processed_data_folder = "data/stocknet-dataset-processed"
    date2vec_model_path = r"Date2Vec\d2v_model\d2v_98291_17.169918439404636.pth"
    
    window_size = 32
    batch_size = 32
    lr = 5.35e-6
    warmup_steps = 2500
    num_epochs = 100
    
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
    classifier_hidden = 64
    
    # Columns
    non_time_feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_10","SMA_30","SMA_50","SMA_200",
        "EMA_10","EMA_30","EMA_50","EMA_200",
        "MOM_10","STOCHRSIk","STOCHRSId","STOCHk","STOCHd",
        "MACD","MACD_signal","CCI_14","MFI_14","AD","OBV","ROC_10",
        "SIG_SMA_10","SIG_SMA_30","SIG_SMA_50","SIG_SMA_200",
        "SIG_EMA_10","SIG_EMA_30","SIG_EMA_50","SIG_EMA_200",
        "SIG_MOM","SIG_STOCHRSI","SIG_STOCH_K","SIG_STOCH_D",
        "SIG_MACD","SIG_CCI","SIG_MFI","SIG_AD","SIG_OBV","SIG_ROC"
    ]
    
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
    criterion = nn.BCELoss()  # Binary classification
    print(f"Model instantiation took {time() - instantation_start:.2f} seconds.")
    
    print("Starting training...")
    training_start = time()
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        criterion=criterion,
        warmup_steps=warmup_steps,
        device=device
    )
    print(f"Training took {time() - training_start:.2f} seconds.")
    
    # After evaluating over the test loader, collect all predictions and true labels:
    all_preds = []
    all_targets = []

    start_test = time()
    model.eval()
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device).float()
            
            outputs = model(X_test)            # (batch,)
            preds = (outputs >= 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_test.cpu().numpy())

    # Compute Accuracy:
    acc = accuracy_score(all_targets, all_preds)

    # Compute MCC:
    mcc = matthews_corrcoef(all_targets, all_preds)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test MCC: {mcc:.4f}")
    print(f"Testing took {time() - start_test:.2f} seconds.")
