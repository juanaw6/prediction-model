import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

class CryptoDataset(Dataset):
    def __init__(self, data, seq_len=60, pred_steps=7, scaler=None, train=True):
        data = data[['close', 'volume', 'rsi', 'macd']].values
        
        if train:
            self.scaler = StandardScaler()
            self.normalized = self.scaler.fit_transform(data)
        else:
            self.normalized = scaler.transform(data)
            
        self.seq_len = seq_len
        self.pred_steps = pred_steps

    def __len__(self):
        return len(self.normalized) - self.seq_len - self.pred_steps

    def __getitem__(self, idx):
        x = self.normalized[idx:idx+self.seq_len]
        y = self.normalized[idx+self.seq_len:idx+self.seq_len+self.pred_steps, 0]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class GeometricAttention(nn.Module):
    """AlphaFold-inspired triangular attention for time series"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, 3*d_model)
        self.gating = nn.Linear(d_model, num_heads*self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)

        self.relpos = nn.Embedding(512, num_heads)
        
        self.edge_update = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.SiLU(),
            nn.Linear(4*d_model, num_heads)
        )

    def forward(self, x, mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [y.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3) for y in qkv]

        gate = torch.sigmoid(self.gating(x))
        gate = gate.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        rel_indices = torch.arange(T, device=x.device)[:, None] - torch.arange(T, device=x.device)[None, :]
        rel_indices = rel_indices.clamp(-255, 255) + 255
        rel_pos = self.relpos(rel_indices)
        rel_pos = rel_pos.permute(2, 0, 1).unsqueeze(0)
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        edge_contribution = self.edge_update(x)
        edge_contribution = edge_contribution.permute(0, 2, 1).unsqueeze(-1)
        attn = attn + rel_pos + edge_contribution

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v * gate).permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return self.out_proj(x)

class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, d_model, expansion=4):
        super().__init__()
        self.dynamic_conv = nn.Conv1d(
            d_model, d_model * expansion, 3,
            padding=1, groups=d_model
        )
        self.attention = GeometricAttention(d_model, num_heads=8)
        self.projection = nn.Sequential(
            nn.LayerNorm(d_model * expansion),
            nn.GELU(),
            nn.Linear(d_model * expansion, d_model)
        )
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        res = x
        x_conv = self.dynamic_conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_conv = self.projection(x_conv)
        x_attn = self.attention(self.norm(x))
        x = x_conv + x_attn
        return self.dropout(x) + res

class HierarchicalTemporalNet(nn.Module):
    def __init__(self, input_dim=4, d_model=256, pred_steps=7, num_layers=12):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(0.1)
        )
        
        self.blocks = nn.ModuleList([
            MultiScaleTemporalBlock(d_model)
            for _ in range(num_layers)
        ])
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.SiLU(),
            nn.Linear(d_model*4, pred_steps)
        )

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        
        x = x.mean(dim=1) + x.amax(dim=1)
        return self.decoder(x)


def train_model(model, train_loader, val_loader, epochs=100, lr=3e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, 
        steps_per_epoch=len(train_loader), 
        epochs=epochs,
        pct_start=0.3
    )
    loss_fn = nn.HuberLoss()
    
    best_val = float('inf')
    metrics = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_loss += loss_fn(pred, y).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'best_temporal_model.pth')

    return metrics

def predict(model, data_loader, scaler):
    device = next(model.parameters()).device
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for X, _ in data_loader:
            X = X.to(device)
            pred = model(X).cpu().numpy()
            dummy = np.zeros((pred.shape[0], pred.shape[1], 4))
            dummy[:, :, 0] = pred
            inverse_pred = scaler.inverse_transform(
                dummy.reshape(-1, 4)).reshape(dummy.shape)[:, :, 0]
            predictions.append(inverse_pred)
    
    return np.concatenate(predictions)


if __name__ == "__main__":
    # Configuration
    SEQ_LEN = 60
    PRED_STEPS = 7
    BATCH_SIZE = 128
    EPOCHS = 100
    
    df = pd.read_csv('rm_output.csv')
    train_size = int(0.8 * len(df))
    
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size - SEQ_LEN:]
    
    print(f"Train data last time: {train_df.index[-1]}")
    print(f"Test data first time: {test_df.index[0]}")
    
    train_dataset = CryptoDataset(train_df, seq_len=SEQ_LEN, pred_steps=PRED_STEPS)
    test_dataset = CryptoDataset(test_df, seq_len=SEQ_LEN, pred_steps=PRED_STEPS,
                                scaler=train_dataset.scaler, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, num_workers=4)
    
    model = HierarchicalTemporalNet(
        input_dim=4,
        d_model=256,
        pred_steps=PRED_STEPS,
        num_layers=12
    )
    
    # Train the model
    print("Starting training...")
    metrics = train_model(model, train_loader, test_loader, epochs=EPOCHS)
    
    # Inference example
    print("\nRunning inference...")
    model.load_state_dict(torch.load('best_temporal_model.pth'))
    predictions = predict(model, test_loader, train_dataset.scaler)
    
    # Example prediction output
    print(f"\nPredicted prices for next {PRED_STEPS} days:")
    print(predictions[-1])  # Latest prediction