import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class EnhancedFinancialTransformer(nn.Module):
    def __init__(self, feature_dim, d_model=128, nhead=8, 
                 num_layers=6, dim_feedforward=512, dropout=0.2, weight_decay=1e-5):
        super(EnhancedFinancialTransformer, self).__init__()

        self.price_projection = nn.Linear(feature_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model)

        self.trend_encoder = self._create_domain_encoder('trend_following', d_model, nhead, 
                                                        dim_feedforward, dropout)
        self.mean_rev_encoder = self._create_domain_encoder('mean_reversion', d_model, nhead, 
                                                          dim_feedforward, dropout)

        self.standard_encoder = self._create_encoder(d_model, nhead, num_layers, 
                                                    dim_feedforward, dropout)
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model * 3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.weight_decay = weight_decay
    
    def _create_encoder(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        return nn.TransformerEncoder(encoder_layers, num_layers)
    
    def _create_domain_encoder(self, attention_type, d_model, nhead, dim_feedforward, dropout):
        encoder_layers = DomainSpecificEncoderLayer(
            d_model=d_model, nhead=nhead, attention_type=attention_type,
            dim_feedforward=dim_feedforward, dropout=dropout
        )
        return nn.ModuleList([encoder_layers for _ in range(3)])
    
    def forward(self, x):
        x = x.permute(1, 0, 2)
        
        x = self.price_projection(x)
        
        x = self.pos_encoder(x)
        
        standard_output = self.standard_encoder(x)
        
        trend_output = x
        for layer in self.trend_encoder:
            trend_output = layer(trend_output)
        
        mean_rev_output = x
        for layer in self.mean_rev_encoder:
            mean_rev_output = layer(mean_rev_output)
        
        standard_repr = standard_output[-1]
        trend_repr = trend_output[-1]
        mean_rev_repr = mean_rev_output[-1]
        
        combined = torch.cat([standard_repr, trend_repr, mean_rev_repr], dim=1)
        
        return self.decoder(combined).squeeze(-1)

class DomainSpecificEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, attention_type, dim_feedforward, dropout):
        super(DomainSpecificEncoderLayer, self).__init__()
        
        if attention_type == 'trend_following':
            self.self_attn = TrendFollowingAttention(d_model, nhead, dropout)
        elif attention_type == 'mean_reversion':
            self.self_attn = MeanReversionAttention(d_model, nhead, dropout)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.GELU()
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class TrendFollowingAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(TrendFollowingAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.time_bias = nn.Parameter(torch.zeros(1, 1, d_model))
        
    def forward(self, query, key, value):
        seq_len = key.size(0)
        
        time_weights = torch.linspace(0, 1, seq_len).unsqueeze(-1).unsqueeze(-1).to(key.device)
        time_bias = time_weights * self.time_bias
        
        biased_key = key + time_bias
        
        attn_output, attn_weights = self.multihead_attn(query, biased_key, value)
        return attn_output, attn_weights


class MeanReversionAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MeanReversionAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.deviation_weight = nn.Parameter(torch.ones(1, 1, d_model))
        
    def forward(self, query, key, value):
        mean_repr = torch.mean(key, dim=0, keepdim=True)
        
        deviation = key - mean_repr
        
        biased_key = key + (deviation * self.deviation_weight * -0.1)
        
        attn_output, attn_weights = self.multihead_attn(query, biased_key, value)
        return attn_output, attn_weights

def preprocess_data(df):
    """Preprocess OHLCV data for the model - using only original features"""
    features = df[['open', 'high', 'low', 'close', 'volume']].values
    
    targets = df['close'].pct_change().shift(-1).values

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    scaled_features = np.nan_to_num(scaled_features)
    targets = np.nan_to_num(targets)
    
    return scaled_features, targets


class FinancialDataset(Dataset):
    def __init__(self, features, targets, seq_length=60):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.features) - self.seq_length
    
    def __getitem__(self, idx):
        X = self.features[idx:idx+self.seq_length]
        y = self.targets[idx+self.seq_length-1]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train_model(model, train_loader, val_loader, epochs=50, lr=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr,
        weight_decay=model.weight_decay,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True
    )
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 10
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for X_batch, y_batch in train_pbar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Valid]')
            for X_val, y_val in val_pbar:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()
                
                val_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Training Loss: {avg_train_loss:.6f}')
        print(f'Validation Loss: {avg_val_loss:.6f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), './results/ft/best_financial_transformer.pth')
            print("✓ Saved best model")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        print('-' * 40)
    
    return model

def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc='Evaluating')
        for X_test, y_test in test_pbar:
            X_test, y_test = X_test.to(device), y_test.to(device)
            outputs = model(X_test)
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_test.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    
    correct_direction = np.sum((predictions > 0) == (actuals > 0))
    directional_accuracy = correct_direction / len(predictions)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'directional_accuracy': directional_accuracy,
        'predictions': predictions,
        'actuals': actuals
    }

def main():
    csv_data_path = './data/raw/solusdt_5m_2024_2025.csv'
    print("Loading data...")
    df = pd.read_csv(csv_data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    
    print("Preprocessing data...")
    features, targets = preprocess_data(df)

    train_size = int(0.7 * len(features))
    val_size = int(0.15 * len(features))
    
    train_features = features[:train_size]
    train_targets = targets[:train_size]
    
    val_features = features[train_size:train_size+val_size]
    val_targets = targets[train_size:train_size+val_size]
    
    test_features = features[train_size+val_size:]
    test_targets = targets[train_size+val_size:]
    
    seq_length = 60
    batch_size = 32
    
    print("Creating datasets...")
    train_dataset = FinancialDataset(train_features, train_targets, seq_length)
    val_dataset = FinancialDataset(val_features, val_targets, seq_length)
    test_dataset = FinancialDataset(test_features, test_targets, seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    
    feature_dim = features.shape[1]
    print(f"Creating model with feature dimension: {feature_dim}")
    model = EnhancedFinancialTransformer(
        feature_dim=feature_dim,
        dropout=0.2,
        weight_decay=1e-5
    )
    
    print("Starting training...")
    trained_model = train_model(model, train_loader, val_loader, epochs=50)
    
    print("Evaluating model...")
    results = evaluate_model(trained_model, test_loader)
    
    print("\nFinal Model Performance:")
    print(f"Test MSE: {results['mse']:.6f}")
    print(f"Test RMSE: {results['rmse']:.6f}")
    print(f"Test MAE: {results['mae']:.6f}")
    print(f"Directional Accuracy: {results['directional_accuracy']:.2%}")

if __name__ == "__main__":
    main()