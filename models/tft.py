import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

# --------------------------
# 1. Data Preparation Pipeline
# --------------------------

class FinancialDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]
        
        return {
            'ohlc': torch.tensor(sequence[:, :4], dtype=torch.float32),
            'volume': torch.tensor(sequence[:, 4], dtype=torch.float32),
            'time_features': torch.tensor(sequence[:, 5:], dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32)  # Fixed line
        }

class DataProcessor:
    def __init__(self, seq_length=60, pred_horizon=5, test_size=0.2):
        self.seq_length = seq_length
        self.pred_horizon = pred_horizon
        self.test_size = test_size
        self.scalers = {}
        
    def process_data(self, df):
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Create features
        df = self._create_features(df)
        
        # Normalize
        df = self._normalize(df)
        
        # Create sequences
        sequences, targets = self._create_sequences(df)
        
        # Split data
        return self._train_test_split(sequences, targets)
    
    def _create_features(self, df):
        # Calculate returns
        df['return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volume features
        df['volume_pct'] = df['volume'].pct_change()
        df['volume_z'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
        
        # Time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        
        # Cyclical encoding
        for feat in ['hour', 'day_of_week']:
            df[f'{feat}_sin'] = np.sin(2 * np.pi * df[feat]/df[feat].max())
            df[f'{feat}_cos'] = np.cos(2 * np.pi * df[feat]/df[feat].max())
        
        return df.dropna()

    def _normalize(self, df):
        cols_to_scale = ['open', 'high', 'low', 'close', 'volume', 'return', 'volume_pct', 'volume_z']
        
        # Clean data before scaling
        df[cols_to_scale] = df[cols_to_scale].replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill().dropna()
        
        for col in cols_to_scale:
            # Check for valid values
            if not np.isfinite(df[col]).all():
                print(f"Invalid values in {col}:")
                print(df[col][~np.isfinite(df[col])])
                raise ValueError(f"{col} contains NaNs/infs")
                
            scaler = RobustScaler()
            df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
            self.scalers[col] = scaler
            
        return df

    def _create_sequences(self, df):
        features = df.drop(columns=['day_of_week', 'hour', 'day_of_month']).values
        targets = df['return'].shift(-self.pred_horizon).values
        
        sequences = []
        targets_clean = []
        
        for i in range(len(df) - self.seq_length - self.pred_horizon):
            seq = features[i:i+self.seq_length]
            target = targets[i+self.seq_length:i+self.seq_length+self.pred_horizon]
            
            if not np.isnan(target).any():
                sequences.append(seq)
                targets_clean.append(target.mean())  # Ensure scalar value
                
        return np.array(sequences), np.array(targets_clean)

    def _train_test_split(self, sequences, targets):
        # Time-series aware split
        split_idx = int(len(sequences) * (1 - self.test_size))
        return (sequences[:split_idx], targets[:split_idx]), (sequences[split_idx:], targets[split_idx:])

# --------------------------
# 2. TFT Model Architecture
# --------------------------

class TemporalFinancialTransformer(nn.Module):
    def __init__(self, input_size=12, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        # Feature embeddings
        self.ohlc_embed = nn.Linear(4, d_model//2)
        self.volume_embed = nn.Linear(1, d_model//4)
        self.time_embed = nn.Linear(7, d_model//4)
        
        # Transformer
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Single output head for returns
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

    def forward(self, ohlc, volume, time_features):
        # Feature embeddings
        ohlc_emb = self.ohlc_embed(ohlc)
        volume_emb = self.volume_embed(volume.unsqueeze(-1))
        time_emb = self.time_embed(time_features)
        
        # Concatenate and process
        x = torch.cat([ohlc_emb, volume_emb, time_emb], dim=-1)
        x = self.pos_encoder(x)
        
        # Transformer
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        
        # Prediction
        return self.output_head(x[:, -1, :]).squeeze()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# --------------------------
# 3. Training Infrastructure
# --------------------------

def train_model(train_loader, val_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    model = TemporalFinancialTransformer(
        input_size=config['input_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers']
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    criterion = nn.HuberLoss()
    
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            ohlc = batch['ohlc'].to(device)
            volume = batch['volume'].to(device)
            time_feats = batch['time_features'].to(device)
            targets = batch['target'].to(device)
            
            preds = model(ohlc, volume, time_feats)
            loss = criterion(preds, targets)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        val_loss = evaluate_model(model, val_loader, device, criterion)
        scheduler.step(val_loss)
        
        train_losses.append(epoch_loss/len(train_loader))
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{config["epochs"]}')
        print(f'Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Plot training
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.savefig('training_curve.png')
    
    return model

def evaluate_model(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            ohlc = batch['ohlc'].to(device)
            volume = batch['volume'].to(device)
            time_feats = batch['time_features'].to(device)
            targets = batch['target'].to(device)
            
            preds = model(ohlc, volume, time_feats)
            loss = criterion(preds, targets)
            
            total_loss += loss.item()
    
    return total_loss / len(loader)

# --------------------------
# 4. Main Execution
# --------------------------

import pandas as pd

def validate_csv(file_path):
    df = pd.read_csv(file_path)
    
    # Check required columns
    assert {'timestamp', 'open', 'high', 'low', 'close', 'volume'}.issubset(df.columns)
    
    # Validate OHLC relationships
    assert (df['high'] >= df['low']).all()
    assert (df['high'] >= df['close']).all()
    assert (df['low'] <= df['close']).all()
    
    # Check for NaNs
    assert not df[['open', 'high', 'low', 'close', 'volume']].isnull().any().any()
    
    # Verify chronological order
    assert pd.to_datetime(df['timestamp']).is_monotonic_increasing
    
    print("CSD validation passed!")

if __name__ == "__main__":
    # Load data
    
    data_path = 'data_21_25_sol.csv'
    validate_csv(data_path)
    df = pd.read_csv(data_path)  # Expected columns: timestamp, open, high, low, close, volume
    
    # Process data
    processor = DataProcessor(seq_length=60, pred_horizon=5)
    (train_seq, train_tgt), (val_seq, val_tgt) = processor.process_data(df)
    
    # Create dataloaders
    train_dataset = FinancialDataset(train_seq, train_tgt)
    val_dataset = FinancialDataset(val_seq, val_tgt)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Training config
    config = {
        'input_size': train_seq.shape[-1],
        'd_model': 64,
        'nhead': 4,
        'num_layers': 3,
        'lr': 1e-4,
        'epochs': 20,
    }
    
    print(f"Input size: {train_seq.shape[-1]}")  # Should be 12
    print(f"Time features count: {train_seq[0,0,5:].shape[0]}")  # Should be 7
    
    # Train model
    model = train_model(train_loader, val_loader, config)
    
    # Example prediction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample = next(iter(val_loader))

    with torch.no_grad():
        predictions = model(
            sample['ohlc'].to(device),
            sample['volume'].to(device),
            sample['time_features'].to(device)
        )

    # Print predictions
    for i in range(3):  # First 3 samples
        print(f"Sample {i+1}: Predicted Return = {predictions[i].item():.4f}")