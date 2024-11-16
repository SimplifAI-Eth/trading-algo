import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import datetime
import asyncio
import aiohttp
from typing import Dict, List, Tuple
import threading
from flask import Flask, jsonify
import queue
import joblib
import os
from flask_cors import CORS
from pyngrok import ngrok


class CryptoDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)  # Changed to FloatTensor for regression
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class EnhancedLSTMTrader(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=4, dropout=0.3, device='cuda'):
        super(EnhancedLSTMTrader, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        # Bidirectional LSTM for better pattern recognition
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        ).to(device)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        ).to(device)
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 128).to(device)
        self.bn1 = nn.BatchNorm1d(128).to(device)
        self.fc2 = nn.Linear(128, 64).to(device)
        self.bn2 = nn.BatchNorm1d(64).to(device)
        self.fc3 = nn.Linear(64, 1).to(device)  # Single output for probability
        self.sigmoid = nn.Sigmoid()
    
    def attention_net(self, lstm_output):
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(self.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        attention_out = self.attention_net(lstm_out)
        
        out = self.dropout(attention_out)
        out = self.fc1(out)
        out = self.bn1(out)
        out = torch.relu(out)
        
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = torch.relu(out)
        
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

class PythPriceFeeder:
    def __init__(self, eth_price_id="0xff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace"):
        self.eth_price_id = eth_price_id
        self.base_url = "https://hermes.pyth.network/v2/updates/price/latest"
        self.last_price_data = None
        self.last_fetch_time = None
    
    async def get_latest_price(self) -> Dict:
        current_time = datetime.datetime.now(datetime.timezone.utc)
        
        # Rate limiting
        if (self.last_fetch_time is not None and 
            (current_time - self.last_fetch_time).total_seconds() < 1):
            return self.last_price_data
        
        url = f"{self.base_url}?ids%5B%5D={self.eth_price_id}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        price_info = data['parsed'][0]
                        
                        price = float(price_info['price']['price']) / (10 ** abs(price_info['price']['expo']))
                        conf = float(price_info['price']['conf']) / (10 ** abs(price_info['price']['expo']))
                        ema_price = float(price_info['ema_price']['price']) / (10 ** abs(price_info['ema_price']['expo']))
                        ema_conf = float(price_info['ema_price']['conf']) / (10 ** abs(price_info['ema_price']['expo']))
                        
                        self.last_price_data = {
                            'timestamp': current_time,
                            'price': price,
                            'conf': conf,
                            'ema_price': ema_price,
                            'ema_conf': ema_conf
                        }
                        self.last_fetch_time = current_time
                        return self.last_price_data
                    else:
                        raise Exception(f"Failed to fetch price. Status code: {response.status}")
            except Exception as e:
                print(f"Error fetching price: {e}")
                return self.last_price_data

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for feature engineering"""
    df = df.copy()
    
    # Price and EMA differences
    df['price_ema_diff'] = ((df['price'] - df['ema_price']) / df['ema_price']) * 100
    df['conf_ratio'] = df['conf'] / df['ema_conf']
    
    # Returns and Volatility
    df['returns'] = df['price'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(365 * 24 * 60)  # Annualized
    
    # Multi-timeframe RSI
    for period in [14, 28, 56]:
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi_14']  # Default RSI
    
    # MACD
    exp1 = df['price'].ewm(span=12).mean()
    exp2 = df['price'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # Bollinger Bands
    sma = df['price'].rolling(window=20).mean()
    std = df['price'].rolling(window=20).std()
    df['bollinger_upper'] = sma + (std * 2)
    df['bollinger_lower'] = sma - (std * 2)
    
    # Volume Profile (simulated based on price action)
    df['volume_profile'] = df['returns'].abs() * df['volatility']
    
    # Multi-timeframe momentum
    for period in [10, 30, 60]:
        df[f'momentum_{period}'] = df['price'].pct_change(periods=period)
    df['momentum'] = df['momentum_10']  # Default momentum
    
    return df

app = Flask(__name__)
CORS(app)

# Thread-safe queue to store the latest data
latest_data = queue.Queue(maxsize=1)

def setup_ngrok():
    """Start ngrok tunnel for exposing Flask app"""
    try:
        public_url = ngrok.connect(5000).public_url
        print(f"\n🌍 API is now globally accessible at:")
        print(f"{public_url}/api/eth-signal")
        print("\nShare this URL to access your trading signals from anywhere!")
        return public_url
    except Exception as e:
        print(f"Error setting up ngrok: {e}\nAPI will be available locally at http://localhost:5000/api/eth-signal")
        return None
    
def start_flask():
    """Start the Flask app"""
    try:
        setup_ngrok()
        app.run(host='0.0.0.0', port=5000)

    except Exception as e:
        print(f"Error starting Flask app: {e}")

def update_latest_data(timestamp, price, signal):
    """Update the latest data in the queue"""
    # Remove old data if queue is full
    if latest_data.full():
        try:
            latest_data.get_nowait()
        except queue.Empty:
            pass
    
    # Add new data
    data = {
        "timestamp": timestamp,
        "price": price,
        "signal": signal
    }
    latest_data.put(data)

@app.route('/api/eth-signal', methods=['GET'])
def get_eth_signal():
    """API endpoint to get the latest ETH data"""
    try:
        data = latest_data.get_nowait()
        latest_data.put(data)  # Put it back in the queue
        return jsonify(data)
    except queue.Empty:
        return jsonify({"error": "No data available"}), 404

class EnhancedCryptoTrader:
    def __init__(self, sequence_length=120, batch_size=64, update_interval=15):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = MinMaxScaler()
        self.buy_threshold = 0.6
        self.sell_threshold = 0.4
        self.update_interval = update_interval
        
        self.features = [
            'price', 'conf', 'ema_price', 'ema_conf',
            'price_ema_diff', 'conf_ratio', 'volatility',
            'rsi', 'rsi_28', 'rsi_56', 'momentum',
            'momentum_30', 'momentum_60', 'macd',
            'macd_signal', 'bollinger_upper', 'bollinger_lower',
            'volume_profile'
        ]
        
        self.price_history = pd.DataFrame()
        self.last_update = None
        
        # Load historical data and initialize model
        self.load_historical_data()
        self.initialize_model()
        self.initial_training()
    
    def load_historical_data(self):
        """Load and prepare historical data"""
        try:
            # Load historical data
            hist_df = pd.read_csv('eth_price_hist.csv')
            
            # Ensure required columns exist
            required_cols = ['price', 'conf', 'ema_price', 'ema_conf']
            if not all(col in hist_df.columns for col in required_cols):
                raise ValueError("Historical data missing required columns")
            
            # Calculate technical indicators
            hist_df = calculate_technical_indicators(hist_df)
            
            # Store in price history
            self.price_history = hist_df
            print(f"Loaded {len(hist_df)} historical data points")
            
        except Exception as e:
            print(f"Error loading historical data: {e}")
            raise
    
    def initialize_model(self):
        """Initialize the LSTM model"""
        input_size = len(self.features)
        self.model = EnhancedLSTMTrader(
            input_size=input_size,
            hidden_size=256,
            num_layers=4,
            dropout=0.3,
            device=self.device
        ).to(self.device)
    
    def initial_training(self):
        """Perform initial training with historical data"""
        if len(self.price_history) < self.sequence_length:
            raise ValueError("Not enough historical data for training")
        
        print("Starting initial training...")
        
        # Prepare data for training
        df_train = self.price_history.copy()
        
        # Calculate future returns and create targets
        df_train['future_return'] = df_train['price'].pct_change().shift(-1)
        df_train['target'] = (df_train['future_return'] > 0).astype(float)
        
        # Remove NaN values
        df_train = df_train.dropna()
        
        # Scale features
        scaled_data = self.scaler.fit_transform(df_train[self.features])
        
        # Create sequences
        X, y = [], []
        for i in range(len(df_train) - self.sequence_length):
            X.append(scaled_data[i:i + self.sequence_length])
            y.append(df_train['target'].iloc[i + self.sequence_length - 1])
        
        X = np.array(X)
        y = np.array(y)
        
        # Create dataset and loader
        dataset = CryptoDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Train model
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        criterion = nn.BCELoss()
        
        self.model.train()
        for epoch in range(10):  # Initial training with 10 epochs
            total_loss = 0
            for sequences, targets in loader:
                sequences = sequences.to(self.device)
                targets = targets.unsqueeze(1).to(self.device)  # Add dimension to match model output
                
                optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/10, Loss: {avg_loss:.4f}")
        
        print("Initial training completed")
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences and targets for training"""
        if len(df) <= self.sequence_length:
            return np.array([]), np.array([])
            
        try:
            # Calculate technical indicators
            df = calculate_technical_indicators(df)
            
            # Handle NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Create targets based on future returns
            df['future_return'] = df['price'].pct_change().shift(-1)  # Changed to next-tick return
            df['target'] = (df['future_return'] > 0).astype(float)
            
            # Remove any remaining NaN values
            df = df.dropna()
            
            if len(df) <= self.sequence_length:
                return np.array([]), np.array([])
            
            # Verify all features are present
            if not all(feature in df.columns for feature in self.features):
                missing_features = [f for f in self.features if f not in df.columns]
                print(f"Missing features in data preparation: {missing_features}")
                return np.array([]), np.array([])
            
            # Scale features with fit_transform only during training
            scaled_data = self.scaler.fit_transform(df[self.features])
            
            X, y = [], []
            for i in range(len(df) - self.sequence_length):
                sequence = scaled_data[i:i + self.sequence_length]
                if not np.any(np.isnan(sequence)):  # Only add sequence if it contains no NaN values
                    X.append(sequence)
                    y.append(df['target'].iloc[i + self.sequence_length - 1])
            
            if not X or not y:
                return np.array([]), np.array([])
                
            return np.array(X), np.array(y)
            
        except Exception as e:
            print(f"Error in data preparation: {e}")
            return np.array([]), np.array([])
    
    async def update_model(self):
        """Update model every 15 minutes"""
        current_time = datetime.datetime.now(datetime.timezone.utc)
        
        if (self.last_update is None or 
            (current_time - self.last_update).total_seconds() >= self.update_interval * 60):
            
            if len(self.price_history) >= self.sequence_length:
                recent_data = self.price_history.tail(1000)
                await self.train_incremental(recent_data)
                self.last_update = current_time
                print(f"Model updated at {current_time}")
    
    async def train_incremental(self, new_data: pd.DataFrame):
        """Incremental training on new data"""
        X, y = self.prepare_data(new_data)
        
        if len(X) == 0 or len(y) == 0:
            print("Not enough data for training")
            return
        
        dataset = CryptoDataset(X, y)
        
        # Only create DataLoader if we have enough samples
        if len(dataset) >= self.batch_size:
            loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True)
            
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
            criterion = nn.BCELoss()
            
            self.model.train()
            for _ in range(5):  # Quick update with 5 epochs
                for sequences, targets in loader:
                    sequences = sequences.to(self.device)
                    targets = targets.unsqueeze(1).to(self.device)  # Add dimension to match model output
                    
                    optimizer.zero_grad()
                    outputs = self.model(sequences)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
            print(f"Incremental training completed with {len(dataset)} samples")
        else:
            print(f"Not enough samples for training: {len(dataset)} < {self.batch_size}")
    
    async def predict(self, current_data: Dict) -> Dict:
        """Generate trading signal with confidence metrics"""
        try:
            # Add current data to price history
            df = pd.DataFrame([current_data])
            self.price_history = pd.concat([self.price_history, df]).tail(1000)  # Keep last 1000 points
            
            # Prepare latest sequence
            df_sequence = self.price_history.tail(self.sequence_length)
            df_prepared = calculate_technical_indicators(df_sequence)
            
            if len(df_prepared) < self.sequence_length:
                return {
                    'signal': 'WAIT',
                    'confidence': 0.0,
                    'metrics': self._calculate_metrics(calculate_technical_indicators(df)),
                    'buffer_size': len(df_prepared)
                }
            
            # Ensure all features are present and handle NaN values
            df_prepared = df_prepared.fillna(method='ffill').fillna(method='bfill')
            
            if not all(feature in df_prepared.columns for feature in self.features):
                missing_features = [f for f in self.features if f not in df_prepared.columns]
                print(f"Missing features: {missing_features}")
                return {
                    'signal': 'WAIT',
                    'confidence': 0.0,
                    'metrics': self._calculate_metrics(df_prepared),
                    'buffer_size': len(df_prepared)
                }
            
            # Use transform instead of fit_transform for prediction
            sequence = self.scaler.transform(df_prepared[self.features])
            
            # Ensure sequence has correct shape
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            # Generate prediction
            self.model.eval()
            with torch.no_grad():
                try:
                    probability = self.model(sequence_tensor).item()
                    print(f"Raw prediction probability: {probability}")  # Debug print
                    
                    # Ensure probability is valid
                    if np.isnan(probability):
                        print("Warning: NaN probability detected")
                        return {
                            'signal': 'WAIT',
                            'confidence': 0.0,
                            'metrics': self._calculate_metrics(df_prepared),
                            'buffer_size': len(df_prepared)
                        }
                    
                    # Determine signal
                    if probability > self.buy_threshold:
                        signal = 'BUY'
                    elif probability < self.sell_threshold:
                        signal = 'SELL'
                    else:
                        signal = 'HOLD'
                    
                    return {
                        'signal': signal,
                        'confidence': probability,
                        'metrics': self._calculate_metrics(df_prepared),
                        'buffer_size': len(df_prepared)
                    }
                except Exception as e:
                    print(f"Error during model prediction: {e}")
                    return {
                        'signal': 'WAIT',
                        'confidence': 0.0,
                        'metrics': self._calculate_metrics(df_prepared),
                        'buffer_size': len(df_prepared)
                    }
        except Exception as e:
            print(f"Error in prediction pipeline: {e}")
            return {
                'signal': 'WAIT',
                'confidence': 0.0,
                'metrics': self._calculate_metrics(df),
                'buffer_size': len(df_prepared) if 'df_prepared' in locals() else 0
            }
    
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate additional trading metrics"""
        metrics = {
            'price_ema_diff': 0.0,
            'rsi': 50.0,
            'volatility': 0.0,
            'momentum': 0.0,
            'macd': 0.0,
            'macd_signal': 0.0
        }
        
        if not df.empty:
            for key in metrics.keys():
                if key in df.columns:
                    metrics[key] = df[key].iloc[0]
        
        return metrics

async def main():
    try:
        # Initialize components
        price_feeder = PythPriceFeeder()
        trader = EnhancedCryptoTrader(
            sequence_length=120,  # 2 hours of 1-minute data
            batch_size=64,
            update_interval=15    # 15-minute updates
        )
        
        print("Starting trading bot...")
        
        while True:
            try:
                # Fetch latest price
                price_data = await price_feeder.get_latest_price()
                
                if price_data:
                    # Generate prediction
                    prediction = await trader.predict(price_data)
                    
                    # Update model if needed
                    await trader.update_model()

                    # Current time
                    current_time = datetime.datetime.now(datetime.timezone.utc)

                    # Update latest data for API
                    update_latest_data(current_time.strftime('%Y-%m-%d %H:%M:%S'), price_data['price'], prediction['signal'])
                    
                    # Clear screen for better visibility
                    print("\033[H\033[J")
                    
                    # Display results
                    print("\n=== Trading Signal Update ===")
                    print(f"Time (UTC): {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Current ETH Price: ${price_data['price']:.2f}")
                    print(f"EMA Price: ${price_data['ema_price']:.2f}")
                    
                    print("\nSignal Analysis:")
                    print(f"Trading Signal: {prediction['signal']}")
                    print(f"Confidence: {prediction['confidence']:.2f}")
                    
                    print("\nTechnical Indicators:")
                    metrics = prediction['metrics']
                    print(f"Price-EMA Difference: {metrics['price_ema_diff']:.2f}%")
                    print(f"RSI: {metrics['rsi']:.2f}")
                    print(f"Volatility: {metrics['volatility']:.4f}")
                    print(f"Momentum: {metrics['momentum']:.4f}")
                    print(f"MACD: {metrics['macd']:.4f}")
                    print(f"MACD Signal: {metrics['macd_signal']:.4f}")
                    
                    print("\nData Buffer Status:")
                    print(f"Current Size: {prediction['buffer_size']}/{trader.sequence_length}")
                    if prediction['buffer_size'] < trader.sequence_length:
                        print(f"Still accumulating data... Need {trader.sequence_length - prediction['buffer_size']} more points")
                    
                    if trader.last_update:
                        next_update = trader.last_update + datetime.timedelta(minutes=trader.update_interval)
                        time_to_update = next_update - current_time
                        print(f"\nNext Model Update in: {time_to_update.seconds // 60} minutes")
                    
                # Wait for next update (5 seconds)
                await asyncio.sleep(5)
                
            except Exception as e:
                print(f"Error in trading loop: {e}")
                await asyncio.sleep(10)
                continue
    
    except Exception as e:
        print(f"Critical error in main loop: {e}")
    finally:
        print("Trading bot stopped")

if __name__ == "__main__":
    # Start the Flask server in a separate thread
    threading.Thread(target=start_flask, daemon=True).start()
    asyncio.run(main())
