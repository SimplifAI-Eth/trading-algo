This project is a trading algorithm using LSTM model designed for the ETHGlobal Bangkok project.

It includes a trading bot that fetches real-time data from Pyth Network Hermes API, and uses the LSTM model to output a BUY/SELL/HOLD signal.

It calculates the trading signal through various market indicators, and the LSTM model is trained on historical data.

Key components:
- `CryptoDataset`: A PyTorch dataset class for handling sequence and target.
- `EnchancedLSTMTrade`: A PyTorch neural network model with an LSTM and attention mechanism for predicting trading signals.
- `PythPriceFeeder`: A class for fetching real-time price data from the Pyth Network Hermes API.
- `calculate_technical_indicators`: A function for calculating technical indicators for feature engineering.
- `EnchancedCryptoTrader`: A class that manages data loading, model training, and trading signal prediction.
- `Flask API`: A RESTful API endpoint that get the trading signal from the LSTM model.
