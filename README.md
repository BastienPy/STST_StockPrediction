# STST Stock Prediction

## Description

This project implements a spatiotemporal time series prediction model for stock market data using the STST architecture.

## Installation

1. Clone this repository:
   ```bash
   git clone <REPOSITORY_URL>
   cd STST_StockPrediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

1. Configure the parameters in the main script `stst.py`.
2. Start the training:
   ```bash
   python stst.py
   ```

### Temporal Encoding with Date2Vec

The Date2Vec model is used to generate temporal embeddings. You can use a pre-trained model or train a new one by following the instructions in the `Date2Vec` folder.

## Project Structure

```
STST_StockPrediction/
├── Date2Vec/               # Date2Vec implementation
├── data/                   # Raw and preprocessed data
├── models/                 # Saved models
├── stst.py                 # Main script
├── dataset.py              # Data handling and sliding windows
├── README.md               # Documentation
└── requirements.txt        # Python dependencies
```

## Results

The training results, including loss and accuracy curves, are saved as graphs in the output folder.
