# Stock Market Trend Prediction Web App

A Flask-based web application that predicts stock price trends using historical stock data and news sentiment analysis. The app fetches stock prices via Yahoo Finance, analyzes news headlines sentiment, predicts future trends with an XGBoost model, and visualizes results.

## 📌 Features
- Fetches historical stock price data (Yahoo Finance)
- Collects news headlines (NewsAPI or Google News RSS fallback)
- Performs sentiment analysis on headlines (VADER)
- Merges stock data with sentiment scores
- Builds predictive features using past price & sentiment
- Trains XGBoost regression model to predict stock prices
- Calculates RMSE and determines trend (up/down)
- Displays interactive plot of actual vs. predicted prices
- Returns JSON output via API endpoint

## 🛠️ Technologies
- Python 3.x
- Flask
- yfinance
- newsapi-python
- VADER Sentiment Analyzer
- Pandas, NumPy
- Matplotlib
- XGBoost
- Scikit-learn

## 💻 Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/Stock-Market-Trend-Prediction.git
````

2. Navigate to the project folder:

```bash
cd Stock-Market-Trend-Prediction
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. Run the Flask app:

```bash
python app.py
```

2. Open your browser and go to:

```
http://localhost:5000
```

3. Enter a stock ticker (e.g., `AAPL`) and optional NewsAPI key.
4. Click **Predict** to see trend predictions, RMSE, sentiment, and plot.

## 📝 API Endpoint

* **POST /predict**

  * Payload (JSON):

```json
{
  "ticker": "AAPL",
  "news_api_key": "YOUR_KEY",
  "debug": true
}
```

* Response (JSON):

```json
{
  "ticker": "AAPL",
  "headlines_count": 50,
  "sentiment": 0.12,
  "trend": "up",
  "rmse": 7.24,
  "plot_image": "data:image/png;base64,..."
}
```

## 📈 Example Output

* Predicted trend: `up` / `down`
* RMSE: `7.24`
* Average sentiment: `0.12`
* Interactive plot of actual vs. predicted prices

## 🔮 Future Improvements

* Support multiple tickers at once
* Add LSTM or Prophet models for more accurate predictions
* Deploy as a web dashboard with live updates
* Highlight trends and sentiment in interactive plots





