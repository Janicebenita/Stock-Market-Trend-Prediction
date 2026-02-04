import yfinance as yf
try:
    from newsapi import NewsApiClient
except Exception:
    try:
        from newsapi.newsapi_client import NewsApiClient
    except Exception:
        NewsApiClient = None
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from flask import Flask, render_template, request, jsonify
import io, base64
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import urllib.parse
import feedparser

# 1. Data Collection
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    # Reset index so the date becomes a column
    data = data.reset_index()
    # If yfinance returned MultiIndex columns (e.g., for multiple tickers), flatten them
    if getattr(data.columns, 'nlevels', 1) > 1:
        data.columns = ['_'.join([str(c) for c in col]).strip() for col in data.columns.values]
    # Ensure a column named 'Date' exists
    if 'Date' not in data.columns:
        # try common alternatives
        for alt in ['date', 'index', 'Datetime']:
            if alt in data.columns:
                data.rename(columns={alt: 'Date'}, inplace=True)
                break
        else:
            # fallback: find first datetime-like column and rename it
            for col in data.columns:
                if pd.api.types.is_datetime64_any_dtype(data[col]):
                    data.rename(columns={col: 'Date'}, inplace=True)
                    break
    return data

def fetch_news_headlines(query, from_date, to_date, api_key):
    # Try to fetch using the NewsAPI client
    if NewsApiClient is None:
        raise RuntimeError('NewsApiClient is not available (newsapi-python not installed)')
    newsapi = NewsApiClient(api_key=api_key)
    all_articles = []
    page = 1
    while True:
        res = newsapi.get_everything(q=query, language='en', from_param=from_date, to=to_date, page=page, page_size=100, sort_by='publishedAt')
        if res.get('status') != 'ok' or not res.get('articles'):
            break
        all_articles.extend(res['articles'])
        if len(res['articles']) < 100:
            break
        page += 1
    headlines = [(article.get('title', ''), article.get('publishedAt', '')[:10]) for article in all_articles]
    return pd.DataFrame(headlines, columns=['headline', 'date'])


def fetch_news_headlines_fallback(query, from_date, to_date, max_items=100):
    """Fallback: fetch headlines from Google News RSS for the query.
    This does not require an API key. It may return different coverage than NewsAPI.
    """
    q = urllib.parse.quote_plus(query)
    rss_url = f'https://news.google.com/rss/search?q={q}'
    feed = feedparser.parse(rss_url)
    entries = feed.get('entries', [])[:max_items]
    rows = []
    for e in entries:
        title = e.get('title', '')
        pub = e.get('published', '') or e.get('published_parsed')
        # try to normalize date string
        date_str = ''
        try:
            if 'published' in e:
                # published is a string; take date part
                date_str = e.published.split(',')[-1].strip().split(' ')[0]
            elif 'published_parsed' in e and e.published_parsed:
                # build YYYY-MM-DD
                t = e.published_parsed
                date_str = f"{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d}"
        except Exception:
            date_str = ''
        rows.append((title, date_str))
    df = pd.DataFrame(rows, columns=['headline', 'date'])
    # try to filter by date range if possible
    try:
        if not df['date'].replace('', pd.NA).isna().all():
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            start = pd.to_datetime(from_date)
            end = pd.to_datetime(to_date)
            df = df[(df['date'] >= start) & (df['date'] <= end)]
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    except Exception:
        pass
    return df

# 2. Sentiment Analysis
def analyze_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df['headline'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df['date'] = pd.to_datetime(df['date'])
    return df

# 3. Data Preparation
def merge_data(stock_df, sentiment_df):
    # Ensure sentiment_df has the expected columns
    if sentiment_df is None:
        sentiment_df = pd.DataFrame(columns=['headline', 'date', 'sentiment'])
    else:
        if 'sentiment' not in sentiment_df.columns:
            sentiment_df = sentiment_df.copy()
            sentiment_df['sentiment'] = 0.0
        if 'date' not in sentiment_df.columns:
            # try to find a date-like column
            for col in sentiment_df.columns:
                if 'date' in str(col).lower():
                    sentiment_df = sentiment_df.rename(columns={col: 'date'})
                    break

    # prepare sentiment daily
    sentiment_daily = sentiment_df.groupby('date')['sentiment'].mean().reset_index()

    # Workaround for MultiIndex or non-standard columns in stock_df
    stock = stock_df.copy()
    if getattr(stock.columns, 'nlevels', 1) > 1:
        stock.columns = ['_'.join([str(c) for c in col]).strip() for col in stock.columns.values]

    # Ensure stock has a 'Date' column to merge on
    if 'Date' not in stock.columns:
        # look for a column name containing 'date'
        date_col = None
        for col in stock.columns:
            if 'date' in str(col).lower():
                date_col = col
                break
        if date_col is None:
            # try to reset index and find datetime-like column
            stock = stock.reset_index()
            for col in stock.columns:
                if pd.api.types.is_datetime64_any_dtype(stock[col]):
                    date_col = col
                    break
        if date_col:
            stock.rename(columns={date_col: 'Date'}, inplace=True)

    # Ensure date columns are datetime
    if 'Date' in stock.columns:
        stock['Date'] = pd.to_datetime(stock['Date'])
    if 'date' in sentiment_daily.columns:
        sentiment_daily['date'] = pd.to_datetime(sentiment_daily['date'])

    merged = pd.merge(stock, sentiment_daily, how='left', left_on='Date', right_on='date')
    merged['sentiment'].fillna(0, inplace=True)

    # Determine a suitable price column. Prefer 'Close' or any column containing 'close'.
    price_col = None
    for col in merged.columns:
        if str(col).lower() == 'close':
            price_col = col
            break
    if price_col is None:
        for col in merged.columns:
            if 'close' in str(col).lower():
                price_col = col
                break

    # If still not found, pick a numeric column that is not 'sentiment' or the date
    if price_col is None:
        numeric_cols = [c for c in merged.columns if c not in ('Date', 'date', 'sentiment') and pd.api.types.is_numeric_dtype(merged[c])]
        if numeric_cols:
            price_col = numeric_cols[0]

    if price_col is None:
        # No price column found; return merged with sentiment so caller can handle the error
        return merged[['Date', 'sentiment']]

    # Ensure there is a 'Close' column for downstream code
    if 'Close' not in merged.columns or price_col != 'Close':
        merged['Close'] = merged[price_col]

    # Build output columns: Date, standard OHLCV where available, Close, sentiment
    cols = ['Date']
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if c in merged.columns:
            cols.append(c)
    cols.append('sentiment')
    # ensure unique and existing
    cols = [c for c in cols if c in merged.columns]
    return merged[cols]

# 4. Feature Creation for Prediction (using past 5 days of features)
def create_features(data, window=5):
    # auto-detect close/price column and sentiment column (handle flattened names)
    cols = list(data.columns)
    close_col = None
    sentiment_col = None
    for c in cols:
        cname = str(c).lower()
        if cname == 'close' or cname.endswith('_close') or cname.endswith('close'):
            close_col = c
            break
    for c in cols:
        if 'sentiment' in str(c).lower():
            sentiment_col = c
            break

    # fallback: pick a numeric column as price if Close not found
    if close_col is None:
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(data[c])]
        if numeric_cols:
            # prefer columns named 'adj close' or 'adj_close'
            for c in numeric_cols:
                if 'adj' in str(c).lower() and 'close' in str(c).lower():
                    close_col = c
                    break
            if close_col is None:
                close_col = numeric_cols[0]

    if close_col is None:
        raise ValueError('No numeric price column found (expected Close). Columns: ' + ','.join(map(str, cols)))

    # if sentiment missing, create a zero-filled sentiment column
    if sentiment_col is None:
        data = data.copy()
        data['sentiment'] = 0.0
        sentiment_col = 'sentiment'

    X, y = [], []
    for i in range(window, len(data)):
        window_df = data.iloc[i-window:i]
        features = window_df[[close_col, sentiment_col]].values.flatten()
        X.append(features)
        y.append(data.iloc[i][close_col])
    return np.array(X), np.array(y)

# 5. Model Training and Prediction
def train_model(X_train, y_train):
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    return model.predict(X_test)

app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json() or {}
    ticker = (data.get('ticker') or '').strip()
    news_api_key = data.get('news_api_key')
    debug_flag = bool(data.get('debug', False))
    if not ticker:
        return jsonify(error='ticker is required'), 400

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)

    try:
        stock_df = fetch_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    except Exception as e:
        return jsonify(error=f'error fetching stock data: {e}'), 500

    # If yfinance failed to return usable data, fail early with a clear message
    if stock_df is None or stock_df.empty:
        return jsonify(error=f'No stock data found for ticker "{ticker}". The ticker may be invalid, delisted, or yfinance could not retrieve data.'), 400

    # Ensure there is at least one numeric column to use as price
    numeric_cols = [c for c in stock_df.columns if pd.api.types.is_numeric_dtype(stock_df[c])]
    if not numeric_cols:
        return jsonify(error=f'No numeric price columns found for ticker "{ticker}". Columns returned: {list(stock_df.columns)}'), 400

    headlines_count = 0
    sentiment_mean = 0.0
    sentiment_df = pd.DataFrame(columns=['headline', 'date'])

    # Try NewsAPI when key and client available; otherwise fall back to Google News RSS
    news_df = pd.DataFrame(columns=['headline', 'date'])
    if NewsApiClient is not None and news_api_key:
        try:
            news_df = fetch_news_headlines(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), news_api_key)
        except Exception:
            # fallback to RSS if NewsAPI fails
            try:
                news_df = fetch_news_headlines_fallback(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            except Exception:
                news_df = pd.DataFrame(columns=['headline', 'date'])
    else:
        try:
            news_df = fetch_news_headlines_fallback(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        except Exception:
            news_df = pd.DataFrame(columns=['headline', 'date'])

    headlines_count = len(news_df)
    sentiment_mean = 0.0
    if headlines_count > 0:
        try:
            sentiment_df = analyze_sentiment(news_df)
            sentiment_mean = float(sentiment_df['sentiment'].mean())
        except Exception:
            sentiment_df = pd.DataFrame(columns=['headline', 'date', 'sentiment'])
            sentiment_mean = 0.0

    merged_df = merge_data(stock_df, sentiment_df)
    if debug_flag:
        print('\n[DEBUG] merged_df.columns =', list(merged_df.columns))
        try:
            print('[DEBUG] merged_df.head():\n', merged_df.head().to_string())
        except Exception:
            print('[DEBUG] could not print merged_df.head()')
    X, y = create_features(merged_df)
    if debug_flag:
        print('[DEBUG] X.shape, y.shape =', getattr(X, 'shape', None), getattr(y, 'shape', None))
        try:
            print('[DEBUG] X[:3] =', X[:3].tolist())
            print('[DEBUG] y[:6] =', y[:6].tolist())
        except Exception:
            pass
    if len(X) == 0:
        return jsonify(error='Not enough data to create features. Try a longer date range.'), 400

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = train_model(X_train, y_train)
    preds = predict(model, X_test)
    if debug_flag:
        try:
            print('[DEBUG] preds[:10] =', preds[:10].tolist())
            print('[DEBUG] feature_importances =', getattr(model, 'feature_importances_', None))
        except Exception:
            pass
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    trend = 'up' if preds.mean() > y_test.mean() else 'down'

    plt.figure(figsize=(8, 4))
    plt.plot(range(len(y_test)), y_test, label='Actual Price')
    plt.plot(range(len(preds)), preds, label='Predicted Price')
    plt.xlabel('Test Set Days')
    plt.ylabel('Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.legend()
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    img_uri = 'data:image/png;base64,' + img_b64

    resp = dict(ticker=ticker, headlines_count=headlines_count, sentiment=sentiment_mean, trend=trend, rmse=rmse, plot_image=img_uri)
    if debug_flag:
        # include small debug snapshot (safe for JSON)
        try:
            resp['debug'] = {
                'merged_columns': list(merged_df.columns),
                'X_shape': getattr(X, 'shape', None),
                'y_shape': getattr(y, 'shape', None),
                'y_preview': (y[:10].tolist() if len(y) else []),
                'preds_preview': (preds[:10].tolist() if len(preds) else []),
                'feature_importances': (getattr(model, 'feature_importances_', None).tolist() if getattr(model, 'feature_importances_', None) is not None else None)
            }
        except Exception:
            resp['debug'] = 'error collecting debug info'

    return jsonify(resp)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)