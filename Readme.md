# 📈 Stock Price Prediction System (MSFT)

An end-to-end **machine learning pipeline** that predicts the **next-day closing price** of Microsoft (MSFT) stock using historical market data, feature engineering, and a Random Forest regression model.


---

## 🚀 Project Overview

This project demonstrates how to:

- Ingest real-world financial market data using the **Polygon.io API**
- Perform fast feature engineering using **Polars**
- Train a **time-series–aware Random Forest regression model**
- Evaluate performance using **Mean Absolute Error (MAE)**
- Visualize **actual vs predicted stock prices**
- Generate a **next-day price prediction**

> ⚠️ This project is **educational** and focuses on ML & data engineering practices, not financial advice.

---

## 🧠 Problem Statement

Given historical daily OHLCV data for MSFT, predict the **next trading day’s closing price** using only past information (no data leakage).

---

## 🧰 Tech Stack

- **Python 3.10+**
- **Polygon.io** – Market data source
- **Polars** – High-performance DataFrame operations
- **Scikit-learn** – Machine learning (Random Forest Regressor)
- **Matplotlib** – Visualization
- **python-dotenv** – Secure environment variable handling
