# Shanghai Second-Hand House Price Prediction 🏠

A lightweight regression project using real Shanghai housing data  
to predict second-hand house prices based on structural features.

---

## Dataset
- Source: [Shanghai Second-Hand House Dataset (Kaggle)](https://www.kaggle.com/datasets/vnvile/shanghai-second-hand-house-dataset)
- Format: Parquet
- Unit: CNY

Main features:
- Area
- Room / Hall / Toilet count
- Year built
- Renovation
- Floor location
- District

Target:
- `listing_total_price` (total house price)

---

## Problem Statement
Predict the total price of a second-hand apartment  
given its structural characteristics.

---

## Methodology
- Data cleaning (minimal missing values removed)
- Feature selection (structural features only)
- One-hot encoding for categorical variables
- Train / Test split (80% / 20%)
- Model: Random Forest Regressor

---

## Results 

| Metric | Value |
|------|------|
| MAE  | 1.28 |
| RMSE | 2.79 |
| R²   | 0.762 |

Interpretation:
- On average, predictions are within **≈ 128,000 CNY** of the true price.
- The model explains **76.2%** of the variance in housing prices.

---

## Key Insights
Feature importance analysis shows that:
- **Area**
- **Location (District)**
- **Year Built**

are the most influential factors in determining house prices.

---

## Visualizations
- Actual vs Predicted Scatter Plot
- Feature Importance Bar Chart

---

## Usage

- For learning 