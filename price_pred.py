import pandas as pd
df=pd.read_parquet(r"dataset.parquet")
# print(df.columns)
#选特征
features = [
    "listing_total_price",
    "area",
    "room_count",
    "hall_count",
    "toilet_count",
    "year_built",
    "renovation",
    "floor_location",
    "district"
]

df_small = df[features].copy()
print(df_small.head())
# print(df_small.isnull().sum())

df_clean = df_small.dropna(subset=["year_built", "floor_location"])
# print(df_clean.shape)

print(df_clean["renovation"].value_counts())
print(df_clean["district"].value_counts().head())
#编码
df_encoded = pd.get_dummies(
    df_clean,
    columns=["renovation", "district","floor_location"],
    drop_first=True  # 防止多重共线性
)
print(df_encoded.head())

#训练模型
X = df_encoded.drop("listing_total_price", axis=1)
y = df_encoded["listing_total_price"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(list(zip(y_test[:5], y_pred[:5])))

#评估
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_housing_model(y_true, y_pred, scale=1.0, unit="CNY"):
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("📊 Model Evaluation")
    print(f"MAE  : {mae / scale:.2f} ({unit})")
    print(f"RMSE : {rmse / scale:.2f} ({unit})")
    print(f"R²   : {r2:.3f}")


evaluate_housing_model(y_test, y_pred, scale=1, unit="CNY")
