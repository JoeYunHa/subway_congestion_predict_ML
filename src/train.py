import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# settings
DATA_PATH = "../data/congestion_total.csv"
STATION_PATH = "../data/subway_station.csv"
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

# load data and preprocess
print("Loading Data")
congestion_df = pd.read_csv(DATA_PATH)
station_df = pd.read_csv(STATION_PATH)

print(f"Congestion data shape: {congestion_df.shape}")
print(f"Station data shape: {station_df.shape}")

# Wide -> Long conversion
id_vars = ["요일구분", "호선", "역번호", "출발역", "상하구분"]
value_vars = [c for c in congestion_df.columns if c not in id_vars]
df_long = congestion_df.melt(
    id_vars=id_vars, value_vars=value_vars, var_name="time_str", value_name="congestion"
)

# 결측치 제거 -> 0은 우선 혼잡도 0으로 처리, 추후 학습한 패턴 관찰 후 필요에 따라 수정
df_long = df_long.dropna(subset=["congestion"])
df_long = df_long[df_long["congestion"] >= 0].copy()

# 복합키 생성
print("\ncreating composite key")

# data 정규화
df_long["station_code_str"] = df_long["역번호"].astype(str).str.strip()
df_long["direction"] = df_long["상하구분"].astype(str).str.strip()

# composite key 생성
df_long["station_direction_key"] = (
    df_long["station_code_str"] + "_" + df_long["direction"]
)

unique_stations = df_long["station_direction_key"].nunique()
print(f"Unique station-direction combinations: {unique_stations}")

# Feature Engineering
print("\nFeature Engineering")


# 시간 파싱
def parse_time(t_str):
    try:
        if ":" in str(t_str):
            h, m = str(t_str).split(":")
            return int(h), int(m)
        return 0, 0
    except:
        return 0, 0


time_parsed = df_long["time_str"].apply(parse_time)
df_long["hour"] = [t[0] for t in time_parsed]
df_long["minute"] = [t[1] for t in time_parsed]

# 순환형 시간 인코딩 for 24시간제
df_long["hour_sin"] = np.sin(2 * np.pi * df_long["hour"] / 24)
df_long["hour_cos"] = np.cos(2 * np.pi * df_long["hour"] / 24)
df_long["minute_sin"] = np.sin(2 * np.pi * df_long["minute"] / 60)
df_long["minute_cos"] = np.cos(2 * np.pi * df_long["minute"] / 60)

# 요일 및 주말
day_mapping = {"평일": 0, "토요일": 5, "일요일": 6, "공휴일": 6}
df_long["day_type"] = df_long["요일구분"].map(day_mapping).fillna(0).astype(int)
df_long["is_weekend"] = (df_long["day_type"] >= 5).astype(int)

# 역 좌표 병합
station_df["station_cd"] = station_df["station_cd"].astype(str).str.strip()
df_long = df_long.merge(
    station_df[["station_cd", "lat", "lng"]],
    left_on="station_code_str",
    right_on="station_cd",
    how="left",
)
# 좌표 없음 => -1 처리
df_long["lat"] = df_long["lat"].fillna(-1)
df_long["lng"] = df_long["lng"].fillna(-1)
df_long["has_coordinates"] = (df_long["lat"] != -1).astype(int)

# 출퇴근 시간대 & 순환선 여부
df_long["is_morning_rush"] = ((df_long["hour"] >= 7) & (df_long["hour"] <= 9)).astype(
    int
)
df_long["is_evening_rush"] = ((df_long["hour"] >= 18) & (df_long["hour"] <= 20)).astype(
    int
)
df_long["is_circular"] = df_long["direction"].isin(["내선", "외선"]).astype(int)

# train/test split
train_df, test_df = train_test_split(
    df_long, test_size=0.2, random_state=42, stratify=df_long["요일구분"]
)

# target encoding -> 역별 평균 혼잡도 추가
print("\napplying target encoding (mean congestion by station)")

station_mean_map = train_df.groupby("station_direction_key")["congestion"].mean()

# Train에 매핑
train_df["station_mean_congestion"] = train_df["station_direction_key"].map(
    station_mean_map
)

# Test에 매핑 (Train에 없던 새로운 역이 나오면 전체 평균값으로 대체)
global_mean = train_df["congestion"].mean()
test_df["station_mean_congestion"] = (
    test_df["station_direction_key"].map(station_mean_map).fillna(global_mean)
)

print("Target encoding added.")

# unseen label 처리
print("\nencoding categorical features")

categorical_cols = ["station_direction_key", "호선"]
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    # Train 데이터로만 fit
    train_df.loc[:, f"{col}_encoded"] = le.fit_transform(train_df[col].astype(str))

    # Test 데이터 transform (Unknown -> -1)
    test_encoded = []
    classes_set = set(le.classes_)  # 검색 속도 향상
    for val in test_df[col].astype(str):
        if val in classes_set:
            test_encoded.append(le.transform([val])[0])
        else:
            test_encoded.append(-1)  # 0 대신 -1 사용

    test_df.loc[:, f"{col}_encoded"] = test_encoded
    encoders[col] = le

# train
feature_cols = [
    "station_mean_congestion",
    "station_direction_key_encoded",
    "호선_encoded",
    "day_type",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
    "is_morning_rush",
    "is_evening_rush",
    "is_circular",
    "lat",
    "lng",
    "has_coordinates",
]
target_col = "congestion"

X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]
y_test = test_df[target_col]

print("\nTraining XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=7,
    learning_rate=0.05,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)

model.fit(
    X_train, y_train, eval_set=[(X_test, y_test)], verbose=100, early_stopping_rounds=50
)

# evaluate & save model
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"\n[Test Result] RMSE: {rmse:.4f}, R2: {r2:.4f}")

# 모델 저장
model.save_model(os.path.join(MODEL_DIR, "congestion_model.json"))

# 메타데이터 저장
metadata = {
    "encoders": encoders,
    "feature_cols": feature_cols,
    "day_mapping": day_mapping,
    "unique_stations": unique_stations,
    "key_format": "{station_code}_{direction}",
    "station_mean_map": station_mean_map.to_dict(),
    "global_mean": global_mean,
}
joblib.dump(metadata, os.path.join(MODEL_DIR, "model_metadata.pkl"))

print("\nAll process completed successfully")
