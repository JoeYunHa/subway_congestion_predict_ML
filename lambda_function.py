import json
import os
import sys
import logging
from datetime import datetime, timedelta
import boto3
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import psycopg2
from io import BytesIO

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 환경 변수
DB_HOST = os.environ.get("DB_HOST")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_NAME = os.environ.get("DB_NAME")
DB_PORT = os.environ.get("DB_PORT", "5432")

S3_BUCKET = os.environ.get("S3_BUCKET")
MODEL_KEY = os.environ.get("MODEL_KEY", "congestion_model.json")
META_KEY = os.environ.get("META_KEY", "model_metadata.pkl")

# 로컬 임시 경로 => 필요할 경우 추가
LOCAL_MODEL_PATH = "/tmp/congestion_model.json"
LOCAL_META_PATH = "/tmp/model_metadata.pkl"

s3_client = boto3.client("s3")

# warm start
model = None
metadata = None


def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            dbname=DB_NAME,
            port=DB_PORT,
            connect_timeout=5,
        )
        return conn
    except Exception as e:
        logger.error(f"DB Connection Error: {e}")
        raise e


def load_artifacts():
    """모델과 메타데이터 로드"""
    global model, metadata
    if model is not None and metadata is not None:
        return model, metadata

    # S3 다운로드
    if not os.path.exists(LOCAL_MODEL_PATH):
        logger.info("Downloading model from S3...")
        s3_client.download_file(S3_BUCKET, MODEL_KEY, LOCAL_MODEL_PATH)

    if not os.path.exists(LOCAL_META_PATH):
        logger.info("Downloading metadata from S3...")
        s3_client.download_file(S3_BUCKET, META_KEY, LOCAL_META_PATH)

    # 로드
    logger.info("Loading model & metadata...")
    model = xgb.XGBRegressor()
    model.load_model(LOCAL_MODEL_PATH)
    metadata = joblib.load(LOCAL_META_PATH)

    return model, metadata


def get_station_info_from_db(conn):
    """
    DB에서 모든 역의 기본 정보(호선, 좌표)를 조회하여 딕셔너리로 반환
    Return: {'150': {'line': '1호선', 'lat': 37.55, 'lng': 126.97}, ...}
    """
    station_info = {}
    sql = "SELECT station_cd, line, lat, lng FROM subway_station"

    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
        for r in rows:
            # station_cd를 문자열로 변환 및 공백 제거 (학습시 로직과 동일하게)
            code = str(r[0]).strip()
            station_info[code] = {
                "line": str(r[1]).strip(),
                "lat": float(r[2]) if r[2] else -1.0,
                "lng": float(r[3]) if r[3] else -1.0,
            }
    return station_info


def prepare_features(target_time, station_info_map, metadata):
    """
    예측할 입력 데이터(DataFrame) 생성
    학습 시 사용한 Feature 순서와 로직을 정확히 따라야 함
    """
    # 1. 예측 대상 목록 생성
    # 모델이 학습한 '역_방향' 조합(keys)에 대해서만 예측 가능
    station_mean_map = metadata["station_mean_map"]
    target_keys = list(station_mean_map.keys())  # 예: ['150_상선', '150_하선', ...]

    data = []

    # 시간 변수 계산
    hour = target_time.hour
    minute = target_time.minute
    weekday = target_time.weekday()  # 0=월, 6=일

    # 시간 순환 인코딩
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    minute_sin = np.sin(2 * np.pi * minute / 60)
    minute_cos = np.cos(2 * np.pi * minute / 60)

    # 요일/휴일/러시아워
    day_mapping = metadata["day_mapping"]
    # 오늘이 무슨 요일인지 (평일/토/일) - 단순화 로직
    if weekday >= 5:
        day_str = "토요일" if weekday == 5 else "일요일"
    else:
        day_str = "평일"

    day_type = day_mapping.get(day_str, 0)
    is_weekend = 1 if day_type >= 5 else 0

    is_morning_rush = 1 if 7 <= hour <= 9 else 0
    is_evening_rush = 1 if 18 <= hour <= 20 else 0

    # 2. 데이터 구성 Loop
    for key in target_keys:
        # key format: "StationCode_Direction" (예: 150_상선)
        try:
            code, direction = key.split("_")
        except ValueError:
            continue  # 형식이 맞지 않으면 패스

        # DB에서 역 정보 조회 (없으면 기본값)
        info = station_info_map.get(code, {"line": "Unknown", "lat": -1, "lng": -1})

        row = {
            "station_direction_key": key,
            "station_code": code,  # 나중에 DB 저장용
            "direction": direction,  # 나중에 DB 저장용
            # Feature: Station Mean Congestion (Target Encoding)
            "station_mean_congestion": station_mean_map.get(
                key, metadata["global_mean"]
            ),
            # Feature: Encoders
            "station_direction_key_raw": key,  # 인코딩 전
            "호선_raw": info["line"],  # 인코딩 전
            # Feature: Time & Date
            "day_type": day_type,
            "is_weekend": is_weekend,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "minute_sin": minute_sin,
            "minute_cos": minute_cos,
            "is_morning_rush": is_morning_rush,
            "is_evening_rush": is_evening_rush,
            # Feature: Etc
            "is_circular": 1 if direction in ["내선", "외선"] else 0,
            "lat": info["lat"],
            "lng": info["lng"],
            "has_coordinates": 1 if info["lat"] != -1 else 0,
        }
        data.append(row)

    df = pd.DataFrame(data)

    # 3. 인코딩 적용 (Unseen Label 처리)
    encoders = metadata["encoders"]

    def safe_transform(encoder, values):
        # 학습된 클래스에 없으면 -1 반환
        classes = set(encoder.classes_)
        return [encoder.transform([v])[0] if v in classes else -1 for v in values]

    df["station_direction_key_encoded"] = safe_transform(
        encoders["station_direction_key"], df["station_direction_key_raw"]
    )
    df["호선_encoded"] = safe_transform(encoders["호선"], df["호선_raw"])

    # 4. 컬럼 순서 정렬 (학습 시와 동일하게)
    feature_cols = metadata["feature_cols"]
    return df[feature_cols], df[["station_code", "direction"]]  # X, 메타정보 반환


def lambda_handler(event, context):
    conn = None
    try:
        # 1. 모델 로드
        clf, meta = load_artifacts()

        # 2. 예측 시간 설정 (KST 기준 다음 시간 정각)
        # Lambda는 기본 UTC이므로 +9시간
        kst_now = datetime.utcnow() + timedelta(hours=9)
        target_time = kst_now + timedelta(hours=1)
        target_time = target_time.replace(minute=0, second=0, microsecond=0)

        logger.info(f"Predicting for time: {target_time}")

        # 3. DB 연결 및 역 정보 조회
        conn = get_db_connection()
        station_info_map = get_station_info_from_db(conn)

        # 4. 입력 데이터 준비
        X_pred, meta_df = prepare_features(target_time, station_info_map, meta)

        if X_pred.empty:
            return {"statusCode": 200, "body": "No targets to predict."}

        # 5. 예측 수행
        preds = clf.predict(X_pred)

        # 6. 결과 저장 (Upsert)
        # congestion_forecast 테이블 구조: (station_cd, direction, forecast_time, congestion, created_at)
        # PK는 (station_cd, direction, forecast_time) 복합키 권장

        data_to_insert = []
        for (idx, row), pred_val in zip(meta_df.iterrows(), preds):
            val = round(float(pred_val), 2)
            val = max(0.0, min(100.0, val))  # 0~100 사이로 클립

            # (station_code, direction, time, congestion)
            data_to_insert.append(
                (row["station_code"], row["direction"], target_time, val)
            )

        upsert_sql = """
            INSERT INTO congestion_forecast 
            (station_cd, direction, forecast_time, congestion_level, created_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (station_cd, direction, forecast_time) 
            DO UPDATE SET 
                congestion_level = EXCLUDED.congestion_level,
                created_at = NOW();
        """

        with conn.cursor() as cur:
            cur.executemany(upsert_sql, data_to_insert)
        conn.commit()

        msg = f"Successfully saved {len(data_to_insert)} predictions for {target_time}"
        logger.info(msg)

        return {"statusCode": 200, "body": json.dumps(msg)}

    except Exception as e:
        logger.error(f"Critical Error: {e}")
        if conn:
            conn.rollback()
        return {"statusCode": 500, "body": str(e)}

    finally:
        if conn:
            conn.close()
