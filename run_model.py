import logging
import numpy as np
import os
import pandas as pd
from PyEMD import CEEMDAN
from multiprocessing import Pool
from tqdm import tqdm
from typing import Dict, List, Optional, Sequence, Tuple, Union
from sklearn.preprocessing import RobustScaler

from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text
from pathlib import Path, PurePosixPath
from psycopg2.extras import execute_values

import torch
import torch.nn as nn
import pickle
from datetime import datetime, date
from zoneinfo import ZoneInfo

from typing import Tuple, Optional

import json

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DB_URL = os.getenv("DOCKER_DB_URL")

CYBOS_TICKER_LIST = json.loads(os.getenv("CYBOS_TICKER_LIST"))
# 경로 설정
MODEL_DIR = os.getenv('MODEL_DIR')
SCALERS_PATH = os.getenv('SCALERS_PATH')

print(os.environ.get("MODEL_DIR"))
print(os.environ.get("SCALERS_PATH"))

engine = create_engine(
    DB_URL,
    pool_pre_ping=True,
    future=True
)


# VGG-LSTM 모델 클래스 정의
class VGGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(VGGBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(2, stride=2)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        return x


class VGG_LSTM_Model(nn.Module):
    """1D VGG + LSTM 모델"""

    def __init__(self, input_dim=15, sequence_length=250, hidden_size=128,
                 num_layers=2, dropout=0.3, bidirectional=True):
        super(VGG_LSTM_Model, self).__init__()

        # VGG blocks
        self.vgg1_blocks = nn.ModuleList()
        vgg1_channels = [32, 64, 128]
        in_channels = input_dim
        for out_channels in vgg1_channels:
            self.vgg1_blocks.append(VGGBlock(in_channels, out_channels))
            in_channels = out_channels

        self.vgg2_blocks = nn.ModuleList()
        vgg2_channels = [256, 256, 512]
        for out_channels in vgg2_channels:
            self.vgg2_blocks.append(VGGBlock(in_channels, out_channels))
            in_channels = out_channels

        # LSTM
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.bn_lstm = nn.BatchNorm1d(lstm_output_size)

        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_size, 64)
        self.bn_fc = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)

        for block in self.vgg1_blocks:
            x = block(x)

        for block in self.vgg2_blocks:
            x = block(x)

        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.bn_lstm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x.squeeze()


######### target_date 이하 chart 불러오기
def fetch_chart_to_df_by_ticker_and_date(ticker: str, target_date: date):

    sql = text("""
               SELECT c.chart_date AS date, c.chart_open AS open, c.chart_high AS high, c.chart_low AS low, c.chart_close AS close, c.chart_volume AS volume
               FROM public.charts AS c
                   LEFT JOIN public.stocks AS s
               ON s.id = c.stock_id
               WHERE c.chart_date <= :target_date
                 AND s.ticker = :ticker
               ORDER BY c.chart_date ASC
               """)

    with engine.begin() as conn:
        df = pd.read_sql_query(
            sql,
            conn,
            params={"target_date": target_date, "ticker": ticker},
            parse_dates=["date"]
        )
    return df


def fetch_latest_date(engine):
    sql = text("""
               SELECT latest_date
               FROM public.latest_date
               WHERE latest_date_name = 'charts';
               """)
    with engine.connect() as conn:
        result = conn.execute(sql).scalar()
    if result is not None:
        return result if isinstance(result, date) else result.date()
    return None


# == == == == == == == == == == == == = Data Processing == == == == == == == == == == == == =


def process_single_stock_file(ticker: str, window_size: int = 250):
    """
단일 주식 CSV 파일을 처리하여 최근 150개 샘플 생성
Returns: X, y_true (실제 변화율), valid_indices, stock_name
"""

    try:
        # 예측 날짜 설정
        # KST = ZoneInfo("Asia/Seoul")
        # target_date = datetime.now(KST).date()  # 오늘 날짜
        # target_date = date(2025, 1, 20)  # 특정 날짜 지정
        target_date = fetch_latest_date(engine)

        # TODO: 403
        df = fetch_chart_to_df_by_ticker_and_date(ticker, target_date)

        df = df.replace([np.inf, -np.inf], np.nan)  # inf를 NaN으로 통일
        df = df.fillna(0)  # NaN → 0

        stock_name = ticker;

        # 원본 종가 저장
        original_close = df['close'].values

        # 특성 생성
        # 1. 차분값
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[f'{col}_diff'] = df[col].diff()

        # 2. 변화율
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[f'{col}_pct'] = df[col].pct_change() * 100

        # 3. 이격도
        periods = [5, 10, 20, 40, 60]
        for period in periods:
            ma = df['close'].rolling(window=period).mean()
            df[f'disp_{period}'] = (df['close'] / ma - 1) * 100

        # 원래 OHLCV 제거
        df = df.drop(columns=['open', 'high', 'low', 'close', 'volume'])

        # 데이터가 충분한지 확인 (최소 250 + 3 + 150개 필요)
        if len(df) < 250:
            print(len(df) - 249, stock_name, "데이터 개수가 부족합니다.")
            return np.zeros(len(df)), np.zeros(len(df))

        # 특성 선택
        feature_cols = [col for col in df.columns if col != 'date']
        features = df[feature_cols].values
        dates = df[['date']].values

        # 최근 150개 샘플만 생성
        X = []
        X_dates = []

        start_idx = 250  # 150개 + 여유 2개

        for i in range(0, len(features) - window_size + 1):
            X.append(features[i:i + window_size])
            X_dates.append(dates[i + window_size - 1])

        print("X: ", len(X))  # 처리완료해서 -249된 개수
        print("X_dates: ", len(X_dates))

        return np.array(X), np.array(X_dates)

    except Exception as e:
        "알 수 없는 오류가 생겼습니다. 은화에게 문의하세요."
        return None, None, None, stock_name


def apply_scalers(X: np.ndarray, scalers: list) -> np.ndarray:
    """스케일러 적용"""
    X_scaled = np.zeros_like(X)

    for feature_idx, scaler in enumerate(scalers):
        feature_data = X[:, :, feature_idx].reshape(-1, 1)
        median_val = scaler.center_[0] if hasattr(scaler, 'center_') else 0
        feature_data = np.nan_to_num(feature_data, nan=median_val,
                                     posinf=median_val, neginf=median_val)
        feature_scaled = scaler.transform(feature_data)
        X_scaled[:, :, feature_idx] = feature_scaled.reshape(X.shape[0], X.shape[1])

    return X_scaled


def fetch_chart_ids_by_ticker_dates(engine, ticker: str, dates):

    if len(dates) == 0:
        return {}
    dt = pd.to_datetime(dates)
    date_only = [d.date() for d in dt]

    sql = text("""
               SELECT c.id AS chart_id, c.chart_date::date AS chart_date
               FROM public.charts c
                        JOIN public.stocks s ON s.id = c.stock_id
               WHERE s.ticker = :ticker
                 AND c.chart_date::date = ANY(:dates)
               """)
    with engine.begin() as conn:
        rows = conn.execute(sql, {"ticker": ticker, "dates": date_only}).fetchall()
    return {row.chart_date: row.chart_id for row in rows}


def classify_direction(preds, up_th=0.55, down_th=0.45):

    p = np.asarray(preds).reshape(-1)
    out = np.full(p.shape, 'n', dtype=object)
    out[p >= up_th] = 'u'
    out[p <= down_th] = 'd'
    return out


# ========================= Model Evaluation =========================

def validate_on_real_data(cybos_ticker_list: list, model_dir: str, scaler_path: str,
                          device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    실제 주식 데이터로 모델 검증
    """
    print("=" * 60)
    print("Real-World Stock Prediction Model Validation")
    print("=" * 60)

    # 1. 스케일러 로드
    print("\n[1/4] Loading scalers...")
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    print(f"Loaded {len(scalers)} scalers")

    # 2. 모델 로드 (앙상블)
    print("\n[2/4] Loading ensemble models...")
    models = []
    for i in range(5):  # 5개 모델 앙상블
        model = VGG_LSTM_Model()
        model_path = os.path.join(model_dir, f'model_{i}.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            models.append(model)
            print(f"  Loaded model {i + 1}")

    if not models:
        print("No models found!")
        return pd.DataFrame(columns=["chart_id", "record_direction", "record_prediction"])

    all_results = []
    records_by_ticker_df = []

    for cybos_ticker in cybos_ticker_list:  # 처음 100개 종목만 테스트 XX 전체 테스트

        ticker = cybos_ticker.lstrip('A')

        # 데이터 처리
        X, X_dates = process_single_stock_file(ticker)
        # array로 리턴

        # 예외 경우 추가
        if X is None:
            print(f"[SKIP] {ticker}: X is None (insufficient data).")
            continue

        if isinstance(X, np.ndarray) and X.size == 0:
            print(f"[SKIP] {ticker}: X is empty.")
            continue

        if np.allclose(X, 0, atol=1e-8):
            all_results.append(X)
            continue

        # 스케일링
        X_scaled = apply_scalers(X, scalers)

        # 앙상블 예측
        X_tensor = torch.FloatTensor(X_scaled).to(device)

        predictions = []

        with torch.no_grad():
            for model in models:
                pred = model(X_tensor).cpu().numpy()
                predictions.append(pred)

        predictions_mean = np.mean(predictions, axis=0).reshape(-1)
        dates = np.ravel(X_dates)
        dates = pd.to_datetime(dates, format='%Y%m%d')

        chart_map = fetch_chart_ids_by_ticker_dates(engine, ticker, dates)
        chart_ids = [chart_map.get(d.date(), None) for d in dates]

        record_direction = classify_direction(predictions_mean)

        df_res = pd.DataFrame({
            "chart_id": chart_ids,
            "record_prediction": predictions_mean,
            "record_direction": record_direction
        })

        records_by_ticker_df.append(df_res)

    if not records_by_ticker_df:
        print("No results.")
        return pd.DataFrame(columns=["chart_id", "record_direction", "record_prediction"])
    else:
        records_df = pd.concat(records_by_ticker_df, ignore_index=True)

    return records_df


def save_records(engine, input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()

    records = df[[
        'chart_id', 'record_direction', 'record_prediction'
    ]].itertuples(index=False, name=None)

    sql = text("""
               INSERT INTO public.records (chart_id,
                                           record_direction,
                                           record_prediction)
               VALUES %s ON CONFLICT
               ON CONSTRAINT chart_id_unique
                   DO NOTHING
                   RETURNING (xmax = 0) AS inserted;
               """)

    with engine.begin() as conn:
        cur = conn.connection.cursor()
        execute_values(cur, sql.text, records, page_size=1000)

        # RETURNING 결과가 있을 때만 fetch
        if cur.description is not None:
            results = cur.fetchall()
            inserted = sum(r[0] for r in results)
            updated = len(results) - inserted
        else:
            inserted = 0
            updated = 0

        print(f">>> 삽입: {inserted}건, >>> 갱신: {updated}건")


if __name__ == "__main__":
    # 검증 실행
    prediction_df = validate_on_real_data(
        CYBOS_TICKER_LIST,
        model_dir=MODEL_DIR,
        scaler_path=SCALERS_PATH
    )

    save_records(engine, prediction_df)
