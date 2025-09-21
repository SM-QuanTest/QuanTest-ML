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

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DB_URL = os.getenv("DOCKER_DB_URL")

engine = create_engine(
    DB_URL,
    pool_pre_ping=True,
    future=True
)

class Config:
    """모델 설정"""
    # 모델 설정
    INPUT_DIM = 15
    SEQUENCE_LENGTH = 250
    VGG1_CHANNELS = [32, 64, 128]
    VGG2_CHANNELS = [256, 256, 512]
    LSTM_HIDDEN_SIZE = 128
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.3
    LSTM_BIDIRECTIONAL = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    def __init__(self, config: Config):
        super(VGG_LSTM_Model, self).__init__()

        self.vgg1_blocks = nn.ModuleList([
            VGGBlock(config.INPUT_DIM if i == 0 else config.VGG1_CHANNELS[i - 1],
                     config.VGG1_CHANNELS[i])
            for i in range(len(config.VGG1_CHANNELS))
        ])

        self.vgg2_blocks = nn.ModuleList([
            VGGBlock(config.VGG1_CHANNELS[-1] if i == 0 else config.VGG2_CHANNELS[i - 1],
                     config.VGG2_CHANNELS[i])
            for i in range(len(config.VGG2_CHANNELS))
        ])

        lstm_input_size = config.VGG2_CHANNELS[-1]
        lstm_output_size = config.LSTM_HIDDEN_SIZE * (2 if config.LSTM_BIDIRECTIONAL else 1)

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_NUM_LAYERS,
            dropout=config.LSTM_DROPOUT,
            batch_first=True,
            bidirectional=config.LSTM_BIDIRECTIONAL
        )

        self.bn_lstm = nn.BatchNorm1d(lstm_output_size)

        self.dropout = nn.Dropout(config.LSTM_DROPOUT)
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


class EnsembleModel:
    """앙상블 모델"""

    def __init__(self, models: List[nn.Module], config: Config):
        self.models = models
        self.config = config

    def predict_proba(self, X: torch.Tensor) -> np.ndarray:
        predictions = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                X_device = X.to(self.config.DEVICE)
                outputs = model(X_device)
                predictions.append(outputs.cpu().numpy())

        ensemble_proba = np.mean(predictions, axis=0)
        return ensemble_proba

    def predict(self, X: torch.Tensor) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)


def load_charts_joined(engine, window_size: int) -> pd.DataFrame:
    # ws개 이상 있는 데이터만 선별해서 주가데이터와 주식 정보 가져옴

    sql = text("""
               WITH eligible AS (SELECT stock_id
                                 FROM public.charts
                                 GROUP BY stock_id
                                 HAVING COUNT(*) >= :ws)
               SELECT c.stock_id,
                      s.ticker,
                      c.chart_date,
                      c.chart_open,
                      c.chart_high,
                      c.chart_low,
                      c.chart_close,
                      c.chart_volume
               FROM public.charts c
                        JOIN eligible e ON e.stock_id = c.stock_id
                        LEFT JOIN public.stocks s ON s.id = c.stock_id
               ORDER BY c.stock_id, c.chart_date ASC
               """)
    with engine.begin() as conn:
        df = pd.read_sql_query(sql, conn, params={"ws": window_size})
    return df


def process_stock_data_for_prediction(df: pd.DataFrame, scalers: List[RobustScaler],
                                      config: Config) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame], str]:
    """
    주식 데이터를 처리하여 예측용 데이터 생성
    Returns: (X_data, price_df, stock_name)
    """
    try:
        stock_name = df['ticker'].iloc[0] if 'ticker' in df.columns else f"STOCK_{df['stock_id'].iloc[0]}"

        # 날짜순 정렬
        df = df.sort_values('chart_date')

        # print("len(df):", len(df))

        # 필수 컬럼 확인 및 이름 변경
        df = df.rename(columns={
            'chart_date': 'date',
            'chart_open': 'open',
            'chart_high': 'high',
            'chart_low': 'low',
            'chart_close': 'close',
            'chart_volume': 'volume'
        })

        # 날짜를 인덱스로 설정
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # total_days_needed = config.SEQUENCE_LENGTH + 60  # 250 + 105 = 355
        # if len(df) > total_days_needed:
        #    df = df.tail(total_days_needed)
        # 처음 다운받아 전부 저장한 후에 매일 새로 들어오는 것 하나씩만 처리할 때는 켜기

        # 원본 가격 데이터 저장
        price_df = df[['open', 'high', 'low', 'close', 'volume']].copy()

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

        # 원래 OHLCV 컬럼 삭제
        df = df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'ticker', 'stock_id'])

        # NaN 값 제거
        df = df.dropna()
        price_df = price_df.loc[df.index]

        print("len(df):", len(df))

        # 데이터가 부족한 경우
        if len(df) < config.SEQUENCE_LENGTH + 60:
            # print("len(df):", len(df))
            # print("config.SEQUENCE_LENGTH:", config.SEQUENCE_LENGTH)
            # print("stock_name", stock_name)
            return None, price_df, stock_name

        # 특성 선택
        features = df.values

        # 스케일링 적용
        features_scaled = apply_scalers_to_features(features, scalers)

        # 윈도우 슬라이딩으로 X 생성 (최근 105일)
        X = []
        for i in range(len(features_scaled) - config.SEQUENCE_LENGTH - 105 + 1,
                       len(features_scaled) - config.SEQUENCE_LENGTH + 1):
            if i >= 0:
                X.append(features_scaled[i:i + config.SEQUENCE_LENGTH])

        if len(X) == 0:
            print("523")
            return None, price_df.tail(105), stock_name

        # X = np.array(X)
        X = np.stack(X, axis=0).astype(np.float32)
        price_df = price_df.tail(105)

        print("X: ", X.shape)
        print("price_df: ", price_df.shape)
        print("stock_name: ", stock_name)

        return X, price_df, stock_name

    except Exception as e:
        print(f"Error processing stock data: {str(e)}")
        return None, None, df['ticker'].iloc[0] if 'ticker' in df.columns else "UNKNOWN"


def apply_scalers_to_features(features: np.ndarray, scalers: List[RobustScaler]) -> np.ndarray:
    """특성에 스케일러 적용"""
    features_scaled = np.zeros_like(features)

    for i in range(features.shape[1]):
        if i < len(scalers):
            feature_data = features[:, i].reshape(-1, 1)
            # NaN과 Inf 처리
            median_val = scalers[i].center_[0] if hasattr(scalers[i], 'center_') else 0
            feature_data = np.nan_to_num(feature_data, nan=median_val,
                                         posinf=median_val, neginf=median_val)
            features_scaled[:, i] = scalers[i].transform(feature_data).reshape(-1)
        else:
            features_scaled[:, i] = features[:, i]

    return features_scaled

# ============ 순수 예측 함수 ============
def predict_new_data(X_new, current_price, model, scalers_X, scaler_Y):
    """
    Y값이 없는 새로운 X 데이터에 대한 예측

    Args:
        X_new: 새로운 입력 데이터 (n_samples, 5, 8, 250)
        current_price: 현재 가격 (예측의 기준점)
        model: 학습된 모델
        scalers_X: X 데이터용 스케일러들
        scaler_Y: Y 데이터용 스케일러

    Returns:
        predictions: 예측 가격
        returns: 예측 수익률
    """
    # 입력 데이터 검증
    if X_new.ndim == 3:  # 단일 샘플인 경우
        X_new = X_new[np.newaxis, ...]  # (1, 5, 8, 250)로 변환

    n_samples = X_new.shape[0]
    print(f"Predicting {n_samples} samples...")

    # 데이터 정규화
    X_scaled = np.zeros_like(X_new)
    for i in range(5):  # 각 시계열
        for j in range(8):  # 각 IMF
            data = X_new[:, i, j, :].reshape(-1, 250)
            X_scaled[:, i, j, :] = scalers_X[i][j].transform(data)

    # 모델 예측
    model.eval()
    predictions_list = []

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)

        # 배치 처리 (메모리 효율성을 위해)
        batch_size = 32
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i + batch_size]
            with torch.cuda.amp.autocast():
                batch_pred = model(batch).cpu().numpy()
            predictions_list.append(batch_pred)

    # 예측 결과 합치기
    Y_pred_scaled = np.concatenate(predictions_list) if len(predictions_list) > 1 else predictions_list[0]

    # 역변환 (정규화된 변화율 -> 실제 변화율)
    Y_pred_returns = scaler_Y.inverse_transform(Y_pred_scaled.reshape(-1, 1)).ravel()

    # 변화율을 실제 가격으로 변환
    # current_price가 스칼라인 경우 모든 예측에 동일하게 적용
    if np.isscalar(current_price):
        predicted_prices = current_price * (1 + Y_pred_returns)
    else:
        # current_price가 배열인 경우 각각 적용
        predicted_prices = current_price * (1 + Y_pred_returns)

    return predicted_prices, Y_pred_returns

# ============ 단일 샘플 예측 함수 ============
def predict_single(X_single, current_price, model, scalers_X, scaler_Y):
    """
    단일 샘플에 대한 간단한 예측

    Args:
        X_single: 단일 입력 데이터 (5, 8, 250)
        current_price: 현재 가격
        model: 학습된 모델
        scalers_X: X 데이터용 스케일러들
        scaler_Y: Y 데이터용 스케일러

    Returns:
        predicted_price: 예측 가격
        predicted_return: 예측 수익률
    """
    # 차원 추가
    X_single = X_single[np.newaxis, ...]  # (1, 5, 8, 250)

    # 예측
    prices, returns = predict_new_data(X_single, current_price, model, scalers_X, scaler_Y)

    return prices[0], returns[0]


def parse_file_key(p: Path):
    stem = p.stem
    parts = stem.split("_")

    if len(parts) < 2 or not parts[1].isdigit():
        raise ValueError(f"파일명에서 날짜 파싱 실패: {p.name}")
    ticker = parts[0]
    chart_date = pd.to_datetime(parts[1], format="%Y%m%d").date()
    return ticker, chart_date

def collect_keys(output_dir: str, target_date: date | None = None) -> pd.DataFrame:
    # bin에서 npz 확장자 가진 파일들
    files = sorted(Path(output_dir).glob("*.npz"))
    rows = []
    for f in files:
        try:
            t, d = parse_file_key(f)
            if target_date is None or d == target_date:
                rows.append({"path": str(f), "ticker": t, "chart_date": d})
        except Exception as e:
            print(f"[SKIP] {f.name} - {e}")
    ################################
    print(pd.DataFrame(rows))
    return pd.DataFrame(rows)


def resolve_chart_ids(engine, keys_df: pd.DataFrame) -> pd.DataFrame:
    if keys_df.empty:
        return keys_df.assign(chart_id=pd.NA, current_price=pd.NA)

    values_sql = []
    params = {}
    for i, row in keys_df.reset_index(drop=True).iterrows():
        values_sql.append(f"(:t{i}, :d{i})")
        params[f"t{i}"] = row["ticker"]
        params[f"d{i}"] = row["chart_date"]

    # sql = text(f"""
    #     WITH keys(ticker, chart_date) AS (
    #         VALUES {", ".join(values_sql)}
    #     )
    #     SELECT k.ticker, k.chart_date, c.id AS chart_id, c.chart_close AS current_price
    #     FROM keys k
    #     JOIN public.stocks s ON s.ticker = k.ticker
    #     JOIN public.charts c ON c.stock_id = s.id AND c.chart_date = k.chart_date
    # """)

    sql = text(f"""
            WITH keys(ticker, chart_date) AS (
                VALUES {", ".join(values_sql)}
            )
            SELECT k.ticker, k.chart_date, c.id AS chart_id, c.chart_close, c.chart_open, c.chart_high, c.chart_low, c.chart_volume
            FROM keys k
            JOIN public.stocks s ON s.ticker = k.ticker
            JOIN public.charts c ON c.stock_id = s.id AND c.chart_date = k.chart_date
        """)
    with engine.begin() as conn:
        df_map = pd.read_sql_query(sql, conn, params=params)

    return keys_df.merge(df_map, on=["ticker", "chart_date"], how="left")


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

MODEL_DIR = os.getenv('MODEL_DIR')
SCALERS_PATH = os.getenv('SCALERS_PATH')

print(os.environ.get("MODEL_DIR"))
print(os.environ.get("SCALERS_PATH"))

# 모델 로드 함수 수정
# def load_models_and_scalers(model_dir: str = MODEL_DIR, scaler_path: str = SCALERS_PATH):
# def load_models_and_scalers(model_dir: str = 'models', scaler_path: str = 'scalers.pkl'):
def load_models_and_scalers(model_dir: str = MODEL_DIR, scaler_path: str = SCALERS_PATH):
# def load_models_and_scalers(model_dir: str = MODEL_DIR):
    """
    저장된 앙상블 모델과 스케일러를 불러오는 함수
    """
    config = Config()

    # 스케일러 로드
    print("스케일러 로드 중...")
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    print(f"스케일러 로드 완료: {len(scalers)}개")

    # 앙상블 모델 로드
    print("앙상블 모델 로드 중...")
    models = []
    for i in range(5):  # 5개 앙상블 모델
        model = VGG_LSTM_Model(config)
        model_path = os.path.join(model_dir, f'model_{i}.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
            model.to(config.DEVICE)
            model.eval()
            models.append(model)
        else:
            print(f"Warning: {model_path} not found")

    ensemble = EnsembleModel(models, config)
    print(f"앙상블 모델 로드 완료: {len(models)}개")

    return ensemble, scalers, config


def predict_stocks_from_db(engine, target_date: date = None):
    """
    데이터베이스에서 주식 데이터를 가져와 예측 수행
    """
    config = Config()

    # 모델과 스케일러 로드
    ensemble, scalers, config = load_models_and_scalers()

    # 355일 이상의 데이터를 가진 종목 조회
    window_size = config.SEQUENCE_LENGTH + 105  # 250 + 105
    df_all = load_charts_joined(engine, window_size)
    print("df_all: \n", df_all) # stock_id 다 돌고있음, 전체 데이터 불러오는....

    if df_all.empty:
        print(f"충분한 데이터({window_size}일 이상)를 가진 종목이 없습니다.")
        return

    # target_date가 지정된 경우 해당 날짜 데이터만 필터링
    if target_date:
        df_all['chart_date'] = pd.to_datetime(df_all['chart_date']).dt.date
        # 각 종목별로 target_date를 포함한 이전 355일 데이터 필터링
        filtered_dfs = []

        print("df_all['stock_id'].unique(): ", df_all['stock_id'].unique())

        for stock_id in df_all['stock_id'].unique():
            # stock_df = df_all[df_all['stock_id'] == stock_id]
            stock_df = df_all[df_all['stock_id'] == stock_id].copy()
            stock_df = stock_df.sort_values('chart_date').reset_index(drop=True)

            # target_date를 포함한 데이터가 있는지 확인
            if target_date in stock_df['chart_date'].values:
                # target_date 이전 355일 데이터 선택
                end_idx = stock_df[stock_df['chart_date'] == target_date].index[0]
                start_idx = max(0, end_idx - window_size + 1)
                filtered_dfs.append(stock_df.iloc[start_idx:end_idx + 1])

        if filtered_dfs:
            df_all = pd.concat(filtered_dfs, ignore_index=True)
            print("df_all_기준데이터로: \n", df_all)


        else:
            print(f"{target_date}에 해당하는 데이터가 없습니다.")
            return

    # 종목별 예측 수행
    all_predictions = []

    for stock_id, stock_df in tqdm(df_all.groupby('stock_id'), desc="종목별 예측"):
        # 데이터 처리 및 예측
        X, price_df, stock_name = process_stock_data_for_prediction(stock_df, scalers, config)

        # print("X: \n", X)
        # print("price_df: \n", price_df)

        if X is not None and price_df is not None:
            # 예측 수행
            X_tensor = torch.FloatTensor(X).to(config.DEVICE)
            predictions = ensemble.predict(X_tensor)

            print("1111111111111111")

            # 예측 결과와 chart_id 매핑
            # price_df의 마지막 105개 행과 predictions 매핑
            if len(predictions) <= len(price_df):
                # predictions 개수만큼 price_df의 마지막 행들 사용
                pred_df = price_df.tail(len(predictions)).copy()
                pred_df['prediction'] = predictions
                pred_df['stock_id'] = stock_id
                pred_df['ticker'] = stock_name

                # chart_id 가져오기
                dates = pred_df.index.strftime('%Y-%m-%d').tolist()
                chart_ids = []

                for pred_date in dates:
                    sql = text("""
                               SELECT id
                               FROM public.charts
                               WHERE stock_id = :stock_id
                                 AND chart_date = :chart_date
                               """)
                    with engine.begin() as conn:
                        result = conn.execute(sql, {"stock_id": stock_id, "chart_date": pred_date}).fetchone()
                        if result:
                            chart_ids.append(result[0])
                        else:
                            chart_ids.append(None)

                pred_df['chart_id'] = chart_ids
                pred_df['record_direction'] = [get_direction(p) for p in predictions]
                pred_df['record_prediction'] = predictions
                # pred_df['record_prediction'] = predictions * 100  # 백분율로 변환

                # chart_id가 있는 행만 선택
                pred_df = pred_df[pred_df['chart_id'].notna()]
                print("pred_df : ", pred_df)

                if not pred_df.empty:
                    all_predictions.append(pred_df[['chart_id', 'record_direction', 'record_prediction']])
                    print("sdlkfj;slkfja;slfja;sklfja;slkfjajsuiohweu")

    # 모든 예측 결과 합치기
    if all_predictions:
        final_predictions = pd.concat(all_predictions, ignore_index=True)

        print(f"\n총 {len(final_predictions)}개 예측 생성")
        print("\n예측 결과 샘플:")
        print(final_predictions.head(10))

        # 데이터베이스에 저장
        save_records(engine, final_predictions)

        return final_predictions
    else:
        print("예측 결과가 없습니다.")
        return None


def get_direction(p):
    if p >= 0.55:  # 상승
        return 'u'
    elif p <= 0.45:  # 하락
        return 'd'
    else:
        return 'n'  # 보합


# ============ 메인 사용 예제 ============
def main():
    # ====변경한부분 시작====

    # GPU 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 예측 날짜 설정
    KST = ZoneInfo("Asia/Seoul")
    # target_date = datetime.now(KST).date()  # 오늘 날짜
    # target_date = date(2025, 1, 20)  # 특정 날짜 지정
    target_date = None

    # 데이터베이스에서 데이터 가져와 예측 수행
    predictions = predict_stocks_from_db(engine, target_date)
    '''
    if predictions is not None:
        # 예측 결과 요약
        print("\n=" * 50)
        print("예측 결과 요약")
        print("=" * 50)
        print(f"총 예측 건수: {len(predictions)}")

        direction_counts = predictions['record_direction'].value_counts()
        print(f"\n방향별 예측:")
        print(f"  상승(u): {direction_counts.get('u', 0)}건")
        print(f"  하락(d): {direction_counts.get('d', 0)}건")
        print(f"  중립(n): {direction_counts.get('n', 0)}건")

        print(f"\n예측 확률 통계:")
        print(f"  평균: {predictions['record_prediction'].mean():.2f}%")
        print(f"  표준편차: {predictions['record_prediction'].std():.2f}%")
        print(f"  최대: {predictions['record_prediction'].max():.2f}%")
        print(f"  최소: {predictions['record_prediction'].min():.2f}%")'''
    # ====변경한부분 끝====

    # # 1. 모델과 스케일러 로드
    # print("Loading model and scalers...")
    # model, scalers_X, scaler_Y = load_model_and_scalers(
    #     model_path=MODEL_PATH,
    #     scalers_path=SCALERS_PATH
    # )
    #
    # # 2. 새로운 데이터로 예측 (예제)
    # print("\n" + "="*50)
    # print("예측 시작")
    # print("="*50)
    #
    # """# 옵션 1: npz 파일에서 X 데이터 로드하여 예측
    # npz_path = '/content/drive/MyDrive/BK21_2/코스닥전종목/processed/A000440_processed.npz'
    # current_price = 1000.0  # 현재 주가 (예시)
    #
    # results = predict_from_npz(
    #     npz_path=npz_path,
    #     current_prices=current_price,
    #     model=model,
    #     scalers_X=scalers_X,
    #     scaler_Y=scaler_Y
    # )"""
    #
    # KST = ZoneInfo("Asia/Seoul")
    # # TODO: date 변경
    # # today_kst = datetime.now(KST).date()
    # today_kst = date(2025, 8, 8)
    #
    # # TODO: db에서 today_kst로 df 불러와야함
    #
    # keys_df = collect_keys(PROJECT_OUTPUT, target_date=today_kst)
    # if keys_df.empty:
    #     print("예측할 .npz 파일이 없습니다.")
    #     return
    #
    # keys_df = resolve_chart_ids(engine, keys_df)
    #
    # # 매핑 실패한 파일 로그
    # missing = keys_df[keys_df["chart_id"].isna()]
    # if not missing.empty:
    #     print("\n[WARN] 매핑 실패(차트 행 없음):")
    #     print(missing[["path", "ticker", "chart_date"]].head(20))
    #     print(f"... 총 {len(missing)}개 파일 스킵")
    #
    # ok_df = keys_df.dropna(subset=["chart_id"]).copy()
    #
    # all_rows = []
    # for _, row in ok_df.iterrows():
    #     npz_path = row["path"]
    #     cur_price = float(row["current_price"])
    #     res = predict_from_npz(
    #         npz_path=npz_path,
    #         current_prices=cur_price,
    #         model=model,
    #         scalers_X=scalers_X,
    #         scaler_Y=scaler_Y
    #     )
    #
    #     df = pd.DataFrame({
    #         "chart_id": row["chart_id"],
    #         "record_prediction": res['predicted_returns'] * 100,
    #         'record_direction': ['u' if r > 0.02 else 'd' if r < -0.02 else 'n'
    #                for r in res['predicted_returns']]
    #     })
    #     all_rows.append(df)
    #
    # if not all_rows:
    #     print("예측 결과가 없습니다.")
    #     return
    #
    #
    # df_all = pd.concat(all_rows, ignore_index=True)
    # print("\n예측 결과 요약:")
    # print(df_all[["chart_id", "record_prediction", "record_direction"]].head(10))
    #
    # # db 저장
    # save_records(engine, df_all)


if __name__ == "__main__":
    print(os.cpu_count())
    print(DB_URL)
    print("한글테스트")


    main()
