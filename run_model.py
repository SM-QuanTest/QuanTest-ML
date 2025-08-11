import logging
import numpy as np
import os
import pandas as pd
from PyEMD import CEEMDAN
from multiprocessing import Pool
from tqdm import tqdm
from typing import Dict, List, Optional, Sequence, Tuple, Union

from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text
from pathlib import Path, PurePosixPath
from psycopg2.extras import execute_values

import torch
import torch.nn as nn
import pickle
from datetime import datetime

from typing import Tuple, Optional

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DB_URL = os.getenv("DOCKER_DB_URL")

engine = create_engine(
    DB_URL,
    pool_pre_ping=True,
    future=True
)

class CEEMDAN11:
    logger = logging.getLogger(__name__)

    noise_kinds_all = ["normal", "uniform"]

    def __init__(self, trials: int = 100, epsilon: float = 0.005, iter_imf: int = 9, ext_EMD=None,
                 parallel: bool = False, **kwargs):
        # Ensemble constants
        self.trials = trials
        self.epsilon = epsilon
        self.noise_scale = float(kwargs.get("noise_scale", 1.0))
        self.range_thr = float(kwargs.get("range_thr", 0.01))
        self.total_power_thr = float(kwargs.get("total_power_thr", 0.05))
        self.iter_imf = iter_imf

        self.beta_progress = bool(kwargs.get("beta_progress", True))  # Scale noise by std
        self.random = np.random.RandomState(seed=kwargs.get("seed"))
        self.noise_kind = kwargs.get("noise_kind", "normal")

        self._max_imf = int(kwargs.get("max_imf", 100))
        self.parallel = parallel
        self.processes = kwargs.get("processes")  # Optional[int]
        if self.processes is not None and not self.parallel:
            self.logger.warning("Passed value for process has no effect when `parallel` is False.")

        self.all_noise_EMD = []

        if ext_EMD is None:
            from PyEMD import EMD  # fmt: skip
            self.EMD = EMD(**kwargs)
        else:
            self.EMD = ext_EMD

        self.C_IMF = None  # Optional[np.ndarray]
        self.residue = None  # Optional[np.ndarray]

    def __call__(
            self, S: np.ndarray, T: Optional[np.ndarray] = None, max_imf: int = -1, progress: bool = False
    ) -> np.ndarray:
        return self.ceemdan(S, T=T, max_imf=max_imf, progress=progress)

    def __getstate__(self) -> Dict:
        self_dict = self.__dict__.copy()
        if "pool" in self_dict:
            del self_dict["pool"]
        return self_dict

    def generate_noise(self, scale: float, size: Union[int, Sequence[int]]) -> np.ndarray:

        if self.noise_kind == "normal":
            noise = self.random.normal(loc=0, scale=scale, size=size)
        elif self.noise_kind == "uniform":
            noise = self.random.uniform(low=-scale / 2, high=scale / 2, size=size)
        else:
            raise ValueError(
                "Unsupported noise kind. Please assigned `noise_kind` to be one of these: {0}".format(
                    str(self.noise_kinds_all)
                )
            )

        return noise

    def noise_seed(self, seed: int) -> None:
        """Set seed for noise generation."""
        self.random.seed(seed)

    def ceemdan(
            self, S: np.ndarray, T: Optional[np.ndarray] = None, max_imf: int = -1, progress: bool = False
    ) -> np.ndarray:

        scale_s = np.std(S)
        S = S / scale_s

        # Define all noise
        self.all_noises = self.generate_noise(self.noise_scale, (self.trials, S.size))

        # Decompose all noise and remember 1st's std
        self.logger.debug("Decomposing all noises")
        self.all_noise_EMD = self._decompose_noise()

        # Create first IMF
        last_imf = self._eemd(S, T, max_imf=1, progress=progress)[0]
        res = np.empty(S.size)

        all_cimfs = last_imf.reshape((-1, last_imf.size))
        prev_res = S - last_imf

        self.logger.debug("Starting CEEMDAN")
        total = (max_imf - 1) if max_imf != -1 else None
        it = iter if not progress else lambda x: tqdm(x, desc="cIMF decomposition", total=total)
        for _ in it(range(self._max_imf)):
            # Check end condition in the beginning because we've already have 1 IMF
            if self.end_condition(S, all_cimfs, max_imf):
                self.logger.debug("End Condition - Pass")
                break

            imfNo = all_cimfs.shape[0]
            beta = self.epsilon * np.std(prev_res)

            local_mean = np.zeros(S.size)
            for trial in range(self.trials):
                # Skip if noise[trial] didn't have k'th mode
                noise_imf = self.all_noise_EMD[trial]
                res = prev_res.copy()
                if len(noise_imf) > imfNo:
                    res += beta * noise_imf[imfNo]

                # Extract local mean, which is at 2nd position
                imfs = self.emd(res, T, max_imf=1)
                local_mean += imfs[-1] / self.trials

            last_imf = prev_res - local_mean
            all_cimfs = np.vstack((all_cimfs, last_imf))
            prev_res = local_mean.copy()
        # END of while

        res = S - np.sum(all_cimfs, axis=0)
        all_cimfs = np.vstack((all_cimfs, res))
        all_cimfs = all_cimfs * scale_s

        # Empty all IMFs noise
        del self.all_noise_EMD[:]

        self.C_IMF = all_cimfs
        self.residue = S * scale_s - np.sum(self.C_IMF, axis=0)

        return all_cimfs

    def end_condition(self, S: np.ndarray, cIMFs: np.ndarray, max_imf: int) -> bool:
        """Test for end condition of CEEMDAN.

        Procedure stops if:

        * number of components reach provided `max_imf`, or
        * last component is close to being pure noise (range or power), or
        * set of provided components reconstructs sufficiently input.

        Parameters
        ----------
        S : numpy array
            Original signal on which CEEMDAN was performed.
        cIMFs : numpy 2D array
            Set of cIMFs where each row is cIMF.
        max_imf : int
            The maximum number of imfs to extract.

        Returns
        -------
        end : bool
            Whether to stop CEEMDAN.
        """
        imfNo = cIMFs.shape[0]

        # Check if hit maximum number of cIMFs
        # 차원 수 맞춰야 하므로 개수만 기준으로 바꿈
        if self.iter_imf <= imfNo:
            return True
        else:
            return False

    def _decompose_noise(self) -> List[np.ndarray]:
        if self.parallel:
            pool = Pool(processes=self.processes)
            all_noise_EMD = pool.map(self.emd, self.all_noises)
            pool.close()
        else:
            all_noise_EMD = [self.emd(noise, max_imf=-1) for noise in self.all_noises]

        # Normalize w/ respect to 1st IMF's std
        if self.beta_progress:
            all_stds = [np.std(imfs[0]) for imfs in all_noise_EMD]
            all_noise_EMD = [imfs / imfs_std for (imfs, imfs_std) in zip(all_noise_EMD, all_stds)]

        return all_noise_EMD

    def _eemd(self, S: np.ndarray, T: Optional[np.ndarray] = None, max_imf: int = -1, progress=True) -> np.ndarray:
        if T is None:
            T = np.arange(len(S), dtype=S.dtype)

        self._S = S
        self._T = T
        self._N = N = len(S)
        self.max_imf = max_imf

        # For trial number of iterations perform EMD on a signal
        # with added white noise
        if self.parallel:
            pool = Pool(processes=self.processes)
            map_pool = pool.imap_unordered
        else:  # Not parallel
            map_pool = map

        self.E_IMF = np.zeros((1, N))
        it = iter if not progress else lambda x: tqdm(x, desc="Decomposing noise", total=self.trials)

        for IMFs in it(map_pool(self._trial_update, range(self.trials))):
            if self.E_IMF.shape[0] < IMFs.shape[0]:
                num_new_layers = IMFs.shape[0] - self.E_IMF.shape[0]
                self.E_IMF = np.vstack((self.E_IMF, np.zeros(shape=(num_new_layers, N))))
            self.E_IMF[: IMFs.shape[0]] += IMFs

        if self.parallel:
            pool.close()

        return self.E_IMF / self.trials

    def _trial_update(self, trial: int) -> np.ndarray:
        """A single trial evaluation, i.e. EMD(signal + noise)."""
        # Generate noise
        noise = self.epsilon * self.all_noise_EMD[trial][0]
        return self.emd(self._S + noise, self._T, self.max_imf)

    def emd(self, S: np.ndarray, T: Optional[np.ndarray] = None, max_imf: int = -1) -> np.ndarray:
        """Vanilla EMD method.

        Provides emd evaluation from provided EMD class.
        For reference please see :class:`PyEMD.EMD`.
        """
        return self.EMD.emd(S, T, max_imf=max_imf)

    def get_imfs_and_residue(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Provides access to separated imfs and residue from recently analysed signal.
        :return: (imfs, residue)
        """
        if self.C_IMF is None or self.residue is None:
            raise ValueError("No IMF found. Please, run EMD method or its variant first.")
        return self.C_IMF, self.residue

# 아까 os.cpu 개수가 8이하면
# CEEMDAN11(trials=trials,iter_imf=num, parallel=True, processes=8)
# 여기서 processed=개수만 네 cpu 개수 이하로 설정해주면 돼 많을수록 빨라 (20)
def CEEMDREAL(S, num):
    # Logging options
    logging.basicConfig(level=logging.INFO)

    max_imf = -1

    # Signal options
    N = len(S)
    tMin, tMax = 0, N
    T = np.linspace(tMin, tMax, N)

    # Prepare and run EEMD
    trials = 100
    ceemdan = CEEMDAN11(trials=trials, iter_imf=num, parallel=True, processes=16)

    return ceemdan(S, T, max_imf)


# 윈도우 크기
window_size = 250
step_size = 1
imf_k = 7

PROJECT_OUTPUT = os.environ.get("PROJECT_OUTPUT", "/app/bin")
out_dir = Path(PROJECT_OUTPUT)
out_dir.mkdir(parents=True, exist_ok=True)

print("PROJECT_OUTPUT:", repr(PROJECT_OUTPUT))
print("exists:", out_dir.exists())
print("isdir:", out_dir.is_dir())


def load_charts_joined(engine, window_size: int) -> pd.DataFrame:

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


def to_npz_from_db(engine):
    df = load_charts_joined(engine, window_size)

    if df.empty:
        print("charts 테이블에 window_size 이상 행을 가진 종목이 없습니다.")
        return

    # 필요한 컬럼만 사용, 중복/결측 처리
    need_cols = [
        "stock_id", "chart_date", "ticker",
        "chart_open", "chart_high", "chart_low", "chart_close", "chart_volume"
    ]
    df = df[need_cols].copy()

    # 종목별 처리
    #  TODO: db에 data 다운 시 결측값 처리 어떻게?
    for i, (stock_id, sub) in enumerate(df.groupby("stock_id", sort=False), 1):
        ticker = (sub["ticker"].iloc[0]
                  if "ticker" in sub and pd.notna(sub["ticker"].iloc[0])
                  else f"ID{stock_id}")
        print(f"[{i}/{df['stock_id'].nunique()}] stock_id={stock_id} ({ticker})")

        # 날짜 오름차순 정렬은 되어 있음. 최신 250개만 사용
        if len(sub) < window_size:
            print(f"[SKIP] stock_id={stock_id} ({ticker}) length={len(sub)} < {window_size}")
            continue

        recent = sub.tail(window_size)
        # 파일명에 넣을 날짜(마지막 날짜 사용)
        last_date = pd.to_datetime(recent["chart_date"].iloc[-1]).date()
        date_str = last_date.strftime("%Y%m%d")

        data_mat = recent[["chart_open", "chart_high", "chart_low", "chart_close", "chart_volume"]].to_numpy()

        # 열(총 5개)에 대해 CEEMD
        procNPZ = []
        for col_idx in range(5):
            print(f"  Processing column {col_idx + 1}...")
            # 해당 열의 데이터 추출
            col_data = data_mat[:, col_idx]

            try:
                imfs = CEEMDREAL(col_data, imf_k)
                procNPZ.append(imfs)
            except Exception as e:
                print(f"    Error processing stock_id={stock_id}, col={col_idx}: {e}")
                continue

        # 저장
        out_path = out_dir / f"{ticker}_{date_str}.npz"
        np.savez_compressed(out_path, imfs=procNPZ)
        print("Saved:", out_path.resolve())

        np.savez(out_path, imfs=procNPZ)
        print(f"Saved: {out_path}")

    print("All stocks processed from DB!")


# ============ 모델 클래스 정의 (동일하게 유지) ============
class CNNLSTMModule(nn.Module):
    def __init__(self, input_channels=5, cnn_filters=512, lstm_hidden=200, seq_length=250):
        super(CNNLSTMModule, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv1d(input_channels, cnn_filters//4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_filters//4, cnn_filters//2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(cnn_filters//2, cnn_filters, kernel_size=3, padding=1)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(cnn_filters//4)
        self.batch_norm2 = nn.BatchNorm1d(cnn_filters//2)
        self.batch_norm3 = nn.BatchNorm1d(cnn_filters)

        # LSTM layer
        self.lstm = nn.LSTM(cnn_filters, lstm_hidden, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.pool(x)

        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)

        x = self.relu(self.batch_norm3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)

        lstm_out, (h_n, c_n) = self.lstm(x)
        output = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)

        return output

class CEEMDEnsembleModel(nn.Module):
    def __init__(self, n_imfs=8, input_channels=5, cnn_filters=512, lstm_hidden=200, seq_length=250):
        super(CEEMDEnsembleModel, self).__init__()

        self.imf_modules = nn.ModuleList([
            CNNLSTMModule(input_channels, cnn_filters, lstm_hidden, seq_length)
            for _ in range(n_imfs)
        ])

        fusion_input_size = lstm_hidden * 2 * n_imfs
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        imf_outputs = []

        for i in range(8):
            imf_data = x[:, :, i, :]
            imf_out = self.imf_modules[i](imf_data)
            imf_outputs.append(imf_out)

        combined = torch.cat(imf_outputs, dim=1)
        output = self.fusion_layers(combined)

        return output.squeeze()

MODEL_PATH = os.getenv('MODEL_PATH', '/app/ceemd_model/ceemd_model.pth')
SCALERS_PATH = os.getenv('SCALERS_PATH', '/app/ceemd_model/scalers.pkl')


# ============ 모델 로드 함수 ============
def load_model_and_scalers(model_path=MODEL_PATH, scalers_path=SCALERS_PATH):
    """
    저장된 모델과 스케일러를 불러오는 함수
    """
    # 모델 로드
    model_info = torch.load(model_path, map_location=device, weights_only=False)
    config = model_info['model_config']

    # 모델 초기화
    model = CEEMDEnsembleModel(
        n_imfs=config['n_imfs'],
        input_channels=config['input_channels'],
        cnn_filters=config['cnn_filters'],
        lstm_hidden=config['lstm_hidden'],
        seq_length=config['seq_length']
    ).to(device)

    # 가중치 로드
    model.load_state_dict(model_info['state_dict'])
    model.eval()
    print(f"Model loaded from {model_path}")

    # 메트릭 출력 (있는 경우)
    if 'metrics' in model_info:
        print("Training performance:")
        for metric, value in model_info['metrics'].items():
            if metric == 'MAPE':
                print(f"  {metric}: {value:.2f}%")
            else:
                print(f"  {metric}: {value:.4f}")

    # 스케일러 로드
    with open(scalers_path, 'rb') as f:
        scalers_info = pickle.load(f)

    scalers_X = scalers_info['scalers_X']
    scaler_Y = scalers_info['scaler_Y']
    print(f"Scalers loaded from {scalers_path}")

    return model, scalers_X, scaler_Y

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
            batch = X_tensor[i:i+batch_size]
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

# ============ npz 파일에서 X만 로드하여 예측 ============
def predict_from_npz(npz_path, current_prices, model, scalers_X, scaler_Y):
    """
    npz 파일에서 X 데이터만 로드하여 예측

    Args:
        npz_path: npz 파일 경로
        current_prices: 현재 가격 (스칼라 또는 배열)
        model: 학습된 모델
        scalers_X: X 데이터용 스케일러들
        scaler_Y: Y 데이터용 스케일러

    Returns:
        predictions_dict: 예측 결과 딕셔너리
    """
    # 데이터 로드
    data = np.load(npz_path)
    X = data['imfs']  # shape: (n_samples, 5, 8, 250)

    print(f"Loaded X data: {X.shape}")

    # 예측 수행
    predicted_prices, predicted_returns = predict_new_data(
        X, current_prices, model, scalers_X, scaler_Y
    )

    # 결과 정리
    results = {
        'predicted_prices': predicted_prices,
        'predicted_returns': predicted_returns,
        'current_prices': current_prices if not np.isscalar(current_prices) else [current_prices] * len(predicted_prices),
        'n_samples': len(predicted_prices)
    }

    # 결과 출력
    print(f"\n예측 완료: {results['n_samples']}개 샘플")
    print(f"예측 가격 범위: {predicted_prices.min():.2f} ~ {predicted_prices.max():.2f}")
    print(f"평균 예측 가격: {predicted_prices.mean():.2f}")
    print(f"예측 수익률 범위: {predicted_returns.min()*100:.2f}% ~ {predicted_returns.max()*100:.2f}%")
    print(f"평균 예측 수익률: {predicted_returns.mean()*100:.2f}%")

    return results

def parse_file_key(p: Path):

    stem = p.stem
    parts = stem.split("_")

    if len(parts) < 2 or not parts[1].isdigit():
        raise ValueError(f"파일명에서 날짜 파싱 실패: {p.name}")
    chart_date = pd.to_datetime(parts[1], format="%Y%m%d").date()
    return ticker, chart_date

def collect_keys(output_dir: str) -> pd.DataFrame:
    files = sorted(Path(output_dir).glob("*.npz"))
    rows = []
    for f in files:
        try:
            t, d = parse_file_key(f)
            rows.append({"path": str(f), "ticker": t, "chart_date": d})
        except Exception as e:
            print(f"[SKIP] {f.name} - {e}")
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

    sql = text(f"""
        WITH keys(ticker, chart_date) AS (
            VALUES {", ".join(values_sql)}
        )
        SELECT k.ticker, k.chart_date, c.id AS chart_id, c.chart_close AS current_price
        FROM keys k
        JOIN public.stocks s ON s.ticker = k.ticker
        JOIN public.charts c ON c.stock_id = s.id AND c.chart_date = k.chart_date
    """)
    with engine.begin() as conn:
        df_map = pd.read_sql_query(sql, conn, params=params)

    return keys_df.merge(df_map, on=["ticker","chart_date"], how="left")


def save_records(engine, input_df:pd.DataFrame) -> pd.DataFrame:

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
        execute_values(cur, sql.text, records)

        # RETURNING 결과가 있을 때만 fetch
        if cur.description is not None:
            results = cur.fetchall()
            inserted = sum(r[0] for r in results)
            updated = len(results) - inserted
        else:
            inserted = 0
            updated = 0

        print(f">>> 삽입: {inserted}건, >>> 갱신: {updated}건")

# ============ 메인 사용 예제 ============
def main():
    # 1. 모델과 스케일러 로드
    print("Loading model and scalers...")
    model, scalers_X, scaler_Y = load_model_and_scalers(
        model_path=MODEL_PATH,
        scalers_path=SCALERS_PATH
    )

    # 2. 새로운 데이터로 예측 (예제)
    print("\n" + "="*50)
    print("예측 시작")
    print("="*50)

    """# 옵션 1: npz 파일에서 X 데이터 로드하여 예측
    npz_path = '/content/drive/MyDrive/BK21_2/코스닥전종목/processed/A000440_processed.npz'
    current_price = 1000.0  # 현재 주가 (예시)

    results = predict_from_npz(
        npz_path=npz_path,
        current_prices=current_price,
        model=model,
        scalers_X=scalers_X,
        scaler_Y=scaler_Y
    )"""

    keys_df = collect_keys(PROJECT_OUTPUT)
    if keys_df.empty:
        print("예측할 .npz 파일이 없습니다.")
        return

    keys_df = resolve_chart_ids(engine, keys_df)

    # 매핑 실패한 파일 로그
    missing = keys_df[keys_df["chart_id"].isna()]
    if not missing.empty:
        print("\n[WARN] 매핑 실패(차트 행 없음):")
        print(missing[["path", "ticker", "chart_date"]].head(20))
        print(f"... 총 {len(missing)}개 파일 스킵")

    ok_df = keys_df.dropna(subset=["chart_id"]).copy()

    all_rows = []
    for _, row in ok_df.iterrows():
        npz_path = row["path"]
        cur_price = float(row["current_price"])
        res = predict_from_npz(
            npz_path=npz_path,
            current_prices=cur_price,
            model=model,
            scalers_X=scalers_X,
            scaler_Y=scaler_Y
        )

        df = pd.DataFrame({
            "chart_id": row["chart_id"],
            "record_prediction": res['predicted_returns'] * 100,
            'record_direction': ['u' if r > 0.02 else 'd' if r < -0.02 else 'n'
                   for r in res['predicted_returns']]
        })
        all_rows.append(df)

    if not all_rows:
        print("예측 결과가 없습니다.")
        return


    # # 3. 결과 저장 (선택사항)
    # output_path = '/content/drive/MyDrive/BK21_2/predictions.npz'
    # np.savez(
    #     output_path,
    #     predicted_prices=results['predicted_prices'],
    #     predicted_returns=results['predicted_returns'],
    #     current_prices=results['current_prices'],
    #     timestamp=datetime.now().isoformat()
    # )
    # print(f"\n예측 결과 저장 완료: {output_path}")

    # 4. 결과를 DataFrame으로 변환 (선택사항)
    # df_results = pd.DataFrame({
    #     'current_price': results['current_prices'],
    #     'predicted_price': results['predicted_prices'],
    #     'predicted_return(%)': results['predicted_returns'] * 100,
    #     'signal': ['BUY' if r > 0.02 else 'SELL' if r < -0.02 else 'HOLD'
    #                for r in results['predicted_returns']]
    # })

    df_all = pd.concat(all_rows, ignore_index=True)
    print("\n예측 결과 요약:")
    print(df_all[["chart_id", "record_prediction", "record_direction"]].head(10))

    # db 저장
    save_records(engine, df_all)

    # # CSV로 저장 (선택사항)
    # csv_path = '/content/drive/MyDrive/BK21_2/predictions.csv'
    # df_results.to_csv(csv_path, index=False)
    # print(f"CSV 저장 완료: {csv_path}")

    # return results, df_results
    # return df_results


'''
# ============ 실시간 예측 예제 ============
def predict_realtime_example():
    """
    실시간으로 들어오는 데이터에 대한 예측 예제
    """
    # 모델 로드
    model, scalers_X, scaler_Y = load_model_and_scalers(
        model_path='/content/drive/MyDrive/BK21_2/ceemd_model.pth',
        scalers_path='/content/drive/MyDrive/BK21_2/scalers.pkl'
    )

    # 실시간 데이터 시뮬레이션 (실제로는 API나 스트림에서 받아옴)
    while True:
        try:
            # 새로운 데이터 생성 (실제로는 실시간 데이터)
            new_X = np.random.randn(1, 5, 8, 250)  # 예시
            current_price = 1050.0  # 현재가

            # 예측
            pred_price, pred_return = predict_single(
                new_X[0], current_price, model, scalers_X, scaler_Y
            )

            # 결과 출력
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"현재가: {current_price:.2f} → "
                  f"예측가: {pred_price:.2f} "
                  f"(수익률: {pred_return*100:+.2f}%)")

            # 매매 신호
            if pred_return > 0.03:
                print("  📈 강한 매수 신호!")
            elif pred_return > 0.01:
                print("  ↗️ 매수 신호")
            elif pred_return < -0.03:
                print("  📉 강한 매도 신호!")
            elif pred_return < -0.01:
                print("  ↘️ 매도 신호")
            else:
                print("  ➡️ 관망")

        except KeyboardInterrupt:
            print("\n예측 종료")
            break
'''



if __name__ == "__main__":
    print(os.cpu_count())  # 결과값이 8이상이 면 이후 코드에 조정 필요함!! -> 20
    print(DB_URL)

    # to_npz_from_db(engine)

    # GPU 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    main()
