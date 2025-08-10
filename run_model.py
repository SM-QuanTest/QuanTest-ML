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
from sqlalchemy import create_engine
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DB_URL = os.getenv("DOCKER_DB_URL")

engine = create_engine(
    DB_URL,
    pool_pre_ping=True,
    future=True
)

# print(os.cpu_count())#결과값이 8이상이 면 이후 코드에 조정 필요함!!


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
        ceemdan = CEEMDAN11(trials=trials, iter_imf=num, parallel=True, processes=10)

        return ceemdan(S, T, max_imf)


if __name__ == "__main__":
    print(os.cpu_count())  # 결과값이 8이상이 면 이후 코드에 조정 필요함!! -> 20
    print(DB_URL)
