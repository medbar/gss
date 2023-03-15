from dataclasses import dataclass

import cupy as cp

from gss.cacgmm import CACGMMTrainer

from gss.utils.logging_utils import get_logger

logger = get_logger()


@dataclass
class GSS:
    iterations: int
    iterations_post: int

    def __call__(self, Obs, acitivity_freq):
        initialization = cp.asarray(acitivity_freq, dtype=cp.float64)
        initialization = cp.where(initialization == 0, 1e-10, initialization)
        initialization = initialization / cp.sum(initialization, keepdims=True, axis=0)
        initialization = cp.repeat(initialization[None, ...], 513, axis=0)

        source_active_mask = cp.asarray(acitivity_freq, dtype=cp.bool)
        source_active_mask = cp.repeat(source_active_mask[None, ...], 513, axis=0)
        self._debug_call_init(initialization, source_active_mask)

        cacGMM = CACGMMTrainer()

        D, T, F = Obs.shape

        cur = cacGMM.fit(
            y=Obs.T,
            initialization=initialization[..., :T],
            iterations=self.iterations,
            source_activity_mask=source_active_mask[..., :T],
        )

        if self.iterations_post != 0:
            if self.iterations_post != 1:
                cur = cacGMM.fit(
                    y=Obs.T,
                    initialization=cur,
                    iterations=self.iterations_post - 1,
                )
            # is it right ? why without source_active_mask
            affiliation = cur.predict(Obs.T)
        else:
            affiliation = cur.predict(
                Obs.T, source_activity_mask=source_active_mask[..., :T]
            )
        # Freq, time, num_channels -> Time, Num_channels, freq
        posterior = affiliation.transpose(1, 2, 0)

        return posterior

    def _debug_call_init(self, initialization, source_active_mask):
        if logger.level > 10:
            return
        F, D, T = initialization.shape
        sparce_init = initialization[0, 0, :: T // 10]
        logger.debug(
            f"initialization for GMM something like {sparce_init}. "
            f"{initialization.shape = }"
        )
        sam_diff = source_active_mask[..., :-1] - source_active_mask[..., 1:]
        logger.debug(
            f"{source_active_mask.shape = }. {source_active_mask.sum() = }"
            f"Source activities changes :\n"
            f"- from one to zero {(sam_diff==1).sum()} times.\n"
            f"- from zero to one {(sam_diff==-1).sum()} times."
        )
