from dataclasses import dataclass

import cupy as cp

from gss.cacgmm import CACGMMTrainer

from gss.utils.logging_utils import get_logger

logger = get_logger()


@dataclass
class GSS:
    iterations: int
    iterations_post: int
    eps: int = 1e-10

    def __call__(self, Obs, acitivity_freq, initialization=None, normalize_init=True):
        # acitivity_freq.shape is [num_speakers, num_frames]
        # initialization.shape is [513, num_speakers, num_frames] or [num_speakers, num_frames]
        # assert (acitivity_freq <= 1).all(), f"bad acitivity_freq {acitivity_freq.max(axis=-1) = }"

        if initialization is None:
            initialization = cp.asarray(acitivity_freq, dtype=cp.float64)
        else:
            assert initialization.shape[-1] == acitivity_freq.shape[-1]
            assert initialization.shape[-2] == acitivity_freq.shape[-2]
            initialization = cp.asarray(initialization, dtype=cp.float64)
        initialization = cp.where(initialization == 0, self.eps, initialization)
        if normalize_init:
            initialization = initialization / cp.sum(
                initialization, keepdims=True, axis=-2
            )
        if len(initialization.shape) == 2:
            initialization = cp.repeat(initialization[None, ...], 513, axis=0)

        source_active_mask = cp.asarray(acitivity_freq, dtype=cp.bool)
        source_active_mask = cp.repeat(source_active_mask[None, ...], 513, axis=0)
        self._debug_call_init(initialization, source_active_mask)

        cacGMM = CACGMMTrainer()

        # D - number of channels
        # T - time
        # F - freq
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
        logger.debug(
            f"initialization ({initialization.shape=}) for GMM is \n"
            f"{initialization[0, ...].min(axis=-1) = }.\n"
            f"{initialization[0, ...].max(axis=-1) = }.\n"
            f"{initialization[0, ...].mean(axis=-1) = }. "
        )
        sam_int = cp.asarray(source_active_mask[0, ...], dtype=cp.int16)
        sam_diff = sam_int[..., :-1] - sam_int[..., 1:]
        logger.debug(
            f"{source_active_mask.shape = }. {source_active_mask[0, ...].sum(axis=-1) = }\n"
            f"Source activities changes :\n"
            f"- from one to zero {(sam_diff==1).sum(axis=-1)} times.\n"
            f"- from zero to one {(sam_diff==-1).sum(axis=-1)} times."
        )
