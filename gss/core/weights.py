from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

from lhotse import CutSet, SupervisionSet, SupervisionSegment
from gss.utils.logging_utils import get_logger
from gss.utils.chime6_utils import extract_info_from_kaldi_gss_track1_uid
from gss.utils.data_utils import _samples_to_stft_frames
from gss_helpers.stft_utils import linear_resample

logger = get_logger()


@dataclass  # (hash=True)
class Weights:
    speaker_to_idx_map: Dict[str, Dict[str, int]]
    supervisions_index: Any = None
    garbage_class: bool = True
    # cuts with custom vad_weights supervision
    # see utils.chime6_utils.build_sups_from_rspec
    cuts: "CutSet" = None
    eps: float = 1e-10
    stft_shift: int = 256
    stft_size: int = 1024
    sr: int = 16000
    stft_fading: bool = True
    stft_pad: bool = True

    def __post_init__(self):
        if self.supervisions_index is None:
            self.supervisions_index = self.cuts.index_supervisions()
        logger.info(f"Initialized Weights. " f"{self.speaker_to_idx_map = }")

    def _sec_to_stft_frames(self, sec, fading=None, pad=None):
        if fading is None:
            fading = self.stft_fading
        if pad is None:
            pad = self.stft_pad
        return _samples_to_stft_frames(
            sec * self.sr,
            size=self.stft_size,
            shift=self.stft_shift,
            fading=fading,
            pad=pad,
        )

    def get_weights(self, session_id, start_time, duration):
        # number of speakers + garbage
        idx = self.speaker_to_idx_map[session_id]
        cut_num_frames = self._sec_to_stft_frames(duration)
        weights = [np.zeros(cut_num_frames) for _ in range(len(idx))]
        if self.garbage_class:
            weights += [np.ones(cut_num_frames)]
        num_sups = 0
        total_frames = 0
        cut = self.cuts[session_id].truncate(
            offset=start_time,
            duration=duration,
            _supervisions_index=self.supervisions_index,
        )

        for sup in cut.supervisions:
            num_sups += 1
            speaker_id = idx[sup.speaker]
            sup_dur_frames = self._sec_to_stft_frames(sup.duration, fading=False)
            weight_segment = linear_resample(sup.vad_weights, sup_dur_frames)
            logger.debug(
                f"Resample vad_weights from {len(sup.vad_weights)} to {len(weight_segment)} frames."
            )
            if self.stft_fading:
                start_frame = self.stft_size // self.stft_shift - 1
                end_frame = cut_num_frames - (self.stft_size // self.stft_shift - 1)
            else:
                start_frame = 0
                end_frame = cut_num_frames
            if sup.start < 0:
                offset = self._sec_to_stft_frames(-sup.start, fading=False)
                logger.debug(f"Detected {sup.start = }. Start {offset = } frames.")
                weight_segment = weight_segment[..., offset:]
            else:
                start_frame += self._sec_to_stft_frames(sup.start, fading=False)
            if sup.end > duration:
                # offset = self._sec_to_stft_frames(sup.end - duration, fading=False)
                offset = end_frame - start_frame
                logger.debug(
                    f"Detected {sup.end=} larger than {duration=}. "
                    f"End {offset = } frames."
                )
                weight_segment = weight_segment[..., :offset]
            else:
                end_frame = start_frame + weight_segment.shape[-1]
            assert (
                end_frame - start_frame == weight_segment.shape[-1]
            ), f"{end_frame=} {start_frame=} {weight_segment.shape[-1]=}"
            total_frames += weight_segment.shape[-1]
            weights[speaker_id][start_frame:end_frame] = weight_segment
        weights = np.stack(weights)
        assert (
            num_sups > 0
        ), f"Weights for ({session_id=}, {start_time=}, {duration=}) does not exist. Wrong weights manifest?"
        logger.debug(
            f"Weights for ({session_id=}, {start_time=}, {duration=}) contains in {num_sups} supervisions. "
            f"Total supervisions duration {total_frames} frames."
        )
        return weights, idx
