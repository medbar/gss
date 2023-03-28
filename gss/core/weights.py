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
    speaker_to_idx_map: Dict[str, int]
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

    def _sec_to_stft_frames(self, sec):
        return _samples_to_stft_frames(
            sec * self.sr,
            size=self.stft_size,
            shift=self.stft_shift,
            fading=self.stft_fading,
            pad=self.stft_pad,
        )

    def get_weights(self, session_id, start_time, duration):
        # number of speakers + garbage
        total_num_frames = self._sec_to_stft_frames(duration)
        weights = [
            np.zeros(total_num_frames) for _ in range(len(self.speaker_to_idx_map))
        ]
        if self.garbage_class:
            weights += [np.ones(total_num_frames)]
        num_sups = 0
        total_duration = 0
        cut = self.cuts[session_id].truncate(
            offset=start_time,
            duration=duration,
            _supervisions_index=self.supervisions_index,
        )
        for sup in cut.supervisions:
            num_sups += 1
            total_duration += sup.duration
            dur_frames = self._sec_to_stft_frames(sup.duration)
            speaker_id = self.speaker_to_idx_map[sup["speaker"]]
            weight_segment = linear_resample(sup.vad_weights, dur_frames)
            logger.debug(
                f"Resample vad_weights from {len(sup.vad_weights)} to {len(weight_segment)} frames."
            )
            start_frame = self._sec_to_stft_frames(sup.start)
            end_frame = start_frame + dur_frames + 1
            if sup.start < start_time:
                start_diff_frames = self._sec_to_stft_frames(start_time - sup.start)
                assert 0 < start_diff_frames < dur_frames
                logger.debug(
                    f"Trim weights supervision {start_diff_frames} frames from start"
                )
                weight_segment = weight_segment[..., start_diff_frames:]
                start_frame = 0
            if sup.end > start_time + duration:
                end_diff_frames = self._sec_to_stft_frames(
                    sup.end - start_time - duration
                )
                assert 0 < end_diff_frames < dur_frames
                logger.debug(
                    f"Trim weights supervision {end_diff_frames} frames from end"
                )
                weight_segment = weight_segment[..., :-end_diff_frames]
                end_frame = None
            weights[speaker_id][start_frame:end_frame] = weight_segment
        weights = np.stack(weights)
        logger.debug(
            f"Weights for ({session_id=}, {start_time=}, {duration=}) contains in {num_sups} supervisions. "
            f"Total supervisions duration {total_duration}s."
        )
        return weights, self.speaker_to_idx_map[session_id]
