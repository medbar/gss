import logging
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

from collections import defaultdict
from lhotse import CutSet, SupervisionSet, SupervisionSegment
from kaldiio import ReadHelper
from gss.utils.logging_utils import get_logger
from gss.utils.chime6_utils import extract_info_from_kaldi_gss_track1_uid
from gss.utils.data_utils import _samples_to_stft_frames
from gss_helpers.stft_utils import linear_resample

logger = get_logger()


@dataclass  # (hash=True)
class TFArkWeights:
    speaker_to_idx_map: Dict[str, Dict[str, int]]
    rspec: str
    garbage_class: bool = True
    # cuts with custom vad_weights supervision
    # see utils.chime6_utils.build_sups_from_rspec
    # cuts: "CutSet" = None
    eps: float = 1e-10
    sups: Dict[str, List[Dict[str, Any]]] = None
    stft_shift: int = 256
    stft_size: int = 1024
    sr: int = 16000
    stft_fading: bool = True
    stft_pad: bool = True

    # activity_thr: float = 0.99

    def __post_init__(self):
        # if self.supervisions_index is None:
        #     self.supervisions_index = self.cuts.index_supervisions()
        self.sups = defaultdict(list)
        with ReadHelper(self.rspec) as f:
            for key, tf in f:
                # tf - [time, freq] init probs
                uinfo = extract_info_from_kaldi_gss_track1_uid(key)
                element = dict(
                    start=uinfo["start"],
                    duration=uinfo["duration"],
                    end=uinfo["start"] + uinfo["duration"],
                    speaker=uinfo["speaker"],
                    tf_probs=tf,
                )
                self.sups[uinfo["recording_id"]].append(element)

        logger.info(
            f"Initialized Time-Freq Weights. {self.speaker_to_idx_map = }, "
            f"loaded {sum(len(s) for s in self.sups.values())} supervisions"
        )

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

    def get_sups(self, session_id, start_time, duration):
        end_time = start_time + duration
        sups = []
        for s in self.sups[session_id]:
            if (
                start_time <= s["start"] <= end_time
                or start_time <= s["end"] <= end_time
            ):
                s_with_offset = dict(
                    start=s["start"] - start_time,
                    duration=s["duration"],
                    end=s["start"] - start_time + s["duration"],
                    speaker=s["speaker"],
                    tf_probs=s["tf_probs"],
                )
                sups.append(s_with_offset)
        return sups

    def get_weights(self, session_id, start_time, duration):
        weights_no_fading, idx = self.get_weights_no_fading(
            session_id, start_time, duration
        )
        if not self.stft_fading:
            return weights_no_fading, idx

        pad_width = np.zeros([weights_no_fading.ndim, 2], dtype=np.int64)
        pad_width[-1, :] = self.stft_size // self.stft_shift - 1
        return np.pad(weights_no_fading, pad_width, mode="constant"), idx

    def get_weights_no_fading(self, session_id, start_time, duration):
        idx = self.speaker_to_idx_map[session_id]
        cut_num_frames = self._sec_to_stft_frames(duration, fading=False)
        weights = [np.zeros(cut_num_frames) for _ in range(len(idx))]
        if self.garbage_class:
            weights += [np.ones(cut_num_frames)]
        if cut_num_frames < 2:
            logger.warning(
                f"Cuts ({session_id=}, {start_time=}, {duration=}) too small"
                f"{cut_num_frames=}. Return all zeros."
            )
            return np.stack(weights), idx
        num_sups = 0
        total_frames = 0
        sups = self.get_sups(session_id, start_time, duration)
        for sup in sups:
            num_sups += 1
            speaker_id = idx[sup["speaker"]]
            sup_dur_frames = self._sec_to_stft_frames(sup["duration"], fading=False)
            weight_segment = sup["tf_prob"]
            # weight_segment - is librosa stft with 1024, 256, central=True, pad='constant'
            # so signal paded by:
            #    512 (2 frames) in the left
            #    1 not full (or pad) and one pad on the right
            assert 2 + sup_dur_frames <  weight_segment.shape[0], (
                f"{sup_dur_frames} {weight_segment.shape}"
            )
            weight_segment = weight_segment[2:2+sup_dur_frames]
            start_frame = 0
            end_frame = cut_num_frames
            # finding start frame
            if sup["start"] < 0:
                offset_frames = self._sec_to_stft_frames(-sup["start"], fading=False)
                logger.debug(
                    f"Detected {sup['start'] = }. Start {offset_frames = } frames."
                )
                weight_segment = weight_segment[offset_frames:]
            else:
                start_frame += self._sec_to_stft_frames(sup["start"], fading=False)
            start_frame = max(start_frame, 0)
            if start_frame >= end_frame:
                logger.warning(
                    f"Supervision {start_frame}:{end_frame} so small. Skip it"
                )
                continue
            # finding end frame
            if sup["end"] > duration:
                offset_frames = end_frame - start_frame
                logger.debug(
                    f"Detected {sup['end']=} larger than {duration=}. "
                    f"End {offset_frames = } frames."
                )
                weight_segment = weight_segment[:offset_frames]
            end_frame = start_frame + weight_segment.shape[0]
            assert (
                end_frame - start_frame == weight_segment.shape[0]
            ), f"{end_frame=} {start_frame=} {weight_segment.shape[0]=}"
            total_frames += weight_segment.shape[0]
            logger.debug(
                f"Paste {weight_segment.shape=} into [{start_frame=}:{end_frame=}] segment"
            )
            if weight_segment.shape[0] > 0 and end_frame - start_frame > 1:
                weights[speaker_id][start_frame:end_frame] = weight_segment
            else:
                logging.warning(
                    f"Bad vad supervision: {duration=} {sup['start']=} "
                    f"{sup['end']=} {weight_segment.shape=} "
                    f"{start_frame=}:{end_frame=}"
                )
        assert (
            num_sups > 0
        ), f"Weights for ({session_id=}, {start_time=}, {duration=}) does not exist. Wrong weights manifest?"
        logger.debug(
            f"Weights for ({session_id=}, {start_time=}, {duration=}) contains in {num_sups} supervisions. "
            f"Total supervisions duration {total_frames} frames."
        )
        # FxKxT
        weights = np.stack([w.T for w in weights], axis=1)
        return weights, idx
