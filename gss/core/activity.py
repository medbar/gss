from dataclasses import dataclass

import numpy as np
from lhotse import CutSet
from gss.utils.logging_utils import get_logger

logger = get_logger()


@dataclass  # (hash=True)
class Activity:
    garbage_class: bool = False
    cuts: "CutSet" = None

    def __post_init__(self):
        self.activity = {}  # Is it required?
        self.speaker_to_idx_map = {}
        for cut in self.cuts:
            self.speaker_to_idx_map[cut.recording_id] = {
                spk: idx
                for idx, spk in enumerate(
                    sorted(set(s.speaker for s in cut.supervisions))
                )
            }
        self.supervisions_index = self.cuts.index_supervisions()
        logger.info(
            f"Initialized Activity. {len(self.supervisions_index) = }. "
            f"{self.garbage_class = }"
        )

    def get_activity(self, session_id, start_time, duration):
        cut = self.cuts[session_id].truncate(
            offset=start_time,
            duration=duration,
            _supervisions_index=self.supervisions_index,
        )
        activity_mask = cut.speakers_audio_mask(
            speaker_to_idx_map=self.speaker_to_idx_map[session_id]
        )
        if self.garbage_class is False:
            activity_mask = np.r_[activity_mask, [np.zeros_like(activity_mask[0])]]
        elif self.garbage_class is True:
            activity_mask = np.r_[activity_mask, [np.ones_like(activity_mask[0])]]
        idx = self.speaker_to_idx_map[session_id]
        logger.debug(
            f"for cut({session_id=} {start_time=} {duration=}) "
            f"num_supervisions = {len(cut.supervisions)}, "
            f"{activity_mask.shape = }, {activity_mask.sum(axis=1) = },"
            f" {idx = }"
        )
        return activity_mask, idx
