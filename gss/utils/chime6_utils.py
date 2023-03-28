from collections import defaultdict
from kaldiio import ReadHelper
from lhotse import SupervisionSet, SupervisionSegment


def extract_info_from_kaldi_gss_track1_uid(uid):
    split = uid.split("-")
    if len(split) == 4:
        speaker_id, session_id, speaker_id2, b_e = split
        assert speaker_id == speaker_id2, uid
    elif len(split) == 3:
        speaker_id, session_id, b_e = split
    else:
        raise RuntimeError(f"Bad uid {uid}")

    start, stop = map(float, b_e.split("_"))  # it's sec*100
    return dict(
        recording_id=session_id,
        start=start / 100,
        duration=(stop - start) / 100,
        speaker=speaker_id,
    )


def build_sups_from_rspec(rspec: str, num_chanels=42) -> SupervisionSet:
    ses2numid_map = defaultdict(int)
    vad_sups = []
    with ReadHelper(rspec) as f:
        for uid, weights in f:
            u_info = extract_info_from_kaldi_gss_track1_uid(uid)
            session_id = u_info["recording_id"]
            local_id = ses2numid_map[session_id]
            sup_id = f"{session_id}-{local_id}"
            ses2numid_map[session_id] += 1
            sup = SupervisionSegment(
                id=sup_id,
                recording_id=u_info["recording_id"],
                start=u_info["start"],
                duration=u_info["duration"],
                speaker=u_info["speaker"],
                channel=[*range(num_chanels)],
                custom={"vad_weights": weights},
            )
            vad_sups.append(sup)

    return SupervisionSet.from_segments(vad_sups)


def _test_extract_info_from_kaldi_gss_track1_uid():
    d = extract_info_from_kaldi_gss_track1_uid("P01-S01-P01-10_1000")
    assert {"recording_id": "S01", "start": 0.1, "duration": 9.9, "speaker": "P01"} == d
    d = extract_info_from_kaldi_gss_track1_uid("P01-S01-11_1000")
    assert {
        "recording_id": "S01",
        "start": 0.11,
        "duration": 9.99,
        "speaker": "P01",
    } == d


def _test_build_sups_from_rspec():
    build_sups_from_rspec(
        "ark:/mnt/asr/prisyach/kaldi_chime6_2021/egs/chime6/"
        "s5c_track2_2023/exp/dev_gss_inf_lat_to_vad/vad.ark"
    )
