"""BAR-ATH tactical ATD with the broadcast ball filled into its ball gaps (E13, D27 reopened).

    from solution.sources import fused
    match = fused.load()            # needs `make broadcast-align` to have written the alignment

The alignment is read from cache/broadcast/BAR-ATH_align.npz (solution/broadcast.align, written
by tools/broadcast_align.py); the fill rule is solution/broadcast.fuse_ball. Everything else is
solution/sources/atd.load() unchanged. labels is None, as for atd.
"""

import os

import numpy as np

from solution import broadcast as bc
from solution.match import CACHE_DIR, validate
from solution.sources import atd
from games import GAME  # noqa: E402  (atd puts the repo root on sys.path)

ALIGN_PATH = os.path.join(CACHE_DIR, "broadcast", GAME + "_align.npz")


def load(align_path=ALIGN_PATH, data_dir=None):
    if not os.path.isfile(align_path):
        raise IOError("no broadcast alignment at {}: run `make broadcast-align` first".format(
            align_path))
    z = np.load(align_path, allow_pickle=False)
    tac = atd.load(data_dir=data_dir)
    bro = bc.flipped(atd.load_broadcast(data_dir=data_dir), str(z["flip"]))
    if len(z["state"]) != tac.n_frames:
        raise ValueError("alignment covers {} frames, the ATD {}".format(
            len(z["state"]), tac.n_frames))
    fused, _ = bc.fuse_ball(tac, bro, z["state"], z["bro_frame"])
    return validate(fused)


def merged(match):
    """`match` with the ball merge applied, for methods with merge_ball (D27, D32):
      - BAR-ATH: broadcast.fuse_ball with the real broadcast ball and the alignment cache;
      - a degraded proxy match: broadcast.imitate_fill from its clean IDSSE match and simulator
        seed (notes["degraded"]["seed"]), the imitation tools/broadcast_fill_check.py measured.
        The input here already carries the D12 ball stress, so the restored frames are clean
        (fill_check restored before the stress): slightly optimistic. Stresses that move the
        timeline or the positions (tempo, speed_factor, prekickoff) are refused.
    """
    if match.name == GAME:
        if not os.path.isfile(ALIGN_PATH):
            raise IOError("no broadcast alignment at {}: run `make broadcast-align` first".format(
                ALIGN_PATH))
        z = np.load(ALIGN_PATH, allow_pickle=False)
        bro = bc.flipped(atd.load_broadcast(), str(z["flip"]))
        return bc.fuse_ball(match, bro, z["state"], z["bro_frame"])[0]
    seed = (match.notes.get("degraded") or {}).get("seed")
    if seed is None or not match.name.startswith("idsse_"):
        raise ValueError("{}: no ball merge for this match (BAR-ATH or a degraded IDSSE match "
                         "only)".format(match.name))
    refused = {"tempo", "speed_factor", "prekickoff"} & set(match.notes)
    if refused:
        raise ValueError("{}: ball merge imitation does not support {}".format(
            match.name, sorted(refused)))
    from solution.sources import idsse
    clean = idsse.load(match.name[len("idsse_"):])
    return bc.imitate_fill(match, clean, seed)
