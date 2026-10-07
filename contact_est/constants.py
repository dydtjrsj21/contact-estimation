"""Hand / label constants shared by simulation and offline code (no Isaac imports)."""

FINGERS = ("index", "middle", "ring", "thumb")
"""Finger names as used by the Allegro USD (``<finger>_joint_<k>``, ``<finger>_link_<k>``)."""

FINGERTIP_BODIES = tuple(f"{f}_link_3" for f in FINGERS)
"""Distal links that carry the simulated 3-axis fingertip force sensor (the biotac tip bodies have no
collision geometry, so every fingertip contact lands on these links)."""

GT_LINKS = ("palm_link",) + tuple(f"{f}_link_{k}" for f in FINGERS for k in range(4))
"""Every hand link with collision geometry. Ground-truth object contact is recorded for all of them;
choose the estimation targets offline. Labels only, never estimator input."""

DRAFT_REGIONS = ("palm_link", "index_link_1", "index_link_2", "middle_link_1")
"""The 4 regions proposed in the experiment draft (palm, index proximal/middle, middle proximal)."""

REGION_SHORT = {
    "palm_link": "palm",
    **{f"{f}_link_{k}": f"{f[0].upper()}{k}" for f in FINGERS for k in range(4)},
    "index_link_1": "I-prox",
    "index_link_2": "I-mid",
    "middle_link_1": "M-prox",
}

# hysteresis thresholds on |force| between a link and the object [N]
GT_CONTACT_ON_N = 0.20
GT_CONTACT_OFF_N = 0.05
