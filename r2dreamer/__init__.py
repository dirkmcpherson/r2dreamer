"""R2-Dreamer: Redundancy-Reduced World Models for Reinforcement Learning.

Thin re-export layer. Adds the repo root to sys.path so the existing
module-level relative imports (``import networks``, etc.) keep working
both here and in the standalone ``python train.py`` workflow.
"""

import os as _os
import sys as _sys

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

# -- RSSM / world-model core --------------------------------------------------
from rssm import RSSM, Deter  # noqa: E402

# -- Network building blocks ---------------------------------------------------
from networks import (  # noqa: E402
    BlockLinear,
    Conv2dSamePad,
    ConvDecoder,
    ConvEncoder,
    LambdaLayer,
    MLP,
    MLPHead,
    MultiDecoder,
    MultiEncoder,
    Projector,
    ReturnEMA,
    RMSNorm2D,
)

# -- Distributions -------------------------------------------------------------
from distributions import (  # noqa: E402
    Bound,
    MSEDist,
    MultiOneHotDist,
    OneHotDist,
    SymlogDist,
    TwoHot,
    kl,
    symexp,
    symexp_twohot,
    symlog,
    symlog_mse,
)

# -- Optimisation --------------------------------------------------------------
from optim import LaProp, clip_grad_agc_  # noqa: E402

# -- Utilities -----------------------------------------------------------------
import tools  # noqa: E402

__all__ = [
    # RSSM
    "RSSM",
    "Deter",
    # Networks
    "BlockLinear",
    "Conv2dSamePad",
    "ConvDecoder",
    "ConvEncoder",
    "LambdaLayer",
    "MLP",
    "MLPHead",
    "MultiDecoder",
    "MultiEncoder",
    "Projector",
    "ReturnEMA",
    "RMSNorm2D",
    # Distributions
    "Bound",
    "MSEDist",
    "MultiOneHotDist",
    "OneHotDist",
    "SymlogDist",
    "TwoHot",
    "kl",
    "symexp",
    "symexp_twohot",
    "symlog",
    "symlog_mse",
    # Optimisation
    "LaProp",
    "clip_grad_agc_",
    # Utilities
    "tools",
]
