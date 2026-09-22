"""AnyLM2Jev: reproduce the Jev decision interface on a small open LM.

The package is intentionally split so that the data/schema layer has no heavy
dependencies and can be imported without torch:

    schema    -- DecisionItem, typed choices/scores/nouls
    prompts   -- paraphrase x option-order "views"
    modeling  -- frozen LM loading and option-label readout
    induce    -- build targets q* and p_raw from a frozen LM
    head      -- option-score head
    distill   -- forward-KL training
    calibrate -- temperature scaling and ECE
    metrics   -- accuracy / calibration / order-consistency
    data      -- WANLI and synthetic decision sets
"""

__version__ = "0.1.0"
