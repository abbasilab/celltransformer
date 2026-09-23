"""Learning rate schedules.

Vendored from brainformr, originally adapted from HuggingFace Transformers:
https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/optimization.py#L297
"""

import numpy as np
from torch import optim


def get_inverse_sqrt_schedule_with_plateau(
    optimizer,
    num_warmup_steps: int,
    timescale: int = None,
    last_epoch: int = -1,
    plateau_length: int = 0,
):
    """Inverse square-root LR schedule with linear warmup and optional plateau.

    After warmup, holds LR at peak for ``plateau_length`` steps, then decays
    as 1/sqrt(t).

    Parameters
    ----------
    optimizer
        The optimizer to schedule.
    num_warmup_steps
        Steps for the linear warmup phase.
    timescale
        Decay timescale. Defaults to ``num_warmup_steps``.
    last_epoch
        Index of the last epoch when resuming.
    plateau_length
        Steps to hold peak LR before decay begins.
    """
    if timescale is None:
        timescale = num_warmup_steps

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < (num_warmup_steps + plateau_length):
            return 1
        shift = timescale - num_warmup_steps - plateau_length
        decay = 1.0 / np.sqrt((current_step + shift) / timescale)
        return decay

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
