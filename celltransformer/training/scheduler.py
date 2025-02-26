from typing import Optional

import numpy as np
import torch
from torch import optim

"""
    Copied from huggingface https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/optimization.py#L297
"""

def get_inverse_sqrt_schedule(
    optimizer, num_warmup_steps: int, timescale: int = None, last_epoch: int = -1
):
    """
    Create a schedule with an inverse square-root learning rate, from the initial lr set in the optimizer, after a
    warmup period which increases lr linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        timescale (`int`, *optional*, defaults to `num_warmup_steps`):
            Time scale.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    # Note: this implementation is adapted from
    # https://github.com/google-research/big_vision/blob/f071ce68852d56099437004fd70057597a95f6ef/big_vision/utils.py#L930
 
    if timescale is None:
        timescale = num_warmup_steps

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        shift = timescale - num_warmup_steps
        decay = 1.0 / np.sqrt((current_step + shift) / timescale)
        return decay

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_inverse_sqrt_schedule_with_plateau(
    optimizer,
    num_warmup_steps: int,
    timescale: int = None,
    last_epoch: int = -1,
    plateau_length: int = 0,
):
    """
    Create a schedule with an inverse square-root learning rate, from the initial lr set in the optimizer, after a
    warmup period which increases lr linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        timescale (`int`, *optional*, defaults to `num_warmup_steps`):
            Time scale.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        plateu_length (`int`, *optional*, defaults to 0):
            The number of steps for the plateau phase.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    # Note: this implementation is adapted from
    # https://github.com/google-research/big_vision/blob/f071ce68852d56099437004fd70057597a95f6ef/big_vision/utils.py#L930

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


def decay_to_zero(optimizer: torch.optim.Optimizer, lr: Optional[float], num_steps: int = 1000):
    def lr_lambda(current_step: int):
        lr_steps = np.linspace(1, 0, num_steps)

        return lr_steps[current_step]

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    if lr is not None:
        scheduler.base_lrs = [lr]        

    return scheduler
