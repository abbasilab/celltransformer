"""Nested-jagged tensor ↔ padded tensor conversion utilities."""

import torch


def padded_from_njt(nested_tensor) -> torch.Tensor:
    tensors = nested_tensor.unbind()

    bs = len(tensors)
    device = tensors[0].device
    maxlen = max([t.shape[0] for t in tensors])

    padding_mask = torch.zeros((bs, maxlen), dtype=torch.bool, device=device)

    for i, tens in enumerate(tensors):
        padding_mask[i, :tens.shape[0]] = 1

    padded_tens = nested_tensor.to_padded_tensor(0.0, (bs, maxlen, -1))
    # will reshape to (bs, maxlen, feature_dim)

    return padded_tens, padding_mask


def padded_to_nested(padded_tens: torch.Tensor, padding_mask: torch.Tensor):
    lengths = padding_mask.sum(dim=1)
    tensors = [padded_tens[i, :lengths[i]] for i in range(len(lengths))]
    return torch.nested.nested_tensor(tensors, layout=torch.jagged)
