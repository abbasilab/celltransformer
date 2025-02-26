import torch
from scvi import distributions
from torch import nn
from math import e

from brainformr import model
# TODO: split tests into smaller parts, link model generation with fixture

def test_xformer():
    embed_dim = 64
    depth = 3
    num_heads = 8
    dropout = 0.0
    zero_attn = True

    for bias in (True, False):
        xformer_blocks = model.set_up_transformer_layers(
            embed_dim, num_heads, depth, dropout, bias, zero_attn
        )
        assert isinstance(
            xformer_blocks, nn.TransformerEncoder
        ), "`layers` not type(obj) == nn.TransformerEncoder"

        for layer in xformer_blocks.layers: 
            # bias_k / bias_q are set to None in constructor regardless of
            # whether bias=True to nn.MultiHeadedAttention
            assert (layer.get_submodule('self_attn').bias_k is None) is not bias, f"bias={bias} in argument however bias={not bias} in layer"

    n_param = 843456
    num_param_model = sum([p.numel() for p in xformer_blocks.parameters()])
    assert num_param_model == n_param, f"Number of parameters in model is {num_param_model} but should be {n_param}."
    
    # check if the KH Normal initialization for the attn layers works
    # see fn definition for `set_up_transformer_layers` for details and why this is necessary
    assert (
        torch.allclose(
            xformer_blocks.layers[0].self_attn.out_proj.weight,
            xformer_blocks.layers[1].self_attn.out_proj.weight,
        )
        is False
    ), "Weights are equal for blocks 0 and 1"
    assert (
        torch.allclose(
            xformer_blocks.layers[0].self_attn.out_proj.weight,
            xformer_blocks.layers[2].self_attn.out_proj.weight,
        )
        is False
    ), "Weights are equal for blocks 0 and 2"
    assert (
        torch.allclose(
            xformer_blocks.layers[1].self_attn.out_proj.weight,
            xformer_blocks.layers[2].self_attn.out_proj.weight,
        )
        is False
    ), "Weights are equal for blocks 0 and 1"

    random_data = torch.randn(5, 64)
    output = xformer_blocks(random_data)
    loss = output.sum()
    loss.backward()

    assert output.shape == (5, 64), "Output shape is not (5, 64)"
    assert torch.allclose(output, random_data) is False, "Output is not equal to input"
    assert (wt.grad is not None for _, wt in xformer_blocks.named_parameters()), "Gradients are not None after test fwd pass"



def test_zinb_proj():
    embed_dim = 64
    n_genes = 500
    eps = 1e-15
    n_tok = 5

    zinb_proj = model.ZINBProj(embed_dim, n_genes, eps)

    assert isinstance(
        zinb_proj, model.ZINBProj
    ), "`zinb_proj` was not type(obj) == model.ZINBProj"

    random_data = torch.randn(n_tok, 64)
    output = zinb_proj(random_data)

    for key, matrix in output.items():
        if key != "zi_logits":
            assert torch.all(
                torch.ge(matrix, 0.0)
            ), f"All values in mu/theta/scale projections (here {key}) should be greater than or equal to 0; eps was set to {eps}"

    zinb = distributions.ZeroInflatedNegativeBinomial(
        mu=output["mu"],
        theta=output["theta"],
        zi_logits=output["zi_logits"],
        scale=output["scale"],
    )

    counts_sim = torch.randint(low=0, high=1000, size=(n_tok, n_genes))
    
    assert isinstance(
        zinb, distributions.ZeroInflatedNegativeBinomial
    ), "`zinb` was not type(obj) == distributions.ZeroInflatedNegativeBinomial"

    log_prob = zinb.log_prob(counts_sim)
    assert log_prob.shape == (n_tok, n_genes), f"Log prob shape is not ({n_tok}, {n_genes})"

    assert torch.all(
        torch.le(log_prob, e)
    ), "All values in log_prob should be less than e"
    
    assert torch.all(
        ~torch.isnan(log_prob)
    ), "NaN in log_prob!"