import torch

from models.deepsets import DeepSetsEncoder


def test_deepsets_permutation_invariant():
    encoder = DeepSetsEncoder(d_in=4)
    x = torch.randn(1, 5, 4)
    perm = x[:, torch.randperm(5), :]
    out1 = encoder(x)
    out2 = encoder(perm)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_deepsets_squeezes_extra_dimension():
    encoder = DeepSetsEncoder(d_in=4)
    x = torch.randn(2, 3, 4)
    x_extra = x.unsqueeze(1)
    out_plain = encoder(x)
    out_extra = encoder(x_extra)
    assert torch.allclose(out_plain, out_extra, atol=1e-6)
