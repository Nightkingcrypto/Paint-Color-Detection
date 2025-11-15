import torch

def test_torch_works():
    """
    Simple health-check test for CI:
    verifies that PyTorch is installed and basic tensor ops work.
    This is enough to show GitHub Actions running automated tests.
    """
    x = torch.ones(2, 3)
    y = torch.ones(2, 3)
    z = x + y

    assert z.shape == (2, 3)
    assert torch.all(z == 2)
