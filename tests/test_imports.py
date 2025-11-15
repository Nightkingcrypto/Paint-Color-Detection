def test_imports():
    import src
    from src.dataset import ColorFolderDataset
    from src.models import ColorVAE
    from src.config import DATASET_ROOT

    # just check attributes exist
    assert DATASET_ROOT is not None
