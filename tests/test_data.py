"""Basic data tests - run with: pytest   (or the Testing panel in VS Code)."""
import pytest

from shopsmart.config import CONFIG, dataset_path
from shopsmart.data import list_datasets, load
from shopsmart.data.validate import run_checks


@pytest.mark.parametrize("name", list(CONFIG["datasets"]))
def test_dataset_file_exists(name):
    assert dataset_path(name).exists(), f"missing {dataset_path(name)}"


def test_list_datasets_has_all_files():
    df = list_datasets()
    assert df["exists"].all()
    assert df.loc["interactions", "rows"] > 100_000


def test_interaction_dates_are_parsed():
    df = load("interactions", nrows=1000)
    assert str(df["Timestamp"].dtype).startswith("datetime64")


def test_all_validation_checks_pass():
    failed = [(d, n, detail) for d, n, ok, detail in run_checks() if not ok]
    assert not failed, failed
