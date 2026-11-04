from pathlib import Path

import pytest

from tests.lib.fleet.fleetkit import Fleet


@pytest.fixture
def fleet(tmp_path: Path) -> Fleet:
    """A fleet of two hosts (h0, h1) in ``tmp_path``."""
    return Fleet(tmp_path)
