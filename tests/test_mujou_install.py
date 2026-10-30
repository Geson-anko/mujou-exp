"""Check that mujou is the editable install of the git submodule."""

from pathlib import Path

import mujou

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestMujouInstall:
    def test_mujou_comes_from_the_submodule(self):
        assert Path(mujou.__file__).resolve().is_relative_to(REPO_ROOT / "mujou")
