from pathlib import Path

import pytest

from exp.lib.fleet.contract import verify_manifest, write_json, write_manifest


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    (tmp_path / "result.json").write_text('{"status": "completed"}\n')
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "0001.0.pt").write_bytes(b"\x00" * 100)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "attempt-1.out").write_text("not in the manifest\n")
    return tmp_path


class TestVerifyManifest:
    def test_what_was_written_verifies_and_a_changed_byte_does_not(self, run_dir: Path):
        write_manifest(run_dir)
        assert verify_manifest(run_dir) is None
        (run_dir / "logs" / "attempt-1.out").write_text("changed, not listed\n")
        assert verify_manifest(run_dir) is None
        (run_dir / "ckpt" / "0001.0.pt").write_bytes(b"\x00" * 99 + b"\x01")
        assert verify_manifest(run_dir) == "ckpt/0001.0.pt: sha256 differs"
        (run_dir / "ckpt" / "0001.0.pt").unlink()
        assert "ckpt/0001.0.pt" in (verify_manifest(run_dir) or "")

    def test_the_list_form(self, run_dir: Path):
        write_manifest(run_dir)
        from exp.lib.fleet.contract import read_json

        files = read_json(run_dir / "manifest.json")["files"]
        listed = [{"path": k, **v} for k, v in files.items()]
        write_json(run_dir / "manifest.json", {"schema": 1, "files": listed})
        assert verify_manifest(run_dir) is None

    @pytest.mark.parametrize(
        "manifest",
        [
            ["not", "an", "object"],
            {"schema": 1},
            {"schema": 1, "files": {"result.json": "deadbeef"}},
            {"schema": 1, "files": {"result.json": {"size": 24}}},
            {"schema": 1, "files": [{"sha256": "deadbeef"}]},
            {"schema": 1, "files": ["result.json"]},
        ],
    )
    def test_a_malformed_manifest_is_a_problem_not_an_error(
        self, run_dir: Path, manifest
    ):
        write_json(run_dir / "manifest.json", manifest)
        problem = verify_manifest(run_dir)
        assert isinstance(problem, str) and "manifest.json" in problem
