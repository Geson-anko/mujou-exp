import json
import os

import pytest

from exp.lib.rundir import (
    Jsonl,
    ManifestError,
    RunDir,
    verify_manifest,
    write_json,
    write_manifest,
)


class TestWriteJson:
    def test_a_failed_write_leaves_the_old_file_and_no_temporary(self, tmp_path):
        """The new content goes to a temporary file first: if serializing
        fails, the old file is whole and nothing else is left."""
        path = tmp_path / "status.json"
        write_json(path, {"frame": 1})
        with pytest.raises(TypeError):
            write_json(path, {"frame": object()})
        assert json.loads(path.read_text()) == {"frame": 1}
        assert os.listdir(tmp_path) == ["status.json"]

    def test_it_replaces_the_file(self, tmp_path):
        path = tmp_path / "a.json"
        write_json(path, {"x": 1})
        write_json(path, {"x": 2})
        assert json.loads(path.read_text()) == {"x": 2}

    def test_the_mode_is_that_of_a_plain_open(self, tmp_path):
        """The file gets the umask's mode, as ``open`` gives (others can read
        it on the NAS), not the 0o600 of ``tempfile.mkstemp``."""
        plain = tmp_path / "plain.json"
        plain.write_text("{}")
        path = tmp_path / "job.json"
        write_json(path, {})
        assert path.stat().st_mode == plain.stat().st_mode


class TestJsonl:
    def test_the_same_records_make_the_same_bytes(self, tmp_path):
        records = [{"t": 1.0, "preq": 0.1}, {"t": 2.0, "preq": float("nan")}]
        for name in ("a.jsonl", "b.jsonl"):
            with Jsonl(tmp_path / name) as f:
                for r in records:
                    f.append(r)
        a, b = (tmp_path / n for n in ("a.jsonl", "b.jsonl"))
        assert a.read_bytes() == b.read_bytes()
        assert a.read_bytes().count(b"\n") == 2

    def test_sync_returns_the_size(self, tmp_path):
        path = tmp_path / "e.jsonl"
        with Jsonl(path) as f:
            f.append({"t": 1.0})
            assert f.sync() == path.stat().st_size

    def test_a_line_is_on_disk_as_soon_as_it_is_written(self, tmp_path):
        path = tmp_path / "e.jsonl"
        with Jsonl(path) as f:
            f.append({"t": 1.0})
            assert json.loads(path.read_text()) == {"t": 1.0}


class TestTruncate:
    def test_a_resumed_file_is_that_of_a_run_never_stopped(self, tmp_path):
        """Lines 0..2 are written, the checkpoint takes the offset, lines 3..4
        are written and lost to a crash; the resumed run cuts back and writes
        3..5 again: the file is that of 0..5 in one go."""
        whole = RunDir(tmp_path / "whole")
        whole.root.mkdir()
        for k in range(6):
            whole.jsonl("evals.jsonl").append({"k": k})
        whole.close()

        crashed = RunDir(tmp_path / "crashed")
        crashed.root.mkdir()
        for k in range(3):
            crashed.jsonl("evals.jsonl").append({"k": k})
        offsets = crashed.sync()
        for k in range(3, 5):
            crashed.jsonl("evals.jsonl").append({"k": k})
        crashed.close()

        resumed = RunDir(crashed.root)
        resumed.truncate(offsets)
        for k in range(3, 6):
            resumed.jsonl("evals.jsonl").append({"k": k})
        resumed.close()
        got = (crashed.root / "evals.jsonl").read_bytes()
        assert got == (whole.root / "evals.jsonl").read_bytes()

    def test_a_checkpoint_before_the_first_evaluation_cuts_it_back(self, tmp_path):
        """The checkpoint is taken before ``evals.jsonl`` is opened, then an
        evaluation writes a line and the run crashes: the offsets name it with
        0 bytes, so the resumed run drops that line and writes it once."""
        whole = RunDir(tmp_path / "whole")
        whole.root.mkdir()
        whole.jsonl("stream.jsonl").append({"t": 1})
        whole.jsonl("evals.jsonl").append({"t": 2, "split": "val"})
        whole.close()

        crashed = RunDir(tmp_path / "crashed")
        crashed.root.mkdir()
        crashed.jsonl("stream.jsonl").append({"t": 1})
        offsets = crashed.sync()
        assert offsets == {"stream.jsonl": 9, "evals.jsonl": 0}
        crashed.jsonl("evals.jsonl").append({"t": 2, "split": "val"})
        crashed.close()

        resumed = RunDir(crashed.root)
        resumed.truncate(offsets)
        assert resumed.sync() == offsets  # unopened: their sizes on disk
        resumed.jsonl("evals.jsonl").append({"t": 2, "split": "val"})
        resumed.close()
        for name in ("stream.jsonl", "evals.jsonl"):
            got = (crashed.root / name).read_bytes()
            assert got == (whole.root / name).read_bytes()

    def test_a_missing_file_at_offset_zero_stays_missing(self, tmp_path):
        run = RunDir(tmp_path)
        run.truncate({"stream.jsonl": 0, "evals.jsonl": 0})
        assert list(tmp_path.iterdir()) == []

    def test_a_file_shorter_than_its_offset_raises(self, tmp_path):
        run = RunDir(tmp_path)
        (tmp_path / "stream.jsonl").write_bytes(b"{}\n")
        with pytest.raises(ValueError, match="fewer than"):
            run.truncate({"stream.jsonl": 10})

    def test_an_open_file_raises(self, tmp_path):
        run = RunDir(tmp_path)
        run.jsonl("stream.jsonl").append({"t": 0.0})
        with pytest.raises(ValueError, match="open"):
            run.truncate({"stream.jsonl": 0})
        run.close()


class TestJsonlNames:
    def test_a_file_neither_metrics_nor_unlisted_raises(self, tmp_path):
        """A resume would not cut it back, so its lines could repeat."""
        run = RunDir(tmp_path)
        with pytest.raises(ValueError, match="not one of the metrics"):
            run.jsonl("other.jsonl")

    def test_timing_is_open_to_append_but_not_cut_back(self, tmp_path):
        run = RunDir(tmp_path)
        run.jsonl("timing.jsonl").append({"wall": 1.0})
        assert run.sync() == {"stream.jsonl": 0, "evals.jsonl": 0}
        run.close()

    def test_the_metrics_files_can_be_named(self, tmp_path):
        run = RunDir(tmp_path, metrics=("post.jsonl",))
        run.jsonl("post.jsonl").append({"t": 1})
        assert run.sync() == {"post.jsonl": 9}
        run.close()


class TestStop:
    def test_the_request_is_the_file(self, tmp_path):
        run = RunDir(tmp_path)
        assert not run.stop_requested()
        (tmp_path / "stop.request").write_text("drain\n")
        assert run.stop_requested()


def _complete_run(root):
    run = RunDir(root)
    run.write_json("job.json", {"run_id": "r"})
    run.write_json("result.json", {"status": "completed"})
    run.jsonl("evals.jsonl").append({"t": 1.0, "roll10": 0.1})
    (root / "ckpt").mkdir()
    (root / "ckpt" / "0001.0.pt").write_bytes(b"\x00weights")
    for unlisted in ("logs/attempt-1.out", "exit/attempt-1.json", "resume/latest"):
        (root / unlisted).parent.mkdir(exist_ok=True)
        (root / unlisted).write_text("x")
    run.write_json("status.json", {"frame": 30})
    return run.write_manifest()


class TestManifest:
    def test_it_lists_the_files_but_the_unlisted(self, tmp_path):
        manifest = _complete_run(tmp_path)
        assert sorted(manifest["files"]) == [
            "ckpt/0001.0.pt",
            "evals.jsonl",
            "job.json",
            "result.json",
        ]
        assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
        verify_manifest(tmp_path)

    def test_one_byte_changed_fails(self, tmp_path):
        _complete_run(tmp_path)
        path = tmp_path / "ckpt" / "0001.0.pt"
        data = bytearray(path.read_bytes())
        data[3] ^= 1
        path.write_bytes(bytes(data))
        with pytest.raises(ManifestError, match="ckpt/0001.0.pt: sha256"):
            verify_manifest(tmp_path)

    def test_a_missing_or_resized_file_fails(self, tmp_path):
        _complete_run(tmp_path)
        (tmp_path / "result.json").unlink()
        with (tmp_path / "evals.jsonl").open("a") as f:
            f.write("\n")
        with pytest.raises(ManifestError) as exc:
            verify_manifest(tmp_path)
        assert "result.json: missing" in str(exc.value)
        assert "evals.jsonl: " in str(exc.value)

    def test_the_unlisted_may_change(self, tmp_path):
        _complete_run(tmp_path)
        (tmp_path / "status.json").write_text("{}")
        (tmp_path / "logs" / "attempt-1.out").write_text("more")
        (tmp_path / "stop.request").write_text("")
        verify_manifest(tmp_path)

    def test_another_schema_fails(self, tmp_path):
        _complete_run(tmp_path)
        write_json(tmp_path / "manifest.json", {"schema": 2, "files": {}})
        with pytest.raises(ManifestError, match="schema 2"):
            verify_manifest(tmp_path)

    def test_a_temporary_file_is_not_listed(self, tmp_path):
        (tmp_path / ".result.json.123.tmp").write_text("{")
        assert write_manifest(tmp_path)["files"] == {}

    def test_no_manifest_fails(self, tmp_path):
        with pytest.raises(ManifestError, match="no manifest.json"):
            verify_manifest(tmp_path)

    def test_the_manifest_of_the_same_files_is_the_same(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        assert _complete_run(a) == _complete_run(b)
        assert write_manifest(a) == write_manifest(b)
