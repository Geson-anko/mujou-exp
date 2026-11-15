"""The export of a small fake run and its posthoc job, read back with
TensorBoard's own EventAccumulator: the tags and steps of the module
docstring, the PNG bytes as they are, the hyperparameters and the text."""

import json
import struct
import zlib
from pathlib import Path

import pytest
from tb.export import export, png_size
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _png(w: int, h: int) -> bytes:
    def chunk(kind: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(kind + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)

    raw = b"".join(b"\x00" + b"\x80" * (3 * w) for _ in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def runs(tmp_path) -> Path:
    root = tmp_path / "runs"
    run = root / "b" / "r1"
    (run / "provenance").mkdir(parents=True)
    job = {
        "run_id": "r1",
        "batch": "b",
        "seed": 0,
        "overrides": ["+experiment=m17/wt_n16", "learner=pc"],
        "config": {
            "name": "m17/wt_n16",
            "seed": 0,
            "learner": {"kind": "pc"},
            "vision": {"dim": 256, "z_dim": None, "context_masks": [{"blocks": 8}]},
            "optim_vision": {"lr": 1e-4},
            "data": {"tours": {"data_version": "3c3e6805fdaa"}},
        },
    }
    (run / "job.json").write_text(json.dumps(job))
    (run / "CODE.json").write_text(
        json.dumps({"snap": "s" * 40, "mujou_snap": "m" * 40})
    )
    (run / "provenance" / "attempt-1.json").write_text(json.dumps({"host": "std-00"}))
    _write(run / "stream.jsonl", [{"t": 1.0, "frames": 30, "loss": 1.5, "pred": None}])
    _write(
        run / "evals.jsonl",
        [
            {"t": 60.0, "frame": 1800, "split": "val", "loss": 1.0, "rank": 9.0},
            {
                "t": 120.0,
                "frame": 3600,
                "split": "val",
                "loss": 0.5,
                "rank": 12.0,
                "comp_future": 0.8,  # the completion errors (M17)
                "fit_comp_future": 0.6,
            },
        ],
    )
    (run / "baselines.json").write_text(json.dumps({"val": {"init_loss": 2.0}}))
    post = root / "b-post" / "p1"
    (post / "post" / "decode").mkdir(parents=True)
    (post / "job.json").write_text(
        json.dumps({"run_id": "p1", "batch": "b-post", "parent": str(run)})
    )
    d = post / "post" / "decode"
    _write(
        d / "metrics.jsonl",
        [
            {
                "t": 120.0,
                "frame": 3600,
                "split": "val",
                "code": "vit",
                "recon": 0.4,
                "comp_short": 0.7,
            }
        ],
    )
    _write(
        d / "baselines.jsonl",
        [{"t": 0.0, "frame": 0, "split": "val", "code": "pca", "recon": 0.6}],
    )
    (d / "00120.0.png").write_bytes(_png(4, 2))
    (d / "00120.0-restore.png").write_bytes(_png(4, 6))
    (d / "pca-dyn.png").write_bytes(_png(3, 3))
    return root


def _read(path: Path) -> EventAccumulator:
    acc = EventAccumulator(str(path), size_guidance={"images": 0, "scalars": 0})
    acc.Reload()
    return acc


class TestExport:
    def test_the_scalars_images_text_and_hparams(self, runs, tmp_path):
        out = tmp_path / "tb"
        targets = export([runs / "b", runs / "b-post"], out)
        assert targets == [out / "b" / "r1", out / "b" / "r1"]  # the posthoc joins it
        acc = _read(out / "b" / "r1")
        scalars = acc.Tags()["scalars"]
        steps = {t: [(e.step, e.value) for e in acc.Scalars(t)] for t in scalars}
        assert steps["stream/loss"] == [(30, 1.5)]
        assert "stream/pred" not in steps  # a null
        assert steps["eval/val/loss"] == [(60, 1.0), (120, 0.5)]
        assert steps["base/val/run_init/loss"] == [(0, 2.0), (120, 2.0)]
        assert steps["post/val/recon"] == [(120, pytest.approx(0.4))]
        assert steps["base/val/pca/recon"] == [
            (0, pytest.approx(0.6)),
            (120, pytest.approx(0.6)),
        ]
        assert steps["eval/val/comp_future"] == [(120, pytest.approx(0.8))]
        assert steps["eval/val/fit_comp_future"] == [(120, pytest.approx(0.6))]
        assert steps["post/val/comp_short"] == [(120, pytest.approx(0.7))]
        assert steps["hp/loss"] == [(120, 0.5)]
        assert steps["hp/post/recon"] == [(120, pytest.approx(0.4))]  # posthoc's
        images = acc.Tags()["images"]
        assert set(images) == {"post/recon", "post/restore", "base/pca/dyn"}
        img = acc.Images("post/restore")[0]
        assert (img.step, img.width, img.height) == (120, 4, 6)
        assert (
            img.encoded_image_string
            == (
                runs / "b-post" / "p1" / "post" / "decode" / "00120.0-restore.png"
            ).read_bytes()
        )
        assert acc.Tags()["tensors"]  # the text and the hparams' summaries
        text = acc.Tensors("run/text_summary")[0].tensor_proto.string_val[0].decode()
        assert "3c3e6805fdaa" in text and "s" * 40 in text and "std-00" in text

    def test_an_export_again_replaces_its_events(self, runs, tmp_path):
        out = tmp_path / "tb"
        export([runs / "b" / "r1"], out)
        export([runs / "b" / "r1"], out)
        assert len(list((out / "b" / "r1").glob("events.out.tfevents.*"))) == 1

    def test_png_size(self):
        assert png_size(_png(7, 5)) == (7, 5)
        with pytest.raises(ValueError, match="PNG"):
            png_size(b"GIF89a" + b"\0" * 30)
