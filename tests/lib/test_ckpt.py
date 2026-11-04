import mujou.pc as pc
import mujou.testing
import pytest
import torch
from torch import nn

from exp.lib.ckpt import (
    clear_resume,
    eval_path,
    load_eval,
    load_resume,
    save_eval,
    save_resume,
)
from exp.lib.host import Host
from mujou import TensorSpec

TAU = pc.TimeConstants(h=4e-3, lam=2e-3, r=4e-3)
FRAME = 1 / 30


def _modules(seed: int) -> dict[str, nn.Module]:
    torch.manual_seed(seed)
    return {"step": nn.Linear(3, 4), "readout": nn.Linear(4, 2)}


class TestEval:
    def test_the_weights_come_back_into_new_modules(self, tmp_path):
        saved = _modules(0)
        path = save_eval(
            tmp_path, program="dynamics", group="g1", t=60.0, frame=1800, modules=saved
        )
        assert path == tmp_path / "ckpt" / "00060.0.pt" == eval_path(tmp_path, 60.0)
        fresh = _modules(1)
        header = load_eval(path, fresh, group="g1")
        assert header == {
            "schema": 1,
            "program": "dynamics",
            "group": "g1",
            "t": 60.0,
            "frame": 1800,
        }
        for name, m in saved.items():
            for k, v in m.state_dict().items():
                assert torch.equal(fresh[name].state_dict()[k], v)

    def test_another_group_raises(self, tmp_path):
        path = save_eval(
            tmp_path, program="p", group="g1", t=1.0, frame=30, modules=_modules(0)
        )
        with pytest.raises(ValueError, match="group 'g1', not 'g2'"):
            load_eval(path, _modules(1), group="g2")

    def test_other_modules_raise(self, tmp_path):
        path = save_eval(
            tmp_path, program="p", group="g", t=1.0, frame=30, modules=_modules(0)
        )
        with pytest.raises(ValueError, match="has modules"):
            load_eval(path, {"step": nn.Linear(3, 4)})


class TestResume:
    @staticmethod
    def _save(root, frame: int, group: str = "g"):
        return save_resume(
            root,
            group=group,
            frame=frame,
            mujou_state={"timer": {"time_ns": frame}},
            host={"frames": frame, "gen": torch.Generator().manual_seed(0).get_state()},
            offsets={"evals.jsonl": 10 * frame},
        )

    def test_the_two_newest_are_kept_and_latest_is_the_newest(self, tmp_path):
        for frame in (30, 60, 90):
            self._save(tmp_path, frame)
        names = sorted(p.name for p in (tmp_path / "resume").glob("*.pt"))
        assert names == ["00000060.pt", "00000090.pt"]
        tree = load_resume(tmp_path, group="g")
        assert tree is not None
        assert tree["frame"] == 90
        assert tree["mujou"] == {"timer": {"time_ns": 90}}
        assert tree["offsets"] == {"evals.jsonl": 900}
        assert torch.equal(
            tree["host"]["gen"], torch.Generator().manual_seed(0).get_state()
        )

    def test_none_without_a_checkpoint(self, tmp_path):
        assert load_resume(tmp_path, group="g") is None

    def test_another_group_raises(self, tmp_path):
        self._save(tmp_path, 30, group="g1")
        with pytest.raises(ValueError, match="not 'g2'"):
            load_resume(tmp_path, group="g2")

    @pytest.mark.parametrize("latest", ["", "00000999.pt\n", None])
    def test_a_broken_latest_falls_back_to_the_newest(self, tmp_path, latest):
        """``latest`` empty (a power loss before its data reached the disk),
        naming no file, or missing: the newest ``*.pt`` by name is read."""
        for frame in (30, 60):
            self._save(tmp_path, frame)
        path = tmp_path / "resume" / "latest"
        if latest is None:
            path.unlink()
        else:
            path.write_text(latest)
        tree = load_resume(tmp_path, group="g")
        assert tree is not None
        assert tree["frame"] == 60

    def test_another_schema_raises(self, tmp_path):
        path = self._save(tmp_path, 30)
        tree = torch.load(path, weights_only=True)
        torch.save({**tree, "schema": 2}, path)
        with pytest.raises(ValueError, match="schema 2"):
            load_resume(tmp_path, group="g")

    def test_the_files_have_the_mode_of_a_plain_open(self, tmp_path):
        path = self._save(tmp_path, 30)
        plain = tmp_path / "plain"
        plain.write_bytes(b"")
        for got in (path, tmp_path / "resume" / "latest"):
            assert got.stat().st_mode == plain.stat().st_mode

    def test_clear_removes_them(self, tmp_path):
        self._save(tmp_path, 30)
        clear_resume(tmp_path)
        assert not (tmp_path / "resume").exists()
        assert load_resume(tmp_path, group="g") is None


def _chain(seed: int) -> pc.Chain:
    torch.manual_seed(seed)
    return pc.Chain(
        nn.Sequential(nn.Linear(2, 3), nn.Tanh()),
        nn.Linear(3, 1),
        x=TensorSpec((2,)),
        y=TensorSpec((1,)),
        loss=pc.loss.MSELoss(),
        tau=TAU,
        optimizer=pc.optim.SGD(tau=0.5),
        learn_interval=None,
    )


def _sample(k: int) -> pc.msg.Sample:
    g = torch.Generator().manual_seed(k)
    x = torch.randn(2, generator=g)
    yes = torch.tensor(True)
    return pc.msg.Sample(x=x, y=x[:1] * x[1:], mask=yes, reset=yes)


def _show(net: pc.Chain, host: Host, frames: range) -> None:
    for k in frames:
        host.show({net.prev: _sample(k)}, net.next)


class TestResumeARun:
    def test_resumed_it_learns_as_a_run_never_stopped(self, tmp_path):
        """A learning chain shown 6 frames, against one stopped after 3,
        checkpointed (the tree of mujou and the host's frames), rebuilt from
        other initial weights, restored and shown the other 3: the weights are
        the same bits."""
        whole = _chain(0)
        with mujou.testing.run(whole) as run:
            _show(whole, Host(run, frame=FRAME, period=TAU.period), range(6))

        first = _chain(0)
        with mujou.testing.run(first) as run:
            host = Host(run, frame=FRAME, period=TAU.period)
            _show(first, host, range(3))
            save_resume(
                tmp_path,
                group="g",
                frame=host.frames,
                mujou_state=run.state_dict(),
                host=host.state_dict(),
                offsets={},
            )
        tree = load_resume(tmp_path, group="g")
        assert tree is not None
        again = _chain(1)
        with mujou.testing.run(again, state=tree["mujou"]) as run:
            host = Host(run, frame=FRAME, period=TAU.period)
            host.load_state_dict(tree["host"])
            _show(again, host, range(3, 6))
        pairs = [
            (whole.layers[0].module, again.layers[0].module),
            (whole.output.module, again.output.module),
        ]
        for a, b in pairs:
            for p, q in zip(a.parameters(), b.parameters(), strict=True):
                assert torch.equal(p, q)
