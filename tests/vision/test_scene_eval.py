r"""``exp.vision.scene_eval`` (V18c's main measure of the scene categories).

The oracles: the groups kept at the rules' edges counted by hand (a share of
5 % exactly, 5 videos exactly); a Markov chain that keeps its label with
probability $\phi$ (else draws it anew) has the centred one-hot's
autocorrelation $\phi^\ell$, so $\tau = (1 + \phi) / (1 - \phi)$; the
support and the audit's draws by video by hand; ``mean`` scores exactly 0
and BA $1/K$, ``label`` BA 1 and $m = 1 - B$, the videos' one-hot $m$ 0
(``score``); the synthetic codes' third $q$ is $0.25 + \Delta_* / (1 -
B)$; in E1f, ``pred_k3`` (the pooled outputs copied) is ``pool``'s fit,
``mean`` is 0, a predictor of the hidden state in two positions beats their
mean. End to end on the synthetic Walking Tours: the labels by a tiny
stand-in for Places365, the audit's sheets, the yardsticks, the checks, a
V18c-like run's rows and E1f.
"""

import json
from pathlib import Path
from typing import override

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from exp.lib.readout import video_chance
from exp.vision.scene_eval import (
    AUDIT_N,
    DELTA_STAR,
    E1F_STAGES,
    OUT,
    OUT_FILL,
    YARD_PREFIX,
    SceneConf,
    SceneLabels,
    _analysis,
    audit_sample,
    block_length,
    e1f_scene_rows,
    evaluate_audit,
    evaluate_checks,
    evaluate_fill,
    evaluate_frames,
    evaluate_labels,
    evaluate_run,
    evaluate_yardsticks,
    kept_groups,
    random_labels,
    score,
    support,
    synthetic_codes,
)
from tests.vision.test_fill_eval import features
from tests.vision.test_jepa_eval import SPAN, conf_of, tiny_run
from tests.vision.test_sem_eval import V18C, hires_of
from tests.vision.test_stream_eval import EVERY

TINY = SceneConf(
    videos=("A", "B", "C", "H"),
    stride=1,
    keep_share=1e-9,  # any group seen
    keep_videos=1,
    keep_video_share=1e-9,
    min_blocks=1,
    l_min=1,
    max_group=1.0,
    max_dropped=1.0,
    batch=8,
)


class TestKeptGroups:
    def test_the_rules_edges_by_hand(self):
        # 10 videos of 100 labels: group 1 is 5 % of all, in 5 videos at 1 %
        # or more (kept); group 2 is 5 % but in 4 videos (not); group 3 is in
        # 10 videos but 4 % (not); group 0 the rest (kept)
        sconf = SceneConf(videos=tuple("abcdefghij"))
        y = torch.zeros(10, 100, dtype=torch.long)
        y[:5, :10] = 1  # 50 of 1000
        y[5:9, 10:22] = 2  # 48 ...
        y[9, 10:12] = 2  # ... + 2 = 50, but video j holds 2 % too: 5 videos
        y[:, 30:34] = 3  # 40
        video = torch.arange(10)[:, None].expand(10, 100)
        kept, record = kept_groups(y.flatten(), video.flatten(), 4, sconf)
        assert kept == [0, 1, 2]
        y[9, 10:12] = 0  # group 2: 48 of 1000, 4 videos
        y[0, 50:52] = 2  # 50, the share back, but video a at 2 %: 5 videos
        kept, record = kept_groups(y.flatten(), video.flatten(), 4, sconf)
        assert kept == [0, 1, 2] and record["videos_over"][2] == 5
        y[0, 50:52] = 0
        kept, _ = kept_groups(y.flatten(), video.flatten(), 4, sconf)
        assert kept == [0, 1]

    def test_v_c1s_largest_group_is_of_the_kept_stacks(self):
        # 2 videos of 100: group 0 60 %, 1 30 %, 2 10 % (left out at 15 %);
        # of the kept 90 stacks per video, group 0 holds 60/90
        sconf = SceneConf(videos=("a", "b"), keep_share=0.15, keep_videos=2,
                          min_blocks=1, l_min=1)  # fmt: skip
        y = torch.zeros(2, 100, dtype=torch.long)
        y[:, 60:90] = 1
        y[:, 90:] = 2
        video = torch.arange(2)[:, None].expand(2, 100).flatten()
        got = _analysis(y.flatten(), video, 3, sconf, None)
        assert got["kept"] == [0, 1]
        assert got["checks"]["max_group"] == pytest.approx(60 / 90)
        assert got["checks"]["max_group_of_all"] == pytest.approx(0.6)
        assert got["checks"]["dropped"] == pytest.approx(0.1)


class TestBlockLength:
    def test_a_markov_chain_of_keep_phi_has_its_integrated_time(self):
        phi, k, n = 0.8, 4, 200_000
        g = torch.Generator().manual_seed(0)
        keep = torch.rand(n, generator=g) < phi
        fresh = torch.randint(k, (n,), generator=g)
        t = torch.empty(n, dtype=torch.long)
        t[0] = fresh[0]
        for i in range(1, n):  # a plain loop: the chain is sequential
            t[i] = t[i - 1] if keep[i] else fresh[i]
        got = block_length(t, torch.zeros(n, dtype=torch.long), k, l_min=1)
        assert got["tau"] == pytest.approx((1 + phi) / (1 - phi), rel=0.05)

    def test_l_is_cut_to_the_shortest_video(self):
        t = torch.zeros(100, dtype=torch.long)
        t[50:] = 1
        video = torch.cat([torch.zeros(80), torch.ones(20)]).long()
        got = block_length(t, video, 2, l_min=16)
        assert got["L_max"] == 20 // 8 and got["L"] == 2 and got["L_at_max"]

    def test_the_support_by_hand(self):
        t = torch.tensor([0, 0, 1, -1, 1, 0, 2, 2])
        blocks = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3])
        assert support(t, blocks, 3) == [2, 2, 2]


class TestAuditSample:
    def test_draws_in_proportion_to_the_videos(self):
        # class 0: 60 in video 0, 30 in 1, 10 in 2; 10 drawn: 6, 3, 1
        t = torch.zeros(100, dtype=torch.long)
        video = torch.cat(
            [torch.zeros(60), torch.ones(30), torch.full((10,), 2)]
        ).long()
        rows = audit_sample(t, video, 1, 10, torch.Generator().manual_seed(0))[0]
        assert torch.bincount(video[rows], minlength=3).tolist() == [6, 3, 1]
        assert len(set(rows)) == 10

    def test_the_remainders_go_to_the_largest_fractions(self):
        # 7 of 50 / 30 / 20: 4.2, 2.1, 1.4 -> 4, 2, 1
        t = torch.zeros(100, dtype=torch.long)
        video = torch.cat(
            [torch.zeros(50), torch.ones(30), torch.full((20,), 2)]
        ).long()
        rows = audit_sample(t, video, 1, 7, torch.Generator().manual_seed(1))[0]
        assert torch.bincount(video[rows], minlength=3).tolist() == [4, 2, 1]


def labelled(n: int = 240, k: int = 3, videos: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    t = torch.randint(k, (n,), generator=g)
    t[::7] = -1  # stacks left out
    video = torch.arange(n) * videos // n
    return t, video, g


class TestScore:
    def test_mean_label_and_video(self):
        t, video, _ = labelled()
        on = t >= 0
        mean, chosen = score(None, t, video, 3, held_out=3)
        assert mean["m"] == 0.0 and mean["ba"] == pytest.approx(1 / 3, abs=1e-15)
        assert chosen is None and mean["pred"][0] == -1
        label, _ = score(F.one_hot(t.clamp(min=0), 3).double(), t, video, 3, 3)
        assert label["ba"] == 1.0
        b = video_chance(t[on], t[on], video[on], 3)
        assert label["m"] == pytest.approx(1 - b, abs=1e-12)
        vid, _ = score(F.one_hot(video, 4).double(), t, video, 3, 3, jack=True)
        assert abs(vid["m"]) <= 1e-12
        jack = torch.tensor(vid["jack_pred"])
        assert jack.shape == (4, len(t)) and bool((jack[:, ~on] == -1).all())

    def test_random_labels_keep_the_left_out_and_the_shares(self):
        t, _, _ = labelled(3000)
        draws = random_labels(t, 3, 2, seed=0)
        on = t >= 0
        for r in draws:
            assert bool((r[~on] == -1).all()) and bool((r[on] >= 0).all())
            share = torch.bincount(r[on], minlength=3).double() / int(on.sum())
            want = torch.bincount(t[on], minlength=3).double() / int(on.sum())
            assert torch.allclose(share, want, atol=0.04)

    def test_the_synthetic_third_q_is_one_delta_star_apart(self):
        t, video, _ = labelled()
        on = t >= 0
        codes = synthetic_codes(t, video, 3, dim=16)
        qs = sorted(q for q, _, _ in codes.values())
        b = video_chance(t[on], t[on], video[on], 3)
        assert qs[2] == pytest.approx(0.25 + DELTA_STAR / (1 - b))
        for _, s, code in codes.values():
            assert code.shape == (len(t), 16)
            assert torch.equal(code[on, :3].argmax(1), s[on])
        zero = next(s for q, s, _ in codes.values() if q == 0.0)
        assert not torch.equal(zero[on], t[on])


def _labels_of(t: torch.Tensor, video: torch.Tensor, k: int) -> SceneLabels:
    meta = {
        "video": video.tolist(),
        "videos": ["a", "b", "c", "d"],
        "held_out": "d",
    }
    for q in ("tl", "tr", "bl", "br"):
        meta[f"quad/{q}"] = {"t": t.tolist(), "K": k}
    return SceneLabels({}, meta)


class TestE1f:
    def test_the_stages_of_pooled_and_position_codes(self):
        # the label is the sign of u_1 - u_2 of two positions' states; pred
        # holds both, pool their mean (none of the difference)
        g = torch.Generator().manual_seed(3)
        n = 400
        video = torch.arange(n) * 4 // n
        u = torch.randn(n, 2, 2, generator=g, dtype=torch.float64)  # (n, P, d)
        t = ((u[:, 0, 0] - u[:, 1, 0]) > 0).long()
        pool = u.mean(1)
        noise = torch.randn(n, 6, generator=g, dtype=torch.float64)
        stage = {
            "pred": u.flatten(1),
            "pool": pool,
            "pred_k3": pool[:, None].expand(-1, 2, -1).flatten(1),
            "vis_color": noise,
            "oracle": u.flatten(1),
        }
        stages = {(q, k): stage for q in ("tl", "tr", "bl", "br") for k in (4, 8, 12)}
        rows = e1f_scene_rows(stages, _labels_of(t, video, 2), k_run=8)
        assert len(rows) == 3 * len(E1F_STAGES)
        by = {(r["k"], r["stage"]): r for r in rows}
        for k in (4, 8, 12):
            assert by[(k, "mean")]["m"] == 0.0
            assert abs(by[(k, "pred_k3")]["m"] - by[(k, "pool")]["m"]) <= 1e-6
            assert by[(k, "pred")]["m"] > by[(k, "pool")]["m"] + 0.3
        assert "jack_pred" in by[(8, "pred")]["quarters"]["tl"]
        assert set(by[(8, "pool")]["quarters"]["br"]) >= {"cond", "m", "pred"}


# -- end to end -------------------------------------------------------------------------


class TinyPlaces(nn.Module):
    """A stand-in for Places365's ResNet-50: the 4 x 4 means through a fixed
    random layer to 365 logits (its last child the last layer)."""

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.flat = nn.Flatten()
        g = torch.Generator().manual_seed(0)
        self.fc = nn.Linear(48, 365)
        with torch.no_grad():
            self.fc.weight.copy_(20 * torch.randn(365, 48, generator=g))
            self.fc.bias.zero_()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.flat(self.pool(x)))


@pytest.fixture
def root(tmp_path) -> Path:
    from tests.vision.test_eval import random_tours

    return random_tours(tmp_path / "d")


@pytest.mark.slow
class TestEvaluate:
    def test_labels_audit_yardsticks_checks_a_run_and_e1f(self, root, tmp_path):
        conf = conf_of()
        lists = json.loads(
            evaluate_frames(
                tmp_path / "frames.json", conf=conf, sconf=TINY, root=root
            ).read_text()
        )
        assert set(lists) == {"A", "B", "C", "H"}
        hires = hires_of(root, lists, tmp_path / "hires")
        cache = tmp_path / "cache"
        kw = {"conf": conf, "sconf": TINY, "root": root}
        path = evaluate_labels(hires, Path("unused"), cache, net=TinyPlaces(), **kw)
        assert path.name.startswith("v18c-scene-")
        lab = SceneLabels.load(path)
        n = len(lab.meta["places"])
        assert lab.probs["full"].shape == (n, 16)
        torch.testing.assert_close(
            lab.probs["full"].sum(1), torch.ones(n), rtol=0, atol=1e-5
        )
        assert lab.meta["places_max_diff"] == 0.0
        assert lab.probs["places64"].shape == (n, 48)
        full = lab.meta["full"]
        assert full["K"] >= 2 and len(full["t"]) == n and full["L"] >= 1
        assert set(lab.meta["record"]["nuis"]) == {"bright", "white", "blur"}
        k = full["K"]
        # the audit's sheets
        audit = evaluate_audit(tmp_path / "audit", hires, cache=cache, **kw)
        key = json.loads((audit / "key.json").read_text())
        counts = torch.bincount(lab.t()[lab.t() >= 0], minlength=k)
        full_items = [it for it in key if it["quarter"] is None]
        assert len(full_items) == int(counts.clamp(max=AUDIT_N).sum())
        # the judge's directories hold neither the key nor the teacher's views
        assert (audit / key[0]["sheet"]).is_file() and (
            audit / key[0]["sheet64"]
        ).is_file()
        judged = {p.name for d in ("judge", "judge64") for p in (audit / d).iterdir()}
        assert "key.json" not in judged and not any(
            n.startswith("view") for n in judged
        )
        groups = json.loads((audit / "judge" / "groups.json").read_text())
        assert {it["teacher_id"] for it in key} <= {g["id"] for g in groups}
        # the yardsticks: mean 0, label 1 - B, the videos 0
        run = tiny_run(root, tmp_path, *V18C)
        yard = evaluate_yardsticks(
            tmp_path / "y",
            codes=("mean", "label", "video", "color", "pca16", "init_c", "places64"),
            init_runs=(run,), cache=cache, spacing=EVERY, span=SPAN, pca_max=8,
            **kw,
        )  # fmt: skip
        rows = {
            p.stem: json.loads(p.read_text()) for p in sorted((yard / "rows").iterdir())
        }
        mean, label = rows[f"{YARD_PREFIX}-mean"], rows[f"{YARD_PREFIX}-label"]
        assert mean["m"] == 0.0 and mean["ba"] == pytest.approx(1 / k)
        assert label["ba"] == 1.0
        assert label["m"] == pytest.approx(1 - label["chance"], abs=1e-12)
        assert abs(rows[f"{YARD_PREFIX}-video"]["m"]) <= 1e-12
        init = rows[f"{YARD_PREFIX}-init_c-run"]
        assert init["dim"] == 1024 and "jack_pred" in init
        assert rows[f"{YARD_PREFIX}-pca16"]["dim"] == 8  # of pca_max 8
        # the checks
        checks = evaluate_checks(
            tmp_path / "c", cache=cache, features=features, draws=2, **kw
        )
        names = sorted(p.stem for p in (checks / "rows").iterdir())
        assert sum(nm.startswith("syn_") for nm in names) == 4
        assert {"rand-color-01", "rand-inc64-00"} <= set(names)
        # a null row keeps its predictions and labels for its own bootstrap
        rand = json.loads((checks / "rows" / "rand-color-01.json").read_text())
        assert len(rand["pred"]) == n and len(rand["t"]) == n
        syn0 = json.loads((checks / "rows" / "syn_0.0000.json").read_text())
        assert syn0["q"] == 0.0 and "m_true" in syn0
        # a run: its step 0 is the yardstick init_c
        out = evaluate_run(run, steps=(0, 4), root=root, cache=cache, sconf=TINY)
        assert out == run / OUT
        got = sorted(p.stem for p in (out / "rows").iterdir())
        assert got == ["000000-h_mean", "000000-z", "000000-z_k3", "000004-z"]
        step0 = json.loads((out / "rows" / "000000-z.json").read_text())
        assert step0["m"] == init["m"] and step0["pred"] == init["pred"]
        with pytest.raises(ValueError, match="no checkpoint"):
            evaluate_run(run, steps=(3,), root=root, cache=cache, sconf=TINY)
        # E1f of the scene categories
        fill = evaluate_fill(run, steps=(4,), root=root, cache=cache, sconf=TINY)
        assert fill == run / OUT_FILL
        e1f = json.loads((fill / "rows" / "000004-e1f.json").read_text())["rows"]
        by = {(r["k"], r["stage"]): r for r in e1f}
        for kk in (4, 8, 12):
            assert by[(kk, "mean")]["m"] == 0.0
            assert abs(by[(kk, "pred_k3")]["m"] - by[(kk, "pool")]["m"]) <= 1e-6
