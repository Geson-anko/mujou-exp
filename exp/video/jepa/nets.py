r"""The visions of M15: ``Chunker -> Enc1 .. EncL -> JEPAHead`` (PC-ALM) or
``Chunker -> JEPABP`` (BP), made from a ``RunConf`` from the same initial
weights (``vit.build_vision``) and the same masks (the ``Chunker`` of the same
seed). The packed layout (M17): ``StackChunker -> Enc1 .. EncL ->
PackedJEPAHead`` (each stage a ``vit.Segmented``) or ``StackChunker ->
JEPABP``."""

from typing import cast

import mujou.pc as pc
import torch

import mujou
from exp.video.dyn.nets import make_optimizer
from exp.video.jepa.bp import JEPABP
from exp.video.jepa.masks import spatial_keep
from exp.video.jepa.nodes import Chunker, JEPAHead, PackedJEPAHead, StackChunker
from exp.video.jepa.vit import (
    PackedPredictor,
    Predictor,
    Segmented,
    Stage,
    VisionModules,
    build_vision,
    image_shape,
)
from exp.video.schema.run import RunConf
from exp.video.schema.vision import VisionConf
from mujou import TensorSpec

__all__ = [
    "CHUNKER_SEED",
    "HEAD_SEED",
    "VisionBP",
    "VisionPC",
    "build_vision_net",
    "vision_modules",
]

# The generators of the nodes, from the run's seed: the masks, and the
# directions of SIGReg (the same for JEPAHead and JEPABP).
CHUNKER_SEED = 1
HEAD_SEED = 2


def _parts(conf: RunConf) -> tuple[VisionConf, pc.TimeConstants, int]:
    if conf.vision is None or conf.optim_vision is None:
        raise ValueError(f"the run {conf.name!r} needs vision and optim_vision")
    t = conf.time
    return conf.vision, pc.TimeConstants(h=t.h, lam=t.lam, r=t.r), conf.data.streams


def _chunker(conf: RunConf, v: VisionConf, tau: pc.TimeConstants, b: int) -> Chunker:
    channels, size = image_shape(conf)
    return Chunker(
        image=TensorSpec((b, channels, size, size)),
        frames=v.frames,
        views=v.views,
        patch=v.patch,
        ratio=v.mask_ratio,
        blocks=v.mask_blocks,
        period=tau.period,
        seed=conf.seed + CHUNKER_SEED,
        carry_frame=v.recon > 0,
        device=conf.device,
    )


def _stack_chunker(
    conf: RunConf, v: VisionConf, tau: pc.TimeConstants, b: int, mods: VisionModules
) -> StackChunker:
    channels, size = image_shape(conf)
    return StackChunker(
        stack=TensorSpec((b, v.frames, channels, size, size), torch.uint8),
        patch=v.patch,
        tubelet=v.tubelet,
        masks=v.context_masks,
        keeps=[spatial_keep(mods.grid, m) for m in v.context_masks],
        period=tau.period,
        seed=conf.seed + CHUNKER_SEED,
        device=conf.device,
    )


def _first(
    conf: RunConf, v: VisionConf, tau: pc.TimeConstants, b: int, mods: VisionModules
) -> Chunker | StackChunker:
    if v.layout == "packed":
        return _stack_chunker(conf, v, tau, b, mods)
    return _chunker(conf, v, tau, b)


class VisionPC(mujou.Network):
    """``Chunker -> Enc1 .. EncL -> JEPAHead`` of ``mods``: one ``pc.Layer`` per
    stage of the encoder (``enc1``, ``enc2``, ..., in ``names``), the head on
    the predictor. The host writes a ``pc.msg.Sample`` (``x`` ``(B, C, H,
    W)``, ``y`` zeros ``(B, 1)``, ``mask`` and ``reset`` ``(B,)``) to ``prev``
    and reads ``stats``; ``z_out`` is the head's (send-only).

    Args:
        mods: The modules, on ``conf.device``.
        conf: The run: its vision, time, streams, seed and device.
    """

    enc1: pc.Layer  # set by name: those of vision.nodes only
    enc2: pc.Layer

    def __init__(self, mods: VisionModules, *, conf: RunConf) -> None:
        super().__init__()
        v, tau, b = _parts(conf)
        assert conf.optim_vision is not None  # _parts checks
        optimizer = make_optimizer(conf.optim_vision, conf.time.frame)
        self.mods, self.tau = mods, tau
        packed = v.layout == "packed"
        self.chunker = _first(conf, v, tau, b, mods)
        spec, port = self.chunker.act_spec, self.chunker.next
        self.names = tuple(f"enc{i + 1}" for i in range(len(mods.encoder.stages)))
        for name, stage in zip(self.names, mods.encoder.stages, strict=True):
            node = pc.Layer(
                Segmented(cast(Stage, stage), mods.segments) if packed else stage,
                act_spec=spec,
                tau=tau,
                optimizer=optimizer,
                learn_interval=conf.time.learn_interval,
                device=conf.device,
                cuda_graph=conf.ctl.cuda_graph and conf.device.startswith("cuda"),
            )
            setattr(self, name, node)
            port.connect(node.prev)
            spec, port = node.act_spec, node.next
        self.head: JEPAHead | PackedJEPAHead
        if packed:
            assert isinstance(mods.predictor, PackedPredictor)
            self.head = PackedJEPAHead(
                mods.predictor,
                proj=mods.proj,
                act_spec=spec,
                keeps=list(mods.segments[1:]),
                lam=v.sigreg,
                sg=v.sg,
                sigreg_on=v.sigreg_on,
                pred_loss=v.pred_loss,
                token_sigreg=v.token_sigreg,
                directions=v.sigreg_directions,
                tau=tau,
                optimizer=optimizer,
                learn_interval=conf.time.learn_interval,
                seed=conf.seed + HEAD_SEED,
                device=conf.device,
                cuda_graph=conf.ctl.cuda_graph,
            )
        else:
            assert isinstance(mods.predictor, Predictor)
            self.head = JEPAHead(
                mods.predictor,
                act_spec=spec,
                views=v.views,
                extra=mods.encoder.extra,
                z_pool=v.z_pool,
                lam=v.sigreg,
                sg=v.sg,
                sigreg_on=v.sigreg_on,
                decoder=mods.decoder,
                recon=v.recon,
                pixel_mean=v.pixel_mean,
                pixel_std=v.pixel_std,
                directions=v.sigreg_directions,
                tau=tau,
                optimizer=optimizer,
                learn_interval=conf.time.learn_interval,
                seed=conf.seed + HEAD_SEED,
                device=conf.device,
                cuda_graph=conf.ctl.cuda_graph,
            )
        port.connect(self.head.prev)
        self.prev, self.stats, self.z_out = (
            self.chunker.prev,
            self.head.stats,
            self.head.z_out,
        )


class VisionBP(mujou.Network):
    """``Chunker -> JEPABP`` of ``mods``, with the ports of ``VisionPC``."""

    def __init__(self, mods: VisionModules, *, conf: RunConf) -> None:
        super().__init__()
        v, tau, b = _parts(conf)
        assert conf.optim_vision is not None  # _parts checks
        self.mods, self.tau = mods, tau
        self.chunker = _first(conf, v, tau, b, mods)
        self.vision = JEPABP(
            mods,
            act_spec=self.chunker.act_spec,
            views=v.views,
            lam=v.sigreg,
            sg=v.sg,
            sigreg_on=v.sigreg_on,
            recon=v.recon,
            pixel_mean=v.pixel_mean,
            pixel_std=v.pixel_std,
            pred_loss=v.pred_loss,
            token_sigreg=v.token_sigreg,
            directions=v.sigreg_directions,
            tau=tau,
            optimizer=make_optimizer(conf.optim_vision, conf.time.frame),
            learn_interval=conf.time.learn_interval,
            seed=conf.seed + HEAD_SEED,
            device=conf.device,
        )
        self.chunker.next.connect(self.vision.prev)
        self.prev, self.stats, self.z_out = (
            self.chunker.prev,
            self.vision.stats,
            self.vision.z_out,
        )


def build_vision_net(conf: RunConf) -> VisionPC | VisionBP:
    """The vision of ``conf``: the modules of ``build_vision`` (seed
    ``conf.seed``) in the nodes of ``conf.learner`` (``"pc"`` or ``"bp"``),
    with the optimizer of ``conf.optim_vision``.

    Raises:
        ValueError: ``conf`` has no vision or no ``optim_vision``, or its
            learner is ``"none"``.
    """
    _parts(conf)
    kinds = {"pc": VisionPC, "bp": VisionBP}
    if conf.learner.kind not in kinds:
        raise ValueError(
            f"learner {conf.learner.kind!r} learns no vision; use 'pc' or 'bp'"
        )
    mods = build_vision(conf, seed=conf.seed)
    return kinds[conf.learner.kind](mods, conf=conf)


def vision_modules(net: VisionPC | VisionBP) -> VisionModules:
    """The live modules of ``net`` (no copy): read them between the steps of its
    run."""
    return net.mods
