"""The settings of the optimizer of the nodes."""

from dataclasses import dataclass

from exp.vision.schema._check import check_choice

__all__ = ["OPTIMIZERS", "OptimConf"]

OPTIMIZERS = (
    "sgd",
    "sgd_momentum",
    "rmsprop",
    "adamw",
    "nesterov",
    "rmsprop_centered",
    "adadelta",
    "rprop",
    "muon",
)
# The time constants and settings of a kind: their defaults and the kinds that
# use them (another kind must leave them, as a sweep of them would change nothing).
_USED = {
    "tau_m": (1.0, ("sgd_momentum", "adamw", "nesterov", "muon")),
    "tau_v": (30.0, ("rmsprop", "adamw", "rmsprop_centered", "adadelta", "muon")),
    "eps": (1e-8, ("rmsprop", "adamw", "rmsprop_centered", "adadelta", "muon")),
    "weight_decay": (0.0, ("adamw", "muon")),
}


@dataclass(kw_only=True)
class OptimConf:
    """The optimizer of the nodes: ``"sgd"`` (``mujou.pc.optim.SGD(tau)``),
    ``"sgd_momentum"`` (``SGD(tau, tau_m)``), ``"rmsprop"`` (``RMSProp(tau,
    tau_v, eps)``) or ``"adamw"`` of ``mujou.pc.optim``, or one of
    ``exp.lib.optim_zoo``: ``"nesterov"`` (``Nesterov(tau, tau_m)``),
    ``"rmsprop_centered"`` (``RMSPropCentered(tau, tau_v, eps)``),
    ``"adadelta"`` (``Adadelta(tau, tau_v, eps)``), ``"rprop"``
    (``Rprop(tau)``) or ``"muon"`` (``Muon(tau, tau_m, tau_v, eps,
    weight_decay)``). All time constants are in seconds of virtual time;
    ``tau`` is that of the parameters (the learning rate is the time per
    ``tau``), ``tau_m`` that of the first moment, ``tau_v`` that of the
    second. The decoder's nodes (``dec1``, ``dec2``, ``out``) learn with
    ``tau * dec_scale``, the encoder's with ``tau`` (``exp.vision.pc.
    make_optimizers``)."""

    kind: str = "sgd"
    tau: float = 1e5  # conf/optim/sgd.yaml's: the stable power of 10 on v0/base
    tau_m: float = 1.0
    tau_v: float = 30.0
    eps: float = 1e-8
    weight_decay: float = 0.0
    dec_scale: float = 1.0  # of every kind

    def __post_init__(self) -> None:
        check_choice("optim.kind", self.kind, OPTIMIZERS)
        if not self.tau > 0 or not self.tau_m > 0 or not self.tau_v > 0:
            raise ValueError("optim.tau, tau_m and tau_v must be > 0")
        if not 0 < self.dec_scale < float("inf"):
            raise ValueError(f"optim.dec_scale must be > 0, not {self.dec_scale}")
        for name, (default, kinds) in _USED.items():
            if self.kind not in kinds and getattr(self, name) != default:
                raise ValueError(
                    f"optim.{name} is of the kinds {kinds}, not {self.kind!r}: "
                    f"leave it at {default}"
                )
