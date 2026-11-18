"""The settings of the optimizer of the nodes."""

from dataclasses import dataclass

from exp.vision.schema._check import check_choice

__all__ = ["OPTIMIZERS", "OptimConf"]

OPTIMIZERS = ("sgd", "sgd_momentum", "rmsprop", "adamw")
# The time constants and settings of a kind: their defaults and the kinds that
# use them (another kind must leave them, as a sweep of them would change nothing).
_USED = {
    "tau_m": (1.0, ("sgd_momentum", "adamw")),
    "tau_v": (30.0, ("rmsprop", "adamw")),
    "eps": (1e-8, ("rmsprop", "adamw")),
    "weight_decay": (0.0, ("adamw",)),
}


@dataclass(kw_only=True)
class OptimConf:
    """A ``mujou.pc.optim`` optimizer, the same for every node: ``"sgd"``
    (``SGD(tau)``), ``"sgd_momentum"`` (``SGD(tau, tau_m)``), ``"rmsprop"``
    (``RMSProp(tau, tau_v, eps)``) or ``"adamw"``. All time constants are in
    seconds of virtual time; ``tau`` is that of the parameters (the learning
    rate is the time per ``tau``), ``tau_m`` that of the first moment, ``tau_v``
    that of the second."""

    kind: str = "sgd"
    tau: float = 1e5  # conf/optim/sgd.yaml's: the stable power of 10 on v0/base
    tau_m: float = 1.0
    tau_v: float = 30.0
    eps: float = 1e-8
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        check_choice("optim.kind", self.kind, OPTIMIZERS)
        if not self.tau > 0 or not self.tau_m > 0 or not self.tau_v > 0:
            raise ValueError("optim.tau, tau_m and tau_v must be > 0")
        for name, (default, kinds) in _USED.items():
            if self.kind not in kinds and getattr(self, name) != default:
                raise ValueError(
                    f"optim.{name} is of the kinds {kinds}, not {self.kind!r}: "
                    f"leave it at {default}"
                )
