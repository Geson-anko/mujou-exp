"""The settings of the dynamics (M14, M16)."""

from dataclasses import dataclass

__all__ = ["DynamicsConf"]


@dataclass(kw_only=True)
class DynamicsConf:
    """The dynamics: a ``Dynamics`` node (minGRU) and ``nodes - 1``
    ``DynamicsLayer`` nodes on top, read out by a linear map to the next
    observation. The defaults are the small dynamics of M13 (``d_s`` 256, one
    node); the big one is ``d_s`` 512, three nodes."""

    d_s: int = 256  # the width of the state s_k
    nodes: int = 1  # Dynamics and nodes - 1 DynamicsLayer
    # Whether each node keeps the eligibility traces of its minGRU
    # (exp.lib.mingru.MinGRUTrace): the gradient through its own state with the
    # inputs it remembered held; without, the gradient stops at s_{k-1}.
    trace: bool = False
    # Whether the prediction is x_{k-1} + W_o s_k (the change from the last
    # observation) rather than W_o s_k; W_o starts at 0.
    residual: bool = False
    # The fraction of the predictions, the last ones of every open_cycle
    # frames, made from the model's own prediction instead of the observation.
    open_loop: float = 0.5
    open_cycle: int = 20

    def __post_init__(self) -> None:
        if self.nodes < 1:
            raise ValueError(f"nodes must be at least 1, not {self.nodes}")
