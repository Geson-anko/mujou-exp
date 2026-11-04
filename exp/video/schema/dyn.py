"""The settings of the dynamics (M14, M16)."""

from dataclasses import dataclass

from exp.video.schema._check import check_choice
from exp.video.schema.learn import CREDITS, FEEDBACKS

__all__ = ["DynamicsConf", "WindowConf"]


@dataclass(kw_only=True)
class WindowConf:
    """The learning on windows (M16): instead of learning from every frame,
    the dynamics learns every ``stride`` frames from the last ``window`` frames
    of each stream, by the gradient of ``exp.video.dyn.offline.window_loss``
    (from s = 0, ``observed`` frames teacher forced, the rest from its own
    prediction; ``feedback`` and ``credit`` as there) at the first update
    ``relax`` seconds after the window's last frame. The prediction of every
    frame goes on as without it; only the learning changes. BP learns it in
    one node (``DynamicsBP``), PC-ALM by an episode of each window over the
    nodes (``DynamicsBlock``, with ``credit="bptt"`` only), whose messages
    settle (L + 1) K + L updates after the one that opens the window (K =
    window - 1 - observed, L = nodes; K + L with "detach", L with
    "teacher"): ``relax`` must leave room for them (80 periods = 0.16 s of
    the 0.3 s for the default window and 3 nodes). The window sets when the
    dynamics learns: ``time.learn_interval`` is not read (``math.inf`` does
    not freeze it), and ``relax`` must be shorter than ``stride`` frames
    (``RunConf``), so a smaller ``stride`` may need a smaller ``relax`` (0.3
    s is exactly 9 frames at 30 fps). The ring does not read the received
    ``mask``: a window is left out only for a reset or a gap in time
    inside it (``exp.lib.ring.WindowRing``).
    """

    window: int = 36  # W, frames per window
    observed: int = 16  # O < W, frames teacher forced from the start
    stride: int = 18  # S, frames between two windows
    feedback: str = "detach"  # "detach" | "teacher" | "grad"
    credit: str = "bptt"  # "bptt" | "own" (an ablation)
    relax: float = 0.3  # seconds from the window's last frame to the learning

    def __post_init__(self) -> None:
        check_choice("dynamics.window.feedback", self.feedback, FEEDBACKS)
        check_choice("dynamics.window.credit", self.credit, CREDITS)
        if self.window < 2:
            raise ValueError(f"dynamics.window.window must be >= 2, not {self.window}")
        if not 1 <= self.observed < self.window:
            raise ValueError(
                "dynamics.window.observed must be in [1, window), not "
                f"{self.observed} (window {self.window})"
            )
        if self.stride < 1:
            raise ValueError(f"dynamics.window.stride must be >= 1, not {self.stride}")
        if not self.relax >= 0:
            raise ValueError(f"dynamics.window.relax must be >= 0, not {self.relax}")
        if self.credit == "own" and self.feedback == "grad":
            raise ValueError(
                "dynamics.window.credit='own' detaches the feedback: "
                "set dynamics.window.feedback='detach'"
            )


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
    # The learning on windows instead of every frame (None: every frame). Fill
    # it from a preset or with `+window@dynamics.window=m16` (conf/window/).
    window: WindowConf | None = None

    def __post_init__(self) -> None:
        if self.nodes < 1:
            raise ValueError(f"nodes must be at least 1, not {self.nodes}")
        if self.window is not None and self.trace:
            raise ValueError(
                "dynamics.window learns through time on its own: "
                "set dynamics.trace=false"
            )
