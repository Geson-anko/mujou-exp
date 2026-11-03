r"""The objective of the sweeps of M14 (design 5.2), the one place it is
computed: the run writes it in each validation row of ``evals.jsonl`` and in
``result.json``, and optuna takes it as it is.

For one run (one seed),

$$
J = \frac12 \sum_{h \in \{5, 10\}} \log \frac{E^{\mathrm{val}}_h}{\mathrm{CV}^{\mathrm{val}}_h},
$$

the log of the error of the rollout relative to constant velocity, averaged
over 5 and 10 frames ahead (minimized; 0 is constant velocity, $\log 0.5 =
-0.69$ half its error). With ``reference="ca"`` (stage 3: on the pixels the
error of a position that is off is capped, constant acceleration is the
stronger bar there and a blurred prediction is not mistaken for progress) the
denominator is $\mathrm{CA}$ in place of $\mathrm{CV}$. A run that diverged, or whose value is not finite,
takes the penalty $\log 10$ (ten times constant velocity), and so does any
worse value: no run that went on ranks below one that diverged. An error of 0
(a perfect rollout) counts as ``FLOOR``, so it ranks first rather than taking
the penalty.
"""

import math
from collections.abc import Mapping

__all__ = ["FLOOR", "HORIZONS", "PENALTY", "objective"]

HORIZONS = (5, 10)
PENALTY = math.log(10.0)
FLOOR = 1e-12  # the least error of a rollout: log(FLOOR / cv) is finite


def objective(
    row: Mapping[str, float | None],
    *,
    diverged: bool = False,
    reference: str = "cv",
) -> float:
    """$J$ of the module docstring from the ``roll{h}`` and ``{reference}{h}``
    of ``row`` (an evaluation of the validation clips), ``PENALTY`` when
    ``diverged``, when a value is missing (``None``) or not finite, or a
    ``cv{h}`` is not > 0, and at most ``PENALTY``; a ``roll{h}`` below
    ``FLOOR`` counts as ``FLOOR``.

    Raises:
        KeyError: ``row`` has no ``roll{h}`` or ``{reference}{h}`` for an ``h`` of
            ``HORIZONS`` (the run does not measure them).
    """
    if diverged:
        return PENALTY
    terms = []
    for h in HORIZONS:
        roll, cv = row[f"roll{h}"], row[f"{reference}{h}"]
        if roll is None or cv is None or not (cv > 0 and roll >= 0):
            return PENALTY  # also NaN, which fails every comparison
        if not (math.isfinite(roll) and math.isfinite(cv)):
            return PENALTY
        terms.append(math.log(max(roll, FLOOR) / cv))
    j = sum(terms) / len(terms)
    return min(j, PENALTY) if math.isfinite(j) else PENALTY
