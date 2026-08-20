"""Causal walk-forward / purged split helpers for sequential RL.

Stage 0 default validation method. CPCV is intentionally not implemented here
— combinatorial folds are only safe for HRL if temporal causality is proven;
prefer walk-forward + purge + embargo until then.

Fold geometry (expanding)::

    |---- train ----| purge | embargo |---- test ----|
         [ts, te)      [te, pe)  [pe, ee)    [ee, xe)

``te <= pe <= ee <= xe`` and ``test`` never overlaps ``train``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FoldSplit:
    """One causal train / purge / embargo / test fold on a bar index axis."""

    fold: int
    train_start: int
    train_end: int
    purge_end: int
    embargo_end: int
    test_start: int
    test_end: int

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def train_bars(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_bars(self) -> int:
        return self.test_end - self.test_start

    def assert_causal(self) -> None:
        if not (0 <= self.train_start < self.train_end):
            raise ValueError(f"invalid train range: {self}")
        if not (self.train_end <= self.purge_end <= self.embargo_end):
            raise ValueError(f"purge/embargo must follow train: {self}")
        if self.test_start != self.embargo_end:
            raise ValueError(f"test must start at embargo_end: {self}")
        if not (self.test_start < self.test_end):
            raise ValueError(f"invalid test range: {self}")
        if self.train_end > self.test_start:
            raise ValueError(f"train overlaps test (leakage): {self}")


def walk_forward_splits(
    n_bars: int,
    *,
    n_folds: int,
    train_bars: int,
    test_bars: int,
    purge_bars: int = 0,
    embargo_bars: int = 0,
    mode: str = "expanding",
) -> list[FoldSplit]:
    """Build ``n_folds`` causal walk-forward splits over ``[0, n_bars)``.

    Parameters
    ----------
    n_bars:
        Total low-TF bars available on the path.
    n_folds:
        Number of OOS folds.
    train_bars:
        Minimum (expanding) or fixed (rolling) train length for fold 0.
    test_bars:
        OOS length per fold.
    purge_bars / embargo_bars:
        Dead zone between train end and test start.
    mode:
        ``"expanding"`` grows the train window; ``"rolling"`` keeps train length
        fixed and slides it forward.
    """
    n_bars = int(n_bars)
    n_folds = int(n_folds)
    train_bars = int(train_bars)
    test_bars = int(test_bars)
    purge_bars = max(0, int(purge_bars))
    embargo_bars = max(0, int(embargo_bars))
    mode = str(mode).lower().strip()
    if mode not in ("expanding", "rolling"):
        raise ValueError(f"unknown walk-forward mode: {mode!r}")
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if train_bars < 2 or test_bars < 2:
        raise ValueError("train_bars and test_bars must be >= 2")

    gap = purge_bars + embargo_bars
    first_test_start = train_bars + gap
    last_test_end = first_test_start + n_folds * test_bars
    if last_test_end > n_bars:
        raise ValueError(
            f"n_bars={n_bars} too short for n_folds={n_folds}, train_bars={train_bars}, "
            f"test_bars={test_bars}, purge={purge_bars}, embargo={embargo_bars} "
            f"(need >= {last_test_end})"
        )

    folds: list[FoldSplit] = []
    for k in range(n_folds):
        test_start = first_test_start + k * test_bars
        test_end = test_start + test_bars
        embargo_end = test_start
        purge_end = embargo_end - embargo_bars
        train_end = purge_end - purge_bars
        if mode == "expanding":
            train_start = 0
        else:
            train_start = max(0, train_end - train_bars)
        if train_end - train_start < train_bars:
            raise ValueError(
                f"fold {k}: train window {train_end - train_start} < train_bars={train_bars}"
            )
        fold = FoldSplit(
            fold=k,
            train_start=train_start,
            train_end=train_end,
            purge_end=purge_end,
            embargo_end=embargo_end,
            test_start=test_start,
            test_end=test_end,
        )
        fold.assert_causal()
        folds.append(fold)
    return folds


def assert_no_leakage(folds: list[FoldSplit]) -> None:
    """Raise if any fold's train range intersects its own test range."""
    for f in folds:
        f.assert_causal()
        train = range(f.train_start, f.train_end)
        test = range(f.test_start, f.test_end)
        if not set(train).isdisjoint(test):
            raise AssertionError(f"leakage in fold {f.fold}: train∩test nonempty")
