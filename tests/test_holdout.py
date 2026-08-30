"""Held-out evaluation helpers."""
import math

import pytest

from chessgraph.evaluation.holdout import spearman


def test_spearman_perfect_and_inverse():
    assert spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_is_rank_based_not_linear():
    """A monotone but non-linear relation must still score 1.0."""
    assert spearman([1, 2, 3, 4, 5], [1, 4, 9, 16, 25]) == pytest.approx(1.0)


def test_spearman_handles_ties_with_average_ranks():
    rho = spearman([1, 1, 2, 2], [1, 1, 2, 2])
    assert rho == pytest.approx(1.0)


def test_spearman_undefined_below_two_points():
    assert math.isnan(spearman([1], [1]))


def test_spearman_is_nan_when_a_series_is_constant():
    """No variance means no rank correlation, not a divide by zero."""
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))
