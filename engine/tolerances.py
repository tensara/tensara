from __future__ import annotations

from dataclasses import dataclass

import torch


NumericCategory = str


@dataclass(frozen=True)
class Tol:
    rtol: float
    atol: float


# Broad, durable categories (FP32 defaults).
TOLS_FP32: dict[NumericCategory, Tol] = {
    "EXACT": Tol(0.0, 0.0),  # use torch.equal / exact checks
    "ELEMENTWISE": Tol(1e-4, 1e-4),
    "LOCAL": Tol(3e-4, 3e-4),
    "REDUCTION": Tol(1e-3, 1e-3),
    "GEMM_CONV": Tol(3e-3, 3e-3),
}


def should_verify_exact(dtype: torch.dtype, numeric_category: NumericCategory) -> bool:
    if numeric_category == "EXACT":
        return True
    return not dtype.is_floating_point


def tol_for(dtype: torch.dtype, numeric_category: NumericCategory) -> Tol | None:
    """
    Returns:
        - None if verification should be exact (torch.equal)
        - Tol(rtol, atol) otherwise
    """
    if should_verify_exact(dtype, numeric_category):
        return None
    try:
        return TOLS_FP32[numeric_category]
    except KeyError as e:
        raise KeyError(f"Unknown numeric_category={numeric_category!r}") from e

