from __future__ import annotations

from dataclasses import dataclass

import torch


NumericCategory = str


@dataclass(frozen=True)
class Tol:
    rtol: float
    atol: float


TOLS_FP32: dict[NumericCategory, Tol] = {
    "EXACT": Tol(0.0, 0.0),  # torch.equal
    "ELEMENTWISE": Tol(rtol=1e-4, atol=1e-4),
    "LOCAL": Tol(rtol=3e-4, atol=3e-4),
    "REDUCTION": Tol(rtol=1e-3, atol=1e-2),
    "GEMM_CONV": Tol(rtol=3e-3, atol=1e-2),
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
