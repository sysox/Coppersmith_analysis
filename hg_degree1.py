# hg_degree1.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import sympy as sp
from sympy import Poly
from fpylll import IntegerMatrix, LLL

from utils import (
    RSAInstance,
    build_linear_f_from_instance,
    hg_basis_matrix,
    row_to_list,
    vector_to_poly,
    unscale_poly_exact,
    true_x0_degree1,
    x,
)

@dataclass
class HGResult:
    success: bool
    best: Optional[Dict[str, Any]]
    candidates: List[Dict[str, Any]]
    meta: Dict[str, Any]


def _row_norm2(v: List[int]) -> int:
    return sum(int(c) * int(c) for c in v)


def hg_degree1_run(
    inst: RSAInstance,
    m: int,
    t: int,
    *,
    max_rows: Optional[int] = None,
    scan_all_rows: bool = True,
    return_candidates: bool = True,
    verbose: bool = False,
) -> HGResult:
    """
    HG for degree-1 (f(x)=x+a) with a *realistic* success criterion:

    A run is counted as SUCCESS iff we can recover p (or q) from the reduced basis
    WITHOUT using secret information in the decision rule.

    Concretely, for each reduced row we:
      1) build F_scaled from the row vector
      2) unscale to F over ZZ[x]
      3) find *integer* roots of F (for degree 1 this is trivial; for higher deg we use sympy)
      4) for each integer root r, compute gcd(f(r), N) or gcd(F(r), N) and succeed if gcd is a non-trivial factor.

    Notes:
      - We do NOT use the true x0 nor the inequality |F(x0)|<p^m for deciding success.
      - This makes m=1,t=1 stop producing false positives.
      - We still record meta/candidate info for debugging.
    """
    import math
    import sympy as sp

    f, X = build_linear_f_from_instance(inst)
    N = int(inst.N)

    # Build basis and reduce
    B, basis_polys, maxdeg = hg_basis_matrix(f, N, m=m, t=t, X=X)
    Bred = IntegerMatrix(B)
    LLL.reduction(Bred)

    nrows = Bred.nrows
    limit = nrows if max_rows is None else min(nrows, int(max_rows))
    rows_to_scan = range(limit) if scan_all_rows else range(min(limit, 1))

    candidates: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    success = False

    def row_norm2(v: List[int]) -> int:
        return sum(int(c) * int(c) for c in v)

    def is_nontrivial_factor(g: int) -> bool:
        return 1 < g < N

    for i in rows_to_scan:
        row = row_to_list(Bred, i)
        if all(c == 0 for c in row):
            continue

        F_scaled = vector_to_poly(row)
        try:
            F = unscale_poly_exact(F_scaled, X)
        except AssertionError:
            # unscale failed => skip this row
            continue

        # normalize to a primitive polynomial (optional but helps stability)
        F = Poly(F, x, domain=sp.ZZ)
        if F.is_zero:
            continue
        content = int(sp.gcd_list(list(F.all_coeffs())))
        if content not in (0, 1, -1):
            F = Poly(F.as_expr() // content, x, domain=sp.ZZ)

        deg = int(F.degree())
        roots: List[int] = []

        if deg == 1:
            a1 = int(F.nth(1))
            a0 = int(F.nth(0))
            if a1 != 0 and (-a0) % a1 == 0:
                roots.append(int((-a0) // a1))
        else:
            # For higher degree, use sympy integer roots (slow but OK for small experiments)
            try:
                rr = sp.roots(F.as_expr(), x)
                for r, mult in rr.items():
                    if r.is_integer:
                        roots.append(int(r))
            except Exception:
                pass

        found_factor = None
        factor_via = None
        root_used = None

        # Test each integer root candidate
        for r in roots:
            # We expect small roots, but don't strictly require it.
            # Use gcd with N as the real success signal.
            try:
                val_f = int(f.eval(r))
            except Exception:
                val_f = int(sp.expand(f.as_expr()).subs({x: r}))

            g1 = math.gcd(val_f, N)
            if is_nontrivial_factor(g1):
                found_factor = g1
                factor_via = "gcd(f(r),N)"
                root_used = r
                break

            val_F = int(F.eval(r))
            g2 = math.gcd(val_F, N)
            if is_nontrivial_factor(g2):
                found_factor = g2
                factor_via = "gcd(F(r),N)"
                root_used = r
                break

        info = {
            "row_index": i,
            "row_norm2": row_norm2(row),
            "m": m,
            "t": t,
            "X": X,
            "f": f,
            "F": F,
            "F_scaled": F_scaled,
            "deg": deg,
            "int_roots": roots,
            "found_factor": found_factor,
            "factor_via": factor_via,
            "root_used": root_used,
        }
        candidates.append(info)

        if verbose:
            print(f"[row {i}] norm2={info['row_norm2']} deg={deg} roots={roots} factor={found_factor}")

        if found_factor is not None:
            success = True
            if best is None or info["row_norm2"] < best["row_norm2"]:
                best = info

    candidates.sort(key=lambda d: d["row_norm2"])
    if best is None and candidates:
        best = candidates[0]

    meta = {
        "m": m,
        "t": t,
        "basis_rows": int(Bred.nrows),
        "basis_cols": int(Bred.ncols),
        "maxdeg": maxdeg,
    }

    if not return_candidates:
        candidates = []

    return HGResult(success=success, best=best, candidates=candidates, meta=meta)

