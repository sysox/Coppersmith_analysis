# utils.py
# Helper utilities for HG/Coppersmith experiments (degree-1 focus).
# Designed for easy use in Jupyter notebooks.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import math
import random
import time

import sympy as sp
from sympy import Poly
from fpylll import IntegerMatrix

# Global polynomial variable
x = sp.Symbol("x", integer=True)


# ============================================================
# Bit / integer helpers
# ============================================================

def bitlen(n: int) -> int:
    if n < 0:
        raise ValueError("bitlen expects non-negative integer")
    return n.bit_length()


def mask(lo: int, hi: int) -> int:
    """Mask for bit interval [lo, hi) with LSB index 0."""
    if lo < 0 or hi < 0 or hi < lo:
        raise ValueError("invalid bit range")
    return ((1 << (hi - lo)) - 1) << lo


def extract_bits(n: int, lo: int, hi: int) -> int:
    """Extract bits in [lo, hi) as an integer (shifted down by lo)."""
    return (n & mask(lo, hi)) >> lo


def replace_bits(n: int, lo: int, hi: int, value: int) -> int:
    """Replace bits in [lo, hi) with value (value interpreted in that width)."""
    w = hi - lo
    if value < 0 or value >= (1 << w):
        raise ValueError("value doesn't fit the target bit range width")
    n_cleared = n & ~mask(lo, hi)
    return n_cleared | (value << lo)


def normalize_known_range(bitsize: int,
                          known_range: Tuple[Optional[int], Optional[int]]) -> Tuple[int, int]:
    """
    Normalize [lo, hi) bit indices, with None allowed.
    If lo is None -> 0. If hi is None -> bitsize.
    """
    lo, hi = known_range
    lo = 0 if lo is None else lo
    hi = bitsize if hi is None else hi
    if not (0 <= lo <= hi <= bitsize):
        raise ValueError(f"Invalid known_range {known_range} for bitsize={bitsize}")
    return lo, hi


def invmod(a: int, m: int) -> int:
    """Modular inverse a^{-1} mod m."""
    a %= m
    if a == 0:
        raise ValueError("invmod: a==0 has no inverse")
    return int(sp.invert(a, m))


# ============================================================
# RSA instance generation + “known bits” model
# ============================================================

def rand_prime(bits: int, rng: Optional[random.Random] = None) -> int:
    """Random odd prime of exact bit-length using sympy."""
    if bits < 2:
        raise ValueError("bits must be >= 2")
    rng = rng or random
    lo = 1 << (bits - 1)
    hi = (1 << bits) - 1
    p = int(sp.randprime(lo, hi + 1))
    while p.bit_length() != bits:
        p = int(sp.randprime(lo, hi + 1))
    return p


@dataclass
class RSAInstance:
    p: int
    q: int
    N: int

    p_bits: int
    known_range: Tuple[int, int]         # [lo, hi) in p, LSB index 0
    known_value: int                     # extracted bits shifted down by lo
    known_value_aligned: int             # known_value placed back at position lo

    def __repr__(self) -> str:
        lo, hi = self.known_range
        return (f"RSAInstance(N={self.N}, p={self.p}, q={self.q}, "
                f"p_bits={self.p_bits}, known_range=[{lo},{hi}), "
                f"known_value=0x{self.known_value:x})")


def generate_rsa_instance_with_known_p_bits(
    p_bits: int,
    q_bits: Optional[int] = None,
    known_range: Tuple[Optional[int], Optional[int]] = (None, None),
    rng: Optional[random.Random] = None,
) -> RSAInstance:
    """
    Generate RSA (p,q,N) and reveal bits of p on interval [lo, hi) (LSB index 0).

    known_range may contain None:
      (None, k)   -> [0, k) known (LSB known)
      (p_bits-k, None) -> [p_bits-k, p_bits) known (MSB known)
      (None, None) -> all bits known (not useful, but allowed)
    """
    rng = rng or random
    q_bits = q_bits or p_bits

    p = rand_prime(p_bits, rng=rng)
    q = rand_prime(q_bits, rng=rng)
    while q == p:
        q = rand_prime(q_bits, rng=rng)

    N = p * q
    lo, hi = normalize_known_range(p_bits, known_range)

    kv = extract_bits(p, lo, hi)
    kva = kv << lo
    return RSAInstance(
        p=p, q=q, N=N,
        p_bits=p_bits,
        known_range=(lo, hi),
        known_value=kv,
        known_value_aligned=kva,
    )


# ============================================================
# Degree-1 target polynomial construction: f(x)=x+a
# ============================================================

def linear_f_from_known_msb(p_bits: int, known_msb_value: int, known_msb_bits: int) -> Poly:
    """
    MSB-known model:
      p = (known_msb_value << k) + x0,  0 <= x0 < 2^k
    Then f(x) = x + a with a = known_msb_value << k satisfies f(x0)=p, so f(x0)≡0 (mod p).
    """
    if not (1 <= known_msb_bits <= p_bits):
        raise ValueError("known_msb_bits must be in [1, p_bits]")
    k = p_bits - known_msb_bits
    a = int(known_msb_value) << k
    return Poly(x + a, x, domain=sp.ZZ)


def linear_f_from_known_lsb_monic_modN(N: int, known_lsb_value: int, known_lsb_bits: int) -> Poly:
    """
    LSB-known model (monic normalization mod N):
      p = known_lsb_value + 2^k * x0
    raw: f_raw(x) = 2^k x + known
    normalize by multiplying inv(2^k) mod N:
      f(x) = x + a  over ZZ where a = known * inv(2^k) mod N.
    Then f(x0) ≡ 0 (mod p).
    """
    k = int(known_lsb_bits)
    if k <= 0:
        raise ValueError("known_lsb_bits must be >= 1")
    inv2k = invmod(1 << k, int(N))
    a = (int(known_lsb_value) * inv2k) % int(N)
    return Poly(x + a, x, domain=sp.ZZ)


def build_linear_f_from_instance(inst: RSAInstance, N: Optional[int] = None) -> Tuple[Poly, int]:
    """
    Convenience: detect whether known bits are pure LSB-known or pure MSB-known and return:
      (f(x)=x+a, X_bound)

    - If known_range starts at 0 -> LSB known: p = known + 2^k x0, x0 < 2^(p_bits-k)
      returns monic-normalized f(x)=x+a (mod N), X = 2^(p_bits-k).
    - If known_range ends at p_bits -> MSB known: p = (known<<k) + x0, x0 < 2^k
      returns f(x)=x+a, X = 2^k.
    """
    N = int(N) if N is not None else int(inst.N)
    lo, hi = inst.known_range
    width = hi - lo

    if lo == 0 and hi < inst.p_bits:
        # LSB known
        k = width
        f = linear_f_from_known_lsb_monic_modN(N, inst.known_value, k)
        Xb = 1 << (inst.p_bits - k)
        return f, Xb

    if hi == inst.p_bits and lo > 0:
        # MSB known
        known_msb_bits = width
        known_msb_value = inst.known_value  # top bits as integer
        f = linear_f_from_known_msb(inst.p_bits, known_msb_value, known_msb_bits)
        k = inst.p_bits - known_msb_bits
        Xb = 1 << k
        return f, Xb

    raise ValueError(
        f"build_linear_f_from_instance: known_range={inst.known_range} is neither pure LSB nor pure MSB. "
        f"Provide a custom model for middle-bit leaks."
    )


# ============================================================
# Polynomial helpers: scaling, vectorization, matrix conversion
# ============================================================

def pretty_poly(f: Poly) -> str:
    return str(Poly(f, x, domain=sp.ZZ).as_expr())


def poly_sub_xX(f: Poly, X: int) -> Poly:
    """Return f(X*x) over ZZ[x]."""
    f = Poly(f, x, domain=sp.ZZ)
    X = int(X)
    expr = f.as_expr().subs({x: X * x})
    return Poly(sp.expand(expr), x, domain=sp.ZZ)


def poly_degree(f: Poly) -> int:
    return int(Poly(f, x, domain=sp.ZZ).degree())


def poly_to_vector(f: Poly, d: Optional[int] = None) -> List[int]:
    """
    Vectorize polynomial coefficients in ascending monomials [x^0, x^1, ..., x^d].
    """
    f = Poly(f, x, domain=sp.ZZ)
    deg = f.degree() if d is None else int(d)
    return [int(f.nth(i)) for i in range(deg + 1)]


def vector_to_poly(v: Sequence[int]) -> Poly:
    """Inverse of poly_to_vector (ascending order)."""
    expr = sum(int(c) * x**i for i, c in enumerate(v))
    return Poly(expr, x, domain=sp.ZZ)


def to_integer_matrix(rows: Sequence[Sequence[int]]) -> IntegerMatrix:
    """Convert list-of-lists to fpylll.IntegerMatrix."""
    if not rows:
        raise ValueError("Empty rows")
    r = len(rows)
    c = len(rows[0])
    M = IntegerMatrix(r, c)
    for i in range(r):
        if len(rows[i]) != c:
            raise ValueError("Non-rectangular rows")
        for j in range(c):
            M[i, j] = int(rows[i][j])
    return M


# ============================================================
# HG basis construction (generic m,t, generic degree)
# ============================================================

def hg_polynomials(f: Poly, N: int, m: int, t: int, X: int) -> List[Poly]:
    """
    Standard univariate HG/Coppersmith basis (degree-1 friendly):
      - N^m
      - N^i * f^(m-i)   for i = 0..m-1   (includes f^m at i=0)
      - x^j * f^m       for j = 1..t-1   (avoids duplicating f^m)

    Then scale x <- X*x.
    For m=2,t=2 this yields: N^2, N f, f^2, x f^2.
    """
    if m <= 0 or t <= 0:
        raise ValueError("Need m>=1 and t>=1")
    N = int(N)
    X = int(X)
    f = Poly(f, x, domain=sp.ZZ)

    polys: List[Poly] = []

    # 1) N^m
    polys.append(Poly(N**m, x, domain=sp.ZZ))

    # 2) N^i * f^(m-i), i=0..m-1  (includes f^m)
    for i in range(0, m):
        polys.append((N**i) * (f ** (m - i)))

    # 3) x^j * f^m, j=1..t-1
    fm = f ** m
    for j in range(1, t):
        polys.append((x**j) * fm)

    # scale x <- X*x
    return [poly_sub_xX(g, X) for g in polys]




def hg_basis_matrix(f: Poly, N: int, m: int, t: int, X: int) -> Tuple[IntegerMatrix, List[Poly], int]:
    """
    Return (B, polys, maxdeg) where B rows are coefficient vectors of polys (ascending).
    """
    polys = hg_polynomials(f, N, m, t, X)
    maxdeg = max(poly_degree(g) for g in polys) if polys else 0
    rows = [poly_to_vector(g, d=maxdeg) for g in polys]
    return to_integer_matrix(rows), polys, maxdeg


# ============================================================
# Root checks + timing + unscale helpers
# ============================================================

def eval_poly_int(F: Union[Poly, sp.Expr], x0: int) -> int:
    """Evaluate polynomial/expression at x0 and return int."""
    if isinstance(F, Poly):
        return int(F.eval(x0))
    return int(sp.expand(F).subs({x: int(x0)}))


def is_root_Z(F: Union[Poly, sp.Expr], x0: int) -> bool:
    """Check F(x0) == 0 in Z."""
    return eval_poly_int(F, x0) == 0


def is_root_mod(F: Union[Poly, sp.Expr], x0: int, modulus: int) -> bool:
    """Check F(x0) ≡ 0 (mod modulus)."""
    m = int(modulus)
    if m <= 0:
        raise ValueError("modulus must be positive")
    return eval_poly_int(F, x0) % m == 0


def time_call(fn, *args, **kwargs) -> Tuple[Any, float]:
    """Call fn(*args,**kwargs) and return (result, seconds)."""
    t0 = time.perf_counter()
    res = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return res, (t1 - t0)


def unscale_poly_exact(F_scaled: Poly, X: int) -> Poly:
    X = int(X)
    F_scaled = Poly(F_scaled, x, domain=sp.ZZ)
    if F_scaled.is_zero:
        return Poly(0, x, domain=sp.ZZ)

    deg = int(F_scaled.degree())
    coeffs = [int(F_scaled.nth(k)) for k in range(deg + 1)]  # ascending
    new_coeffs: List[int] = []
    for k, ck in enumerate(coeffs):
        denom = X ** k
        if ck % denom != 0:
            raise AssertionError(f"Unscale failed at degree {k}: {ck} not divisible by {denom}")
        new_coeffs.append(ck // denom)

    return Poly(sum(new_coeffs[k] * x**k for k in range(len(new_coeffs))), x, domain=sp.ZZ)




def row_to_list(B: IntegerMatrix, i: int) -> List[int]:
    """IntegerMatrix row -> python list[int]."""
    return [int(B[i, j]) for j in range(B.ncols)]


def true_x0_degree1(inst: RSAInstance, f: Poly) -> int:
    """
    Ground-truth x0 for supported degree-1 models (useful for experiments):
      - MSB-known: p = a + x0, f(x)=x+a => x0 = p - a
      - LSB-known: p = known + 2^k x0 => x0 = (p-known)/2^k
    """
    lo, hi = inst.known_range
    a = int(Poly(f, x, domain=sp.ZZ).eval(0))

    if hi == inst.p_bits and lo > 0:
        return int(inst.p - a)

    if lo == 0 and hi < inst.p_bits:
        k = hi
        return int((inst.p - inst.known_value_aligned) >> k)

    raise ValueError("true_x0_degree1: unsupported known_range (not pure MSB/LSB).")


# ============================================================
# __main__ small demo (20-bit primes; MSB-known 14 / LSB-known 6)
# ============================================================

if __name__ == "__main__":
    from fpylll import LLL

    print("=== utils.py demo (tiny) ===")

    # Case A: MSB known 14 bits => unknown 6 low bits => known_range [6,20)
    instA = generate_rsa_instance_with_known_p_bits(p_bits=20, q_bits=20, known_range=(6, None))
    fA, XA = build_linear_f_from_instance(instA)
    x0A = true_x0_degree1(instA, fA)

    print("\n[Case A] MSB-known 14, unknown 6 low")
    print("p,q,N:", instA.p, instA.q, instA.N)
    print("f(x):", pretty_poly(fA), "X:", XA, "x0:", x0A)
    print("check f(x0) mod p:", is_root_mod(fA, x0A, instA.p))

    BA, polysA, degA = hg_basis_matrix(fA, instA.N, m=2, t=2, X=XA)
    BAred = IntegerMatrix(BA)
    LLL.reduction(BAred)
    # sanity: unscale all rows
    for i in range(BAred.nrows):
        F_scaled = vector_to_poly(row_to_list(BAred, i))
        _ = unscale_poly_exact(F_scaled, XA)
    print("LLL + unscale sanity: OK")

    # Case B: LSB known 6 bits => unknown 14 high bits => known_range [0,6)
    instB = generate_rsa_instance_with_known_p_bits(p_bits=20, q_bits=20, known_range=(None, 6))
    fB, XB = build_linear_f_from_instance(instB)
    x0B = true_x0_degree1(instB, fB)

    print("\n[Case B] LSB-known 6, unknown 14 high")
    print("p,q,N:", instB.p, instB.q, instB.N)
    print("f(x):", pretty_poly(fB), "X:", XB, "x0:", x0B)
    print("check f(x0) mod p:", is_root_mod(fB, x0B, instB.p))

    BB, polysB, degB = hg_basis_matrix(fB, instB.N, m=2, t=2, X=XB)
    BBred = IntegerMatrix(BB)
    LLL.reduction(BBred)
    for i in range(BBred.nrows):
        F_scaled = vector_to_poly(row_to_list(BBred, i))
        _ = unscale_poly_exact(F_scaled, XB)
    print("LLL + unscale sanity: OK")

    print("\n=== done ===")
