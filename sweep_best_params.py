#!/usr/bin/env python3
# sweep_best_params.py
#
# For each K in [ceil(p_bits/2), p_bits-1] (or override),
# we pre-generate instances ONCE and reuse them while searching best (m,t)
# achieving 100% success in the quick set; then optionally validate on another set.
#
# Output: aligned table with percentages.

from __future__ import annotations

import argparse
import statistics
import time
from typing import List, Optional, Tuple

from utils import generate_rsa_instance_with_known_p_bits, RSAInstance
from hg_degree1 import hg_degree1_run


def parse_ab(s: str) -> Tuple[int, int]:
    """Parse 'a:b' into inclusive [a..b]."""
    parts = s.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected a:b (e.g. 1:10)")
    a = int(parts[0])
    b = int(parts[1])
    if a > b:
        raise argparse.ArgumentTypeError("Range must satisfy a <= b")
    return a, b


def known_range(p_bits: int, mode: str, K: int) -> Tuple[Optional[int], Optional[int]]:
    if not (0 <= K <= p_bits):
        raise ValueError("K out of range")
    if mode == "lsb":
        return (None, K)           # [0,K)
    if mode == "msb":
        return (p_bits - K, None)  # [p_bits-K, p_bits)
    raise ValueError("mode must be lsb or msb")


def generate_instances(
    *,
    p_bits: int,
    mode: str,
    K: int,
    count: int,
) -> List[RSAInstance]:
    if count <= 0:
        return []
    kr = known_range(p_bits, mode, K)
    return [
        generate_rsa_instance_with_known_p_bits(p_bits=p_bits, q_bits=p_bits, known_range=kr)
        for _ in range(count)
    ]


def run_on_instances(
    instances: List[RSAInstance],
    *,
    m: int,
    t: int,
    max_rows: Optional[int],
    scan_all_rows: bool,
    early_abort_on_fail: bool,
) -> Tuple[float, float, int]:
    """
    Returns (success_rate, median_ms, executed_trials).
    Uses hg_degree1_run() which should be a realistic success criterion (factor found).
    """
    if not instances:
        return 0.0, float("inf"), 0

    oks: List[int] = []
    times_ms: List[float] = []

    for inst in instances:
        t0 = time.perf_counter()
        res = hg_degree1_run(
            inst,
            m=m,
            t=t,
            max_rows=max_rows,
            scan_all_rows=scan_all_rows,
            return_candidates=False,
            verbose=False,
        )
        t1 = time.perf_counter()

        ok = 1 if res.success else 0
        oks.append(ok)
        times_ms.append((t1 - t0) * 1000.0)

        if early_abort_on_fail and ok == 0:
            break

    rate = sum(oks) / len(oks)
    med = statistics.median(times_ms) if times_ms else float("inf")
    return rate, med, len(oks)


def pct(x: float) -> str:
    return f"{100.0 * x:6.1f}%"


def ms(x: float) -> str:
    if x == float("inf"):
        return "   -   "
    return f"{x:7.1f}"


def main():
    ap = argparse.ArgumentParser(
        description="Sweep known bits K (half..end) and find fastest (m,t) with 100% quick success; reuse instances."
    )
    ap.add_argument("--p-bits", type=int, required=True)
    ap.add_argument("--mode", choices=["lsb", "msb"], required=True)

    ap.add_argument("--m", type=parse_ab, default=(1, 10), help="m range a:b (inclusive), default 1:10")
    ap.add_argument("--t", type=parse_ab, default=(1, 10), help="t range a:b (inclusive), default 1:10")

    ap.add_argument("--trials-quick", type=int, default=10,
                    help="Number of quick instances per K (reused across all (m,t))")
    ap.add_argument("--trials-validate", type=int, default=0,
                    help="Number of validation instances per K (0 = skip)")

    ap.add_argument("--max-rows", type=int, default=None,
                    help="Scan only first max_rows of LLL-reduced basis (None=all)")
    ap.add_argument("--scan-all-rows", action="store_true",
                    help="If set, scan ALL rows (slower but safer). Default scans only up to --max-rows.")
    ap.add_argument("--early-abort", action="store_true",
                    help="If set, during quick testing abort (m,t) as soon as a failure occurs (faster).")

    ap.add_argument("--k-start", type=int, default=None,
                    help="Override start K. Default ceil(p_bits/2).")
    ap.add_argument("--k-end", type=int, default=None,
                    help="Override end K (inclusive). Default p_bits-1.")

    args = ap.parse_args()

    if args.trials_quick <= 0:
        raise SystemExit("--trials-quick must be >= 1")
    if args.trials_validate < 0:
        raise SystemExit("--trials-validate must be >= 0")

    p_bits = args.p_bits
    m_min, m_max = args.m
    t_min, t_max = args.t

    k0 = (p_bits + 1) // 2 if args.k_start is None else args.k_start
    k1 = (p_bits - 1) if args.k_end is None else args.k_end
    if not (0 <= k0 <= k1 <= p_bits - 1):
        raise SystemExit("Bad K interval: ensure 0 <= k-start <= k-end <= p_bits-1")

    # Decide scanning strategy
    scan_all_rows = True if args.scan_all_rows else False
    # if scan_all_rows is False, we still pass max_rows; hg_degree1_run will use it
    # (your hg_degree1_run scans rows depending on max_rows + scan_all_rows)
    # So: pass scan_all_rows flag through; max_rows may be None.
    max_rows = args.max_rows

    # Header
    print(f"\nHG sweep (reuse instances) | p_bits={p_bits} | mode={args.mode.upper()} | "
          f"m=[{m_min},{m_max}] t=[{t_min},{t_max}] | "
          f"quick={args.trials_quick} validate={args.trials_validate}")
    print(f"Row scan: {'ALL' if scan_all_rows else ('first '+str(max_rows) if max_rows else 'ALL')} | "
          f"quick early-abort: {'ON' if args.early_abort else 'OFF'}\n")

    # Pretty table
    # Columns: Known | Unknown | Best(m,t) | Quick% | Quick ms | Val% | Val ms
    col_known = 8
    col_unk = 8
    col_best = 10
    col_qpct = 8
    col_qms = 9
    col_vpct = 8
    col_vms = 8

    def line(ch: str = "-") -> str:
        return (ch * (col_known + col_unk + col_best + col_qpct + col_qms + col_vpct + col_vms + 6))

    header = (
        f"{'Known':<{col_known}} "
        f"{'Unknown':<{col_unk}} "
        f"{'Best(m,t)':<{col_best}} "
        f"{'Quick%':>{col_qpct}} "
        f"{'Quickms':>{col_qms}} "
        f"{'Val%':>{col_vpct}} "
        f"{'Valms':>{col_vms}}"
    )
    print(header)
    print(line("-"))

    for K in range(k0, k1 + 1):
        unk = p_bits - K
        tag = f"{args.mode.upper()}{K}"

        # Pre-generate instances ONCE for this K
        quick_insts = generate_instances(p_bits=p_bits, mode=args.mode, K=K, count=args.trials_quick)
        val_insts = generate_instances(p_bits=p_bits, mode=args.mode, K=K, count=args.trials_validate) if args.trials_validate > 0 else []

        best = None  # (quick_median_ms, m, t, quick_rate, quick_executed)

        # Search best among those with 100% on quick set
        for m in range(m_min, m_max + 1):
            for t in range(t_min, t_max + 1):
                q_rate, q_med, q_exec = run_on_instances(
                    quick_insts,
                    m=m, t=t,
                    max_rows=max_rows,
                    scan_all_rows=scan_all_rows,
                    early_abort_on_fail=args.early_abort,
                )
                if q_rate == 1.0:
                    cand = (q_med, m, t, q_rate, q_exec)
                    if best is None or cand < best:
                        best = cand

        if best is None:
            # No 100% candidate on quick set
            row = (
                f"{tag:<{col_known}} "
                f"{unk:<{col_unk}} "
                f"{'-':<{col_best}} "
                f"{'  0.0%':>{col_qpct}} "
                f"{ms(float('inf')):>{col_qms}} "
                f"{'   -  ':>{col_vpct}} "
                f"{'   -  ':>{col_vms}}"
            )
            print(row)
            continue

        q_med, m_best, t_best, q_rate, _ = best

        # Optional validation
        if args.trials_validate > 0:
            v_rate, v_med, _ = run_on_instances(
                val_insts,
                m=m_best, t=t_best,
                max_rows=max_rows,
                scan_all_rows=scan_all_rows,
                early_abort_on_fail=False,
            )
            v_pct = pct(v_rate)
            v_ms = ms(v_med)
        else:
            v_pct = "   -  "
            v_ms = "   -  "

        row = (
            f"{tag:<{col_known}} "
            f"{unk:<{col_unk}} "
            f"{f'({m_best},{t_best})':<{col_best}} "
            f"{pct(q_rate):>{col_qpct}} "
            f"{ms(q_med):>{col_qms}} "
            f"{v_pct:>{col_vpct}} "
            f"{v_ms:>{col_vms}}"
        )
        print(row)

    print(line("-"))
    print("Legend: Known=MSB/LSB + number of known bits; Unknown = remaining bits of p.")
    print("Quick% is on the reused quick instance set; Val% is on a separate reused validation set (if enabled).")


if __name__ == "__main__":
    main()
