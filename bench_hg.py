# bench_hg.py
from __future__ import annotations

import statistics
import time
from typing import Dict, List, Tuple, Optional

from utils import generate_rsa_instance_with_known_p_bits
from hg_degree1 import hg_degree1_run


def bench_one(
    *,
    p_bits: int,
    known_mode: str,   # "msb" or "lsb"
    known_bits: int,
    m: int,
    t: int,
    trials: int,
    max_rows_scan: Optional[int],
    require_smallness: bool,
    smallness_margin: float,
    early_abort_on_fail: bool = True,
) -> Tuple[float, float]:
    """
    Returns (success_rate, median_time_ms).
    If early_abort_on_fail=True, stops after first failure (faster for searching 100% region).
    """
    if known_mode == "msb":
        known_range = (p_bits - known_bits, None)
    elif known_mode == "lsb":
        known_range = (None, known_bits)
    else:
        raise ValueError("known_mode must be 'msb' or 'lsb'")

    oks: List[int] = []
    times: List[float] = []

    for _ in range(trials):
        inst = generate_rsa_instance_with_known_p_bits(
            p_bits=p_bits,
            q_bits=p_bits,
            known_range=known_range,
        )
        t0 = time.perf_counter()
        res = hg_degree1_run(
            inst, m=m, t=t,
            max_rows=max_rows_scan,
            require_smallness=require_smallness,
            smallness_margin=smallness_margin,
            scan_all_rows=True,
            return_candidates=False,
            verbose=False,
        )
        t1 = time.perf_counter()

        ok = 1 if res.success else 0
        oks.append(ok)
        times.append((t1 - t0) * 1000.0)

        if early_abort_on_fail and ok == 0:
            break

    rate = sum(oks) / len(oks)
    med_ms = statistics.median(times) if times else float("inf")
    return rate, med_ms


def find_best_params(
    *,
    p_bits: int,
    known_mode: str,
    known_bits: int,
    m_values: List[int],
    t_values: List[int],
    trials: int,
    max_rows_scan: Optional[int] = None,
    require_smallness: bool = True,
    smallness_margin: float = 1.0,
) -> None:
    """
    Prints:
      - grid of success rates
      - grid of median times
      - best (m,t) among those with 100% success (within given trials)
    """
    rates: Dict[int, Dict[int, float]] = {m: {} for m in m_values}
    times: Dict[int, Dict[int, float]] = {m: {} for m in m_values}

    # Track best among 100% success
    best = None  # (med_ms, m, t)

    for m in m_values:
        for t in t_values:
            rate, med_ms = bench_one(
                p_bits=p_bits,
                known_mode=known_mode,
                known_bits=known_bits,
                m=m, t=t,
                trials=trials,
                max_rows_scan=max_rows_scan,
                require_smallness=require_smallness,
                smallness_margin=smallness_margin,
                early_abort_on_fail=True,   # key for speed
            )
            rates[m][t] = rate
            times[m][t] = med_ms

            if rate == 1.0:
                cand = (med_ms, m, t)
                if best is None or cand < best:
                    best = cand

    def print_table(title: str, data: Dict[int, Dict[int, float]], fmt: str):
        print("\n" + title)
        header = "m\\t | " + " ".join([f"{t:>7d}" for t in t_values])
        print(header)
        print("-" * len(header))
        for m in m_values:
            line = f"{m:>3d} | " + " ".join([fmt.format(data[m].get(t, float("nan"))) for t in t_values])
            print(line)

    print_table(
        f"Success rate (p_bits={p_bits}, mode={known_mode}, known_bits={known_bits}, trials<= {trials} w/early-abort)",
        rates,
        "{:7.2f}",
    )
    print_table(
        "Median time [ms] (computed over executed trials; early-abort may shorten)",
        times,
        "{:7.1f}",
    )

    if best is None:
        # fallback: best by (highest rate, then fastest)
        fallback = None  # (-rate, med_ms, m, t)
        for m in m_values:
            for t in t_values:
                rate = rates[m][t]
                med_ms = times[m][t]
                cand = (-rate, med_ms, m, t)
                if fallback is None or cand < fallback:
                    fallback = cand
        assert fallback is not None
        _, med_ms, m, t = fallback
        print("\nNo (m,t) achieved 100% within this trial budget.")
        print(f"Best by (rate desc, time asc): m={m}, t={t}, rate={rates[m][t]:.2f}, median_ms={med_ms:.1f}")
    else:
        med_ms, m, t = best
        print(f"\nBEST (100% over executed trials): m={m}, t={t}, median_ms={med_ms:.1f}")

        # Optional: re-validate the winner with full trials (no early abort)
        print("\nRe-validating best with full trials (no early abort)...")
        rate2, med2 = bench_one(
            p_bits=p_bits,
            known_mode=known_mode,
            known_bits=known_bits,
            m=m, t=t,
            trials=trials,
            max_rows_scan=max_rows_scan,
            require_smallness=require_smallness,
            smallness_margin=smallness_margin,
            early_abort_on_fail=False,
        )
        print(f"Validation: rate={rate2:.2f} (over {trials}), median_ms={med2:.1f}")


if __name__ == "__main__":
    # Example: your tiny case
    p_bits = 20

    # MSB-known 14 (unknown 6 low bits)
    known_mode = "msb"
    known_bits = 14

    m_values = list(range(1, 8))   # widen as you like
    t_values = list(range(1, 10))
    trials = 50

    # If you want “realistic” success (not using secret p in inequality), set require_smallness=False
    require_smallness = True
    smallness_margin = 1.0

    find_best_params(
        p_bits=p_bits,
        known_mode=known_mode,
        known_bits=known_bits,
        m_values=m_values,
        t_values=t_values,
        trials=trials,
        max_rows_scan=None,
        require_smallness=require_smallness,
        smallness_margin=smallness_margin,
    )
