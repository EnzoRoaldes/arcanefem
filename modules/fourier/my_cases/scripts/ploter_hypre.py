#!/usr/bin/env python3
"""
Plot ArcaneFEM scaling metrics from PETSc or repeated Hypre solve logs.

Input format:

*I-Fem        [ArcaneFem-Timer] initialize             = 0.0264949798583984
*I-Fem        [ArcaneFem-Timer] get-material-params    = 2.38418579101562e-07
*I-Fem        [ArcaneFem-Timer] lhs-matrix-assembly    = 0.0633952617645264
*I-Fem        [ArcaneFem-Timer] rhs-vector-assembly-gpu = 0.0249457359313965
*I-Fem        [ArcaneFem-Timer] solve-linear-system-1  = 7.76405930519104
*I-Fem        [ArcaneFem-Timer] solve-linear-system-2  = 0.202636241912842
*I-Fem        [ArcaneFem-Timer] update-variables       = 0.000773429870605469
*I-Fem        [ArcaneFem-Timer] compute                = 8.16862058639526

Runs are separated by '# procs=N' headers (or blank lines for timer-only files).
For Hypre, each '[Hypre-Timer] Time to setup' is paired with the following
'Time to solve'. The median of each stage and of each pair's sum is exported.
Scaling plots show Hypre total, setup and solve by default. --include-assembly
adds LHS and RHS. For PETSc, --solver-breakdown adds PCSetUp/KSPSolve/PCApply.
Only the CSV and speedup/efficiency plots are generated.
For PETSc event details, use the full ASCII log produced with -log_view.
Full logs may also be separated by Arcane startup banners.
Process counts in headers/PETSc summaries are used when --procs is omitted.
Keep warm-up output in a separate file. Repeated PETSc profiles/stages are
rejected instead of silently overwriting event times.

PETSc event times are the Time (sec) Max column, accumulated over all calls.
They can overlap (e.g. PCApply inside KSPSolve); no stacked/additive total is
inferred. Separate event maxima can also come from different MPI ranks.
'Time to solve' is the measured [Petsc-Timer] value in PETSc logs.
If several Time to solve lines occur in one run, their durations are summed.

Examples:
    python3 ploter_hypre.py ../outputs/cpu/StrongScaling.txt --procs 192 128 96 64 32 16 8 4 2 --mode strong --logx
    python3 ploter_hypre.py ../outputs/gpu/StrongScaling.txt --mode strong --logx
    options : --mode | --outdir | --procs | --logx --logy | --initial-guess | --solver-breakdown | --include-assembly
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TIMERS = [
    "lhs-matrix-assembly",
    "rhs-vector-assembly-gpu",
    "solve-linear-system",
    "solve-linear-system-1",
    "solve-linear-system-2",
    "solve-linear-system-total",
    "compute",
]

TIMER_RE = re.compile(
    r"\[ArcaneFem-Timer\]\s+([A-Za-z0-9_\-]+)\s*=\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)

PETSC_EVENTS = ["PCSetUp", "KSPSolve", "PCApply"]
TIME_TO_SOLVE = "Time to solve"
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
HYPRE_SETUP = "hypre-setup"
HYPRE_SOLVE = "hypre-solve"
HYPRE_TOTAL = "hypre-total"
HYPRE_TIMERS = [HYPRE_TOTAL, HYPRE_SETUP, HYPRE_SOLVE]
HYPRE_STAGE_RE = re.compile(
    rf"\[Hypre-Timer\]\s+Time to (setup|solve)\s*=\s*({NUMBER})(?=\s|$)"
)
PETSC_EVENT_RE = re.compile(
    rf"^\s*(PCSetUp|KSPSolve|PCApply)\s+\d+\s+{NUMBER}\s+({NUMBER})(?=\s|$)"
)
PETSC_SOLVE_RE = re.compile(rf"\[Petsc-Timer\]\s+Time to solve\s*=\s*({NUMBER})")
RUN_HEADER_RE = re.compile(r"^\s*#\s*procs\s*=\s*(\d+)\b")
ARCANE_BANNER_RE = re.compile(r"^\s*\*I-Internal\s+Arcane\s*$")
PETSC_PROCS_RE = re.compile(r"\bwith\s+(\d+)\s+process(?:es)?\s*,")


def parse_blocks(path: Path) -> List[Dict[str, float]]:
    """Read timer blocks or full Arcane/PETSc logs without splitting their blank lines."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    has_headers = any(RUN_HEADER_RE.match(line) for line in lines)
    has_banners = any(ARCANE_BANNER_RE.match(line) for line in lines)
    full_log = has_banners or any("PETSc Performance Summary" in line for line in lines)
    groups: List[List[str]] = []
    current_lines: List[str] = []

    for line in lines:
        boundary = (
            bool(RUN_HEADER_RE.match(line)) if has_headers else
            bool(ARCANE_BANNER_RE.match(line)) if has_banners else
            not line.strip() if not full_log else False
        )
        if boundary and current_lines:
            groups.append(current_lines)
            current_lines = []
        current_lines.append(line)
    if current_lines:
        groups.append(current_lines)

    blocks: List[Dict[str, float]] = []
    for group in groups:
        current: Dict[str, float] = {}
        hypre_pairs: List[tuple[float, float]] = []
        pending_setup: float | None = None
        saw_hypre_timer = False
        profiles = sum("PETSc Performance Summary" in line for line in group)
        if profiles > 1:
            raise ValueError("Multiple PETSc summaries in one run. Add '# procs=N' run headers.")
        for line in group:
            saw_hypre_timer |= "[Hypre-Timer]" in line
            header = RUN_HEADER_RE.match(line)
            process_count = header or PETSC_PROCS_RE.search(line)
            if process_count:
                count = int(process_count.group(1))
                if "__procs__" in current and current["__procs__"] != count:
                    raise ValueError("Process count differs between run header and PETSc summary.")
                current["__procs__"] = count
            match = TIMER_RE.search(line)
            if match:
                name = match.group(1)
                if name in current and (has_headers or full_log):
                    raise ValueError(f"Repeated timer {name!r}: separate runs or time steps explicitly.")
                current[name] = float(match.group(2))
            event = PETSC_EVENT_RE.match(line)
            if event:
                name = event.group(1)
                if name in current:
                    raise ValueError(f"Repeated {name}: use one PETSc profile/stage per run.")
                current[name] = float(event.group(2))
            solve = PETSC_SOLVE_RE.search(line)
            if solve:
                current[TIME_TO_SOLVE] = current.get(TIME_TO_SOLVE, 0.0) + float(solve.group(1))

            hypre_stage = HYPRE_STAGE_RE.search(line)
            if hypre_stage:
                stage, value = hypre_stage.group(1), float(hypre_stage.group(2))
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid Hypre {stage} time in run {len(blocks)}: {value}")
                if stage == "setup":
                    if pending_setup is not None:
                        raise ValueError(f"Run {len(blocks)}: Hypre setup without following solve")
                    pending_setup = value
                else:
                    if pending_setup is None:
                        raise ValueError(f"Run {len(blocks)}: Hypre solve without preceding setup")
                    hypre_pairs.append((pending_setup, value))
                    pending_setup = None

        if pending_setup is not None:
            raise ValueError(f"Run {len(blocks)}: final Hypre setup has no solve")
        if saw_hypre_timer and not hypre_pairs:
            raise ValueError(f"Run {len(blocks)}: Hypre timers lack a complete setup/solve pair")
        if hypre_pairs:
            setups, solves = zip(*hypre_pairs)
            current[HYPRE_SETUP] = float(np.median(setups))
            current[HYPRE_SOLVE] = float(np.median(solves))
            current[HYPRE_TOTAL] = float(np.median([s + v for s, v in hypre_pairs]))
            current["__hypre_calls__"] = len(hypre_pairs)

        if any(name != "__procs__" for name in current):
            blocks.append(current)
        elif "__procs__" in current:
            raise ValueError(f"Run with {int(current['__procs__'])} processes has no timing data.")

    if not blocks:
        raise RuntimeError(f"No ArcaneFEM/PETSc/Hypre timing data found in {path}")

    return blocks


def build_procs(
    n_blocks: int,
    start_procs: int,
    explicit_procs: Sequence[int] | None,
) -> np.ndarray:
    """Build process counts in simulation execution order: highest to lowest."""
    if explicit_procs:
        if len(explicit_procs) != n_blocks:
            raise ValueError(
                f"--procs contains {len(explicit_procs)} values, "
                f"but {n_blocks} blocks were parsed"
            )
        procs = np.array(explicit_procs, dtype=int)
    else:
        if start_procs <= 0:
            raise ValueError("--start-procs must be positive")

        divisor = 2 ** (n_blocks - 1)
        if start_procs % divisor != 0:
            raise ValueError(
                f"--start-procs={start_procs} cannot be halved "
                f"{n_blocks - 1} times exactly"
            )

        procs = np.array(
            [start_procs // (2 ** i) for i in range(n_blocks)],
            dtype=int,
        )

    if np.any(procs <= 0):
        raise ValueError("All process counts must be positive")

    if len(np.unique(procs)) != len(procs):
        raise ValueError("Process counts must be unique")

    return procs


def get_timer_values(blocks: List[Dict[str, float]], timer: str, initial_guess: bool = False) -> np.ndarray:
    """
    Return one value per block for a timer.

    For solve-linear-system-total:
      - without initial guess: use solve-linear-system
      - with initial guess:
            solve-linear-system-1 + solve-linear-system-2

    Missing values become NaN.
    """
    values = []

    for block in blocks:
        if timer == "solve-linear-system-total":
            if initial_guess:
                t1 = block.get("solve-linear-system-1", np.nan)
                t2 = block.get("solve-linear-system-2", np.nan)

                if np.isnan(t1) or np.isnan(t2):
                    values.append(np.nan)
                else:
                    values.append(t1 + t2)
            else:
                t0 = block.get("solve-linear-system", np.nan)
                values.append(t0)
        else:
            values.append(block.get(timer, np.nan))

    return np.array(values, dtype=float)


def strong_speedup(procs: np.ndarray, times: np.ndarray) -> np.ndarray:
    ref_idx = np.argmin(procs)
    ref_time = times[ref_idx]

    return ref_time / times


def strong_efficiency(procs: np.ndarray, times: np.ndarray) -> np.ndarray:
    ref_idx = np.argmin(procs)
    ref_procs = procs[ref_idx]

    speedup = strong_speedup(procs, times)

    return speedup / (procs / ref_procs)


def weak_efficiency(procs: np.ndarray, times: np.ndarray) -> np.ndarray:
    ref_idx = np.argmin(procs)
    ref_time = times[ref_idx]

    return ref_time / times


def setup_axes(logx: bool, logy: bool) -> None:
    if logx:
        plt.xscale("log", base=2)
    if logy:
        plt.yscale("log", base=10)


def plot_metric(
    procs: np.ndarray,
    values_by_timer: Dict[str, np.ndarray],
    outpath: Path,
    title: str,
    ylabel: str,
    logx: bool,
    logy: bool,
    percent: bool,
    ideal: np.ndarray | None,
    ideal_label: str,
) -> None:
    plt.figure()

    for timer, values in values_by_timer.items():
        mask = ~np.isnan(values)
        if not np.any(mask):
            continue
        y = 100.0 * values if percent else values
        plt.plot(procs[mask], y[mask], marker="o", label=timer)

    if ideal is not None:
        ideal_y = 100.0 * ideal if percent else ideal
        plt.plot(procs, ideal_y, linestyle="--", marker="x", label=ideal_label)

    setup_axes(logx=logx, logy=logy)
    plt.xlabel("Number of MPI processes")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both")
    plt.xticks(procs, [str(p) for p in procs])
    plt.legend()
    plt.tight_layout()

    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[ok] wrote {outpath}")


def write_csv(
    outdir: Path,
    procs: np.ndarray,
    timers: List[str],
    times_by_timer: Dict[str, np.ndarray],
    blocks: List[Dict[str, float]],
    is_hypre: bool,
) -> None:
    csv_path = outdir / "scaling_metrics.csv"

    fieldnames = ["block", "procs"]
    if is_hypre:
        fieldnames.append("hypre_calls")
    for timer in timers:
        fieldnames += [
            f"{timer}:time",
            f"{timer}:strong_speedup",
            f"{timer}:strong_efficiency",
            f"{timer}:strong_efficiency_percent",
            f"{timer}:weak_efficiency",
            f"{timer}:weak_efficiency_percent",
        ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, p in enumerate(procs):
            row = {"block": i, "procs": int(p)}
            if is_hypre:
                row["hypre_calls"] = int(blocks[i]["__hypre_calls__"])

            for timer in timers:
                times = times_by_timer[timer]
                t = times[i]
                ss = strong_speedup(procs, times)[i]
                se = strong_efficiency(procs, times)[i]
                we = weak_efficiency(procs, times)[i]

                row[f"{timer}:time"] = "" if np.isnan(t) else t
                row[f"{timer}:strong_speedup"] = "" if np.isnan(ss) else ss
                row[f"{timer}:strong_efficiency"] = "" if np.isnan(se) else se
                row[f"{timer}:strong_efficiency_percent"] = "" if np.isnan(se) else 100.0 * se
                row[f"{timer}:weak_efficiency"] = "" if np.isnan(we) else we
                row[f"{timer}:weak_efficiency_percent"] = "" if np.isnan(we) else 100.0 * we

            writer.writerow(row)

    print(f"[ok] wrote {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse ArcaneFem-Timer logs and plot strong/weak scaling metrics."
    )

    parser.add_argument("input_file", type=Path)
    parser.add_argument("--mode", choices=["strong", "weak", "none"], default="none")
    parser.add_argument("--start-procs", type=int, default=16)
    parser.add_argument("--procs", nargs="+", type=int, default=None)
    parser.add_argument("--timers", nargs="+", default=DEFAULT_TIMERS)
    parser.add_argument("--outdir", type=Path, default=Path("../plots/plots"))
    parser.add_argument("--logx", action="store_true")
    parser.add_argument("--logy", action="store_true")
    parser.add_argument(
        "--include-assembly",
        action="store_true",
        help="Add LHS and RHS assembly curves to the strong/weak scaling plots.",
    )
    parser.add_argument(
        "--solver-breakdown",
        action="store_true",
        help=("PETSc only: add PCSetUp, KSPSolve and PCApply to the plots and CSV. "
              "Hypre setup and solve are always included."),
    )
    parser.add_argument("--initial-guess", action="store_true", 
        help=("Use solve-linear-system-1 + solve-linear-system-2 "
        "instead of solve-linear-system."))

    args = parser.parse_args()

    try:
        blocks = parse_blocks(args.input_file)
        hypre_blocks = [HYPRE_TOTAL in block for block in blocks]
        if any(hypre_blocks) and not all(hypre_blocks):
            raise ValueError("Some runs lack Hypre setup/solve times; check for mixed or incomplete logs.")
        is_hypre = all(hypre_blocks)
        if is_hypre and args.initial_guess:
            raise ValueError("--initial-guess applies only to the old ArcaneFEM/PETSc format.")
        recorded_procs = [block.get("__procs__") for block in blocks]
        explicit_procs = args.procs
        if explicit_procs is None and all(p is not None for p in recorded_procs):
            explicit_procs = [int(p) for p in recorded_procs]
        elif explicit_procs is None and any(p is not None for p in recorded_procs):
            raise ValueError("Some runs lack process counts. Supply --procs in execution order.")
        procs = build_procs(len(blocks), args.start_procs, explicit_procs)
        for index, (recorded, assigned) in enumerate(zip(recorded_procs, procs)):
            if recorded is not None and recorded != assigned:
                raise ValueError(
                    f"Block {index}: log reports {int(recorded)} processes, "
                    f"but --procs assigns {assigned}. Check run order and warm-up output."
                )
        if args.solver_breakdown and not is_hypre:
            for index, block in enumerate(blocks):
                missing = [name for name in [TIME_TO_SOLVE, *PETSC_EVENTS]
                           if name not in block or not np.isfinite(block[name]) or block[name] < 0]
                if missing:
                    raise ValueError(
                        f"Block {index} (procs={procs[index]}): missing/invalid {', '.join(missing)}. "
                        "Use the full log with [Petsc-Timer] Time to solve and ASCII -log_view; "
                        "StrongScaling.txt filtered to ArcaneFem-Timer does not contain these events."
                    )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    timers = list(dict.fromkeys(args.timers if args.timers != DEFAULT_TIMERS or not is_hypre else
                                ["lhs-matrix-assembly", "rhs-vector-assembly-gpu", *HYPRE_TIMERS, "compute"]))
    if is_hypre:
        timers = list(dict.fromkeys([*timers, *HYPRE_TIMERS]))
    elif args.solver_breakdown:
        timers = list(dict.fromkeys([*timers, TIME_TO_SOLVE, *PETSC_EVENTS]))

    args.outdir.mkdir(parents=True, exist_ok=True)

    times_by_timer = {
        timer: get_timer_values(blocks, timer, initial_guess=args.initial_guess,)
        for timer in timers
    }

    print(f"[info] parsed {len(blocks)} blocks")
    print(f"[info] procs = {list(map(int, procs))}")
    print(f"[info] timers = {timers}")
    if is_hypre:
        print(f"[info] Hypre calls per block = {[int(block['__hypre_calls__']) for block in blocks]}")
        print("[info] hypre-total = median of (setup + solve) for each paired call")

    write_csv(args.outdir, procs, timers, times_by_timer, blocks, is_hypre)


    if args.solver_breakdown and not is_hypre:
        print("[info] PETSc events use Time (sec) Max and may overlap; do not sum them.")

    if args.mode == "none":
        print("[info] CSV only. Use --mode strong or --mode weak for scaling plots.")

    scaling_timers = HYPRE_TIMERS.copy() if is_hypre else ["solve-linear-system-total"]
    if args.include_assembly:
        scaling_timers += ["lhs-matrix-assembly", "rhs-vector-assembly-gpu"]
    if args.solver_breakdown and not is_hypre:
        scaling_timers += PETSC_EVENTS

    scaling_times = {
        timer: get_timer_values(blocks, timer, initial_guess=args.initial_guess)
        for timer in scaling_timers
    }

    if args.mode == "strong":
        speedup_by_timer = {
            timer: strong_speedup(procs, times)
            for timer, times in scaling_times.items()
        }
        efficiency_by_timer = {
            timer: strong_efficiency(procs, times)
            for timer, times in scaling_times.items()
        }

        plot_metric(
            procs,
            speedup_by_timer,
            args.outdir / "strong_speedup.png",
            "Strong scaling speedup",
            "Speedup",
            args.logx,
            args.logy,
            percent=False,
            ideal=procs / np.min(procs),
            ideal_label="Ideal speedup",
        )

        plot_metric(
            procs,
            efficiency_by_timer,
            args.outdir / "strong_efficiency.png",
            "Strong scaling efficiency",
            "Efficiency (%)",
            args.logx,
            False,
            percent=True,
            ideal=np.ones_like(procs, dtype=float),
            ideal_label="Ideal efficiency",
        )

    if args.mode == "weak":
        weak_eff_by_timer = {
            timer: weak_efficiency(procs, times)
            for timer, times in scaling_times.items()
        }

        plot_metric(
            procs,
            weak_eff_by_timer,
            args.outdir / "weak_efficiency.png",
            "Weak scaling efficiency",
            "Weak efficiency (%)",
            args.logx,
            False,
            percent=True,
            ideal=np.ones_like(procs, dtype=float),
            ideal_label="Ideal weak efficiency",
        )


if __name__ == "__main__":
    main()