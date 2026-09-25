#!/usr/bin/env bash

# Compare Cartesian MPI decompositions using the median of three runs.
# Usage: ./compare_partition_xy_median.sh JOB_ID

set -euo pipefail

JOB_ID="${1:?Usage: $0 JOB_ID}"
NB_PROCS=64
REPETITIONS=3

# Mesh and solver parameters.
MESH_N=5000
MESH_M=5000
MAX_IT=500
R_TOL=1e-10
PC="bjacobi"
SOLVER="cg"
AMG_THRESHOLD=0.01

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOURIER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

ARC_FILE="$FOURIER_DIR/my_cases/inputs/conduction.center-source.arc"
OUTPUT_DIR="$FOURIER_DIR/my_cases/outputs/partition_xy_N${NB_PROCS}_median"
RESULTS_FILE="$OUTPUT_DIR/results.csv"

PARTITIONS=(
  "1 64"
  # "2 32"
  # "4 16"
  # "8 8"
  # "16 4"
  # "32 2"
  # "64 1"
)

module load craype-x86-trento
module load craype-accel-amd-gfx90a
module load PrgEnv-amd
module load cmake/3.27.9
module load rocm/6.4.3
module load cray-hdf5-parallel/1.14.3.7

mkdir -p "$OUTPUT_DIR"

if [[ ! -x "$FOURIER_DIR/Fourier" ]]; then
  echo "Error: executable not found: $FOURIER_DIR/Fourier" >&2
  exit 1
fi

if [[ ! -f "$ARC_FILE" ]]; then
  echo "Error: Arcane case file not found: $ARC_FILE" >&2
  exit 1
fi

cd "$FOURIER_DIR"

printf '%s\n' \
  "nx,ny,mpi_processes,runs,wall_time_median_s,compute_median_s,solve_linear_system_median_s,iterations_median,failed_runs" \
  > "$RESULTS_FILE"

# Extract the last value associated with an exact ArcaneFEM timer name.
timer_value()
{
  local timer_name="$1"
  local log_file="$2"

  awk -v wanted="$timer_name" '
    match($0, /\[ArcaneFem-Timer\][[:space:]]+[^[:space:]=]+/) {
      name = substr($0, RSTART, RLENGTH)
      sub(/^.*\][[:space:]]+/, "", name)

      if (name == wanted) {
        split($0, fields, "=")
        value = fields[length(fields)]
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
        last = value
      }
    }

    END {
      if (last != "")
        print last
    }
  ' "$log_file"
}

# Return the middle value of exactly three numeric measurements.
# Return an empty value if at least one measurement is missing.
median_of_three()
{
  local first="$1"
  local second="$2"
  local third="$3"

  if [[ -z "$first" || -z "$second" || -z "$third" ]]; then
    return 0
  fi

  awk -v a="$first" -v b="$second" -v c="$third" '
    BEGIN {
      if (a > b) { tmp = a; a = b; b = tmp }
      if (b > c) { tmp = b; b = c; c = tmp }
      if (a > b) { tmp = a; a = b; b = tmp }
      printf "%.15g", b
    }
  '
}

for partition in "${PARTITIONS[@]}"; do
  read -r NX NY <<< "$partition"

  if (( NX * NY != NB_PROCS )); then
    echo "Error: nx=$NX and ny=$NY do not use $NB_PROCS MPI processes" >&2
    exit 1
  fi

  WALL_TIMES=()
  COMPUTE_TIMES=()
  SOLVE_TIMES=()
  ITERATION_COUNTS=()
  FAILED_RUNS=0

  echo "=================================================="
  echo "nx=$NX, ny=$NY, MPI processes=$NB_PROCS"
  echo "=================================================="

  for (( RUN=1; RUN<=REPETITIONS; ++RUN )); do
    LOG_FILE="$OUTPUT_DIR/nx${NX}_ny${NY}_run${RUN}.log"

    echo "--- Run $RUN/$REPETITIONS ---"

    START_NS="$(date +%s%N)"

    # Temporarily disable immediate exit to record failed executions.
    set +e

    srun \
      --jobid="$JOB_ID" \
      --ntasks="$NB_PROCS" \
      --cpus-per-task=1 \
      ./Fourier "$ARC_FILE" \
      -A,//fem/petsc-flags="-ksp_monitor -ksp_converged_reason \
      -ksp_max_it $MAX_IT -ksp_rtol $R_TOL -ksp_view \
      -pc_type $PC -pc_gamg_threshold $AMG_THRESHOLD \
      -ksp_type $SOLVER -ksp_initial_guess_nonzero" \
      -A,//meshes/mesh/generator/nb-part-x="$NX" \
      -A,//meshes/mesh/generator/nb-part-y="$NY" \
      -A,//meshes/mesh/generator/x/n="$MESH_N" \
      -A,//meshes/mesh/generator/x/length=1.0 \
      -A,//meshes/mesh/generator/y/n="$MESH_M" \
      -A,//meshes/mesh/generator/y/length=1.0 \
      -A,//fem/linear-system/@name=PetscLinearSystem \
      2>&1 | tee "$LOG_FILE"

    RUN_STATUS=${PIPESTATUS[0]}
    set -e

    END_NS="$(date +%s%N)"

    WALL_TIME="$(
      awk -v start="$START_NS" -v end="$END_NS" \
        'BEGIN { printf "%.6f", (end - start) / 1000000000 }'
    )"

    COMPUTE_TIME="$(timer_value "compute" "$LOG_FILE")"
    SOLVE_TIME="$(timer_value "solve-linear-system" "$LOG_FILE")"

    mapfile -t ITERATIONS < <(
      grep -Eo 'iterations[[:space:]]+[0-9]+' "$LOG_FILE" |
        awk '{print $2}'
    )
    ITERATION_COUNT="${ITERATIONS[0]:-}"

    WALL_TIMES+=("$WALL_TIME")
    COMPUTE_TIMES+=("$COMPUTE_TIME")
    SOLVE_TIMES+=("$SOLVE_TIME")
    ITERATION_COUNTS+=("$ITERATION_COUNT")

    if (( RUN_STATUS != 0 )); then
      FAILED_RUNS=$((FAILED_RUNS + 1))
    fi

    echo "Run $RUN: wall=${WALL_TIME}s, compute=${COMPUTE_TIME:-NA}s, solve=${SOLVE_TIME:-NA}s, iterations=${ITERATION_COUNT:-NA}, exit=$RUN_STATUS"
    echo
  done

  WALL_MEDIAN="$(median_of_three "${WALL_TIMES[@]}")"
  COMPUTE_MEDIAN="$(median_of_three "${COMPUTE_TIMES[@]}")"
  SOLVE_MEDIAN="$(median_of_three "${SOLVE_TIMES[@]}")"
  ITERATIONS_MEDIAN="$(median_of_three "${ITERATION_COUNTS[@]}")"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$NX" \
    "$NY" \
    "$NB_PROCS" \
    "$REPETITIONS" \
    "$WALL_MEDIAN" \
    "$COMPUTE_MEDIAN" \
    "$SOLVE_MEDIAN" \
    "$ITERATIONS_MEDIAN" \
    "$FAILED_RUNS" \
    >> "$RESULTS_FILE"

  echo "Median for nx=$NX, ny=$NY:"
  echo "  wall time            = ${WALL_MEDIAN:-NA} s"
  echo "  compute time         = ${COMPUTE_MEDIAN:-NA} s"
  echo "  linear solve time    = ${SOLVE_MEDIAN:-NA} s"
  echo "  iterations           = ${ITERATIONS_MEDIAN:-NA}"
  echo "  failed runs          = $FAILED_RUNS"
  echo
done

echo "Comparison written to:"
echo "$RESULTS_FILE"
echo

column -s, -t "$RESULTS_FILE" 2>/dev/null || cat "$RESULTS_FILE"
