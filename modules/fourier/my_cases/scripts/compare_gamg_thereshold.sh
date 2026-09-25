#!/usr/bin/env bash

# Compare PETSc GAMG thresholds for one fixed MPI configuration.
# Usage: ./compare_gamg_thresholds.sh JOB_ID

set -euo pipefail

JOB_ID="${1:?Usage: $0 JOB_ID}"

# MPI decomposition: PART_X * PART_Y must equal NB_PROCS.
NB_PROCS=24
PART_X=6
PART_Y=4

# Mesh and solver parameters.
MESH_N=2000
MESH_M=2000
MAX_IT=200
R_TOL=1e-8
PC="gamg"
SOLVER="cg"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOURIER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

ARC_FILE="$FOURIER_DIR/my_cases/inputs/conduction.heterogeneous.arc"
OUTPUT_DIR="$FOURIER_DIR/my_cases/outputs/gamg_threshold_N${NB_PROCS}"
RESULTS_FILE="$OUTPUT_DIR/results.csv"

module load craype-x86-trento
module load craype-accel-amd-gfx90a
module load PrgEnv-amd
module load cmake/3.27.9
module load rocm/6.4.3
module load cray-hdf5-parallel/1.14.3.7

if (( PART_X * PART_Y != NB_PROCS )); then
  echo "Error: PART_X * PART_Y must equal NB_PROCS" >&2
  echo "Current values: $PART_X * $PART_Y != $NB_PROCS" >&2
  exit 1
fi

if [[ ! -x "$FOURIER_DIR/Fourier" ]]; then
  echo "Error: executable not found: $FOURIER_DIR/Fourier" >&2
  exit 1
fi

if [[ ! -f "$ARC_FILE" ]]; then
  echo "Error: Arcane case file not found: $ARC_FILE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$FOURIER_DIR"

printf '%s\n' \
  "threshold,nx,ny,mpi_processes,wall_time_s,compute_s,solve_linear_system_s,grid_complexity,operator_complexity,iterations,exit_code" \
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

# Extract the last PETSc GAMG grid and operator complexity values.
complexity_values()
{
  local log_file="$1"

  awk '
    /Complexity:/ {
      for (i = 1; i <= NF; ++i) {
        if ($i == "grid" && $(i + 1) == "=")
          grid = $(i + 2)
        if ($i == "operator" && $(i + 1) == "=")
          operator = $(i + 2)
      }
    }

    END {
      printf "%s %s\n", grid, operator
    }
  ' "$log_file"
}

# Integer loop avoids floating-point accumulation errors in Bash.
for threshold_index in {0..30..2}; do
  printf -v AMG_THRESHOLD '0.0%02d' "$threshold_index"

  LOG_FILE="$OUTPUT_DIR/threshold_${AMG_THRESHOLD}.log"

  echo "=================================================="
  echo "GAMG threshold=$AMG_THRESHOLD"
  echo "MPI processes=$NB_PROCS, nx=$PART_X, ny=$PART_Y"
  echo "=================================================="

  START_NS="$(date +%s%N)"

  # Save the application return code without stopping the threshold sweep.
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
    -A,//meshes/mesh/generator/nb-part-x="$PART_X" \
    -A,//meshes/mesh/generator/nb-part-y="$PART_Y" \
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

  COMPLEXITY_VALUES="$(complexity_values "$LOG_FILE")"
  read -r GRID_COMPLEXITY OPERATOR_COMPLEXITY <<< "$COMPLEXITY_VALUES"

  mapfile -t ITERATIONS < <(
    grep -Eo 'iterations[[:space:]]+[0-9]+' "$LOG_FILE" |
      awk '{print $2}'
  )

  ITERATION_COUNT="${ITERATIONS[0]:-}"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$AMG_THRESHOLD" \
    "$PART_X" \
    "$PART_Y" \
    "$NB_PROCS" \
    "$WALL_TIME" \
    "$COMPUTE_TIME" \
    "$SOLVE_TIME" \
    "$GRID_COMPLEXITY" \
    "$OPERATOR_COMPLEXITY" \
    "$ITERATION_COUNT" \
    "$RUN_STATUS" \
    >> "$RESULTS_FILE"

  echo
  echo "Result:"
  echo "  threshold            = $AMG_THRESHOLD"
  echo "  wall time            = ${WALL_TIME} s"
  echo "  compute time         = ${COMPUTE_TIME:-NA} s"
  echo "  linear solve time    = ${SOLVE_TIME:-NA} s"
  echo "  grid complexity      = ${GRID_COMPLEXITY:-NA}"
  echo "  operator complexity  = ${OPERATOR_COMPLEXITY:-NA}"
  echo "  iterations           = ${ITERATION_COUNT:-NA}"
  echo "  exit code            = $RUN_STATUS"
  echo
done

echo "Comparison written to:"
echo "$RESULTS_FILE"
echo

column -s, -t "$RESULTS_FILE" 2>/dev/null || cat "$RESULTS_FILE"
