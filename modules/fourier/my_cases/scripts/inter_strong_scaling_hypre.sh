#!/usr/bin/env bash

# Usage depuis Fourier/my_cases/scripts/:
#   ./inter_strong_scaling_hypre.sh JOB_ID [--gpu]

set -euo pipefail

JOB_ID="${1:?Usage: $0 JOB_ID [--gpu]}"
MODE="cpu"
case "${2:-}" in
  "") ;;
  --gpu) MODE="gpu" ;;
  *) echo "Usage: $0 JOB_ID [--gpu]" >&2; exit 1 ;;
esac
if (( $# > 2 )); then
  echo "Usage: $0 JOB_ID [--gpu]" >&2
  exit 1
fi

# ----------------------- Parameters to modify -----------------------

N=7000
MAX_IT=500
R_TOL=1e-10
PC="amg"
SOLVER="cg"

AMG_THRESHOLD=0.25
AMG_COARSENER=8 #PMIS
AMG_INTERP_TYPE=6 #extendei
AMG_SMOOTHER=6 #hybrid symmetric GS or SSOR 13 et 14 pour l1-GS et back l1-GS

# Format: "number_of_processes partitions_x partitions_y"
CASES=(
  "256 16 16"
  "128 16 8"
  "64 8 8"
  "32 8 4"
  "16 4 4"
  "8 4 2"
  "4 2 2"
  "2 2 1"
)

if [[ "$MODE" == "gpu" ]]; then
  CASES=(
    # "32 8 4"
    # "16 4 4"
    # "8 4 2"
    # "4 2 2"
    # "2 2 1"
    "1 1 1"
  )
fi

SRUN_FLAGS=()
ARCANE_FLAGS=()
if [[ "$MODE" == "gpu" ]]; then
  SRUN_FLAGS=(--gpus-per-task=1 --gpu-bind=single:1 --ntasks-per-node=8 --cpus-per-task=8)
  ARCANE_FLAGS=(-A,AcceleratorRuntime=hip -A,//fem/matrix-format=AF-BSR)
  export MPICH_GPU_SUPPORT_ENABLED=1
  export HSA_XNACK=1
  export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
fi


OUTPUT_FLAGS=(-A,//arcane-post-processing/output-period=0 -A,//arcane-post-processing/output-frequency=0 -A,//arcane-post-processing/save-init=false -A,//arcane-post-processing/save-final-time=false -A,//arcane-post-processing/end-execution-output=false)

# ----------------------------- Paths -------------------------------

# The script must be launched from Fourier/my_cases/scripts/.
# Move to the Fourier root once, then use absolute paths everywhere.
cd ../..
FOURIER_DIR="$PWD"

ARC_FILE="$FOURIER_DIR/my_cases/inputs/conduction.center-source.arc"
OUTPUT_DIR="$FOURIER_DIR/my_cases/outputs/$MODE"
LOG_FILE="$OUTPUT_DIR/logs_SS.txt"
TIMER_FILE="$OUTPUT_DIR/StrongScaling.txt"

# --------------------------- Environment ---------------------------

module load craype-x86-trento
module load craype-accel-amd-gfx90a
module load PrgEnv-amd #
module load cmake/3.27.9 #
module load rocm/6.4.3 # (différent)
module load cray-hdf5-parallel/1.14.3.7 #

mkdir -p "$OUTPUT_DIR"

if [[ ! -x "$FOURIER_DIR/Fourier" ]]; then
  echo "Error: executable not found: $FOURIER_DIR/Fourier" >&2
  exit 1
fi

if [[ ! -f "$ARC_FILE" ]]; then
  echo "Error: input file not found: $ARC_FILE" >&2
  exit 1
fi

# Empty the output files before starting the benchmark.
for case_config in "${CASES[@]}"; do
  read -r NB_PROCS PART_X PART_Y <<< "$case_config"
  if (( NB_PROCS <= 0 || PART_X <= 0 || PART_Y <= 0 || PART_X * PART_Y != NB_PROCS )); then
    echo "Invalid configuration: $case_config" >&2
    exit 1
  fi
done
echo "Mode=$MODE, PC=$PC, KSP=$SOLVER"
: > "$LOG_FILE"
: > "$TIMER_FILE"

# ------------------------- Fourier command -------------------------

run_fourier()
{
  local nb_procs="$1"
  local part_x="$2"
  local part_y="$3"
  local mesh_size="$4"

  srun \
    --jobid="$JOB_ID" \
    --ntasks="$nb_procs" \
    --threads-per-core=1 \
    --distribution=block:block \
    --cpu-bind=verbose,cores \
    "${SRUN_FLAGS[@]}" \
    ./Fourier "${ARCANE_FLAGS[@]}" "$ARC_FILE" \
    "${OUTPUT_FLAGS[@]}" \
    -A,//fem/linear-system/@name=HypreLinearSystem \
    -A,//fem/linear-system/solver=$SOLVER \
    -A,//fem/linear-system/preconditioner=$PC \
    -A,//fem/linear-system/rtol=$R_TOL \
    -A,//fem/linear-system/atol=0 \
    -A,//fem/linear-system/max-iter=$MAX_IT \
    -A,//fem/linear-system/verbosity=2 \
    -A,//fem/linear-system/amg-threshold="$AMG_THRESHOLD" \
    -A,//fem/linear-system/amg-coarsener="$AMG_COARSENER" \
    -A,//fem/linear-system/amg-interp-type="$AMG_INTERP_TYPE" \
    -A,//fem/linear-system/amg-smoother="$AMG_SMOOTHER" \
    -A,//fem/linear-system/amg-near-null-space=false \
    -A,//meshes/mesh/generator/nb-part-x="$part_x" \
    -A,//meshes/mesh/generator/nb-part-y="$part_y" \
    -A,//meshes/mesh/generator/x/n="$mesh_size" \
    -A,//meshes/mesh/generator/x/length=1.0 \
    -A,//meshes/mesh/generator/y/n="$mesh_size" \
    -A,//meshes/mesh/generator/y/length=1.0
}


# ------------------------------ Warm-up ----------------------------

echo "========== Warm-up =========="

# Use the first configuration with a smaller mesh.
read -r WARMUP_PROCS WARMUP_X WARMUP_Y <<< "${CASES[0]}"
if run_fourier "$WARMUP_PROCS" "$WARMUP_X" "$WARMUP_Y" 1000 \
    >"$OUTPUT_DIR/warmup.log" 2>&1; then
  echo "Warm-up completed."
else
  STATUS=$?
  tail -n 40 "$OUTPUT_DIR/warmup.log" >&2
  echo "Warm-up failed. Full log: $OUTPUT_DIR/warmup.log" >&2
  exit "$STATUS"
fi

# ----------------------------- Benchmark ---------------------------

echo "========== Strong scaling =========="

for case_config in "${CASES[@]}"; do
  read -r NB_PROCS PART_X PART_Y <<< "$case_config"

  if (( PART_X * PART_Y != NB_PROCS )); then
    echo "Error: $PART_X x $PART_Y != $NB_PROCS processes" >&2
    exit 1
  fi

  echo "Running $NB_PROCS processes ($PART_X x $PART_Y)"

  printf '# procs=%s part_x=%s part_y=%s\n' \
    "$NB_PROCS" "$PART_X" "$PART_Y" |
    tee -a "$LOG_FILE" >> "$TIMER_FILE"

  run_fourier "$NB_PROCS" "$PART_X" "$PART_Y" "$N" \
    2>&1 |
    tee -a "$LOG_FILE" |
    awk '
      /\[ArcaneFem-Timer\]/ ||
      /\[Hypre-Timer\]/ ||
      /\[Hypre-Info\][[:space:]]+Converged in[[:space:]]+[0-9]+[[:space:]]+iterations/
    ' >> "$TIMER_FILE"

  printf '\n' >> "$TIMER_FILE"
  printf '\n' >> "$LOG_FILE"
done

echo
echo "Benchmark completed."
echo "Full log:   $LOG_FILE"
echo "Timer data: $TIMER_FILE"