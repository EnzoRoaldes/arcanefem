#!/usr/bin/env bash

# Usage depuis Fourier/my_cases/scripts/:
#   ./strong_scaling_simple.sh JOB_ID [--gpu]

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
PC="gamg"
SOLVER="cg"
AMG_THRESHOLD=0.01

# Format: "number_of_processes partitions_x partitions_y"
CASES=(
  "192 16 12"
  "128 16 8"
  "96 12 8"
  "64 8 8"
  # "48 8 6"
  "32 8 4"
  # "24 6 4"
  "16 4 4"
  # "12 4 3"
  "8 4 2"
  "4 2 2"
  "2 2 1"
)

if [[ "$MODE" == "gpu" ]]; then
  CASES=(
    "32 8 4"
    "16 4 4"
    "8 4 2"
    "4 2 2"
    "2 2 1"
  )
fi

SRUN_FLAGS=()
ARCANE_FLAGS=()
PETSC_DEVICE_FLAGS=""
if [[ "$MODE" == "gpu" ]]; then
  SRUN_FLAGS=(--gpus-per-task=1 --gpu-bind=single:1)
  ARCANE_FLAGS=(-A,AcceleratorRuntime=hip)
  PETSC_DEVICE_FLAGS="-mat_type aijhipsparse -vec_type hip"
  export MPICH_GPU_SUPPORT_ENABLED=1
fi

PC_FLAGS=""
if [[ "$PC" == "gamg" ]]; then
  PC_FLAGS="-pc_gamg_threshold $AMG_THRESHOLD"
fi

# ----------------------------- Paths -------------------------------

# The script must be launched from Fourier/my_cases/scripts/.
# Move to the Fourier root once, then use absolute paths everywhere.
cd ../..
FOURIER_DIR="$PWD"

ARC_FILE="$FOURIER_DIR/my_cases/inputs/conduction.center-source.arc"
OUTPUT_DIR="$FOURIER_DIR/my_cases/outputs"
if [[ "$MODE" == "gpu" ]]; then
  OUTPUT_DIR="$OUTPUT_DIR/gpu"
fi
LOG_FILE="$OUTPUT_DIR/logs_SS.txt"
TIMER_FILE="$OUTPUT_DIR/StrongScaling.txt"

# --------------------------- Environment ---------------------------

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
  local extra_petsc_flags="${5:-}"

  srun \
    --jobid="$JOB_ID" \
    --ntasks="$nb_procs" \
    --cpus-per-task=1 \
    --threads-per-core=1 \
    --distribution=block:block \
    --cpu-bind=verbose,cores \
    "${SRUN_FLAGS[@]}" \
    ./Fourier "${ARCANE_FLAGS[@]}" "$ARC_FILE" \
    -A,//fem/petsc-flags="-ksp_monitor $extra_petsc_flags \
    -ksp_max_it $MAX_IT -ksp_rtol $R_TOL \
    -ksp_view -pc_type $PC $PC_FLAGS $PETSC_DEVICE_FLAGS \
    -ksp_type $SOLVER -ksp_initial_guess_nonzero" \
    -A,//meshes/mesh/generator/nb-part-x="$part_x" \
    -A,//meshes/mesh/generator/nb-part-y="$part_y" \
    -A,//meshes/mesh/generator/x/n="$mesh_size" \
    -A,//meshes/mesh/generator/x/length=1.0 \
    -A,//meshes/mesh/generator/y/n="$mesh_size" \
    -A,//meshes/mesh/generator/y/length=1.0 \
    -A,//fem/linear-system/@name=PetscLinearSystem
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
    "$NB_PROCS" "$PART_X" "$PART_Y" >> "$TIMER_FILE"

  run_fourier "$NB_PROCS" "$PART_X" "$PART_Y" "$N" "-log_view" \
    2>&1 |
    tee -a "$LOG_FILE" |
    awk '/ArcaneFem-Timer/' >> "$TIMER_FILE"

  printf '\n' >> "$TIMER_FILE"
done

echo
echo "Benchmark completed."
echo "Full log:   $LOG_FILE"
echo "Timer data: $TIMER_FILE"










# #!/bin/bash

# # Usage: ./inter_strong_scaling.sh JOB_ID

# JOB_ID="${1:?Usage: $0 JOB_ID}"

# # ----------------------- Parameters to modify -----------------------

# N=7000
# MAX_IT=5000
# R_TOL=1e-10
# PC="bjacobi"
# SOLVER="cg"
# AMG_THRESHOLD=0.01

# # Format: "number_of_processes partitions_x partitions_y".
# CASES=(
#   "192 16 12"
#   # "96 12 8"
#   "48 8 6"
#   # "24 6 4"
#   "12 4 3"
#   # "8 4 2"
#   "4 2 2"
#   # "2 2 1"
#   "1 1 1"
# )

# # ----------------------------- Paths -------------------------------

# # The script must be launched from Fourier/my_cases/scripts/.
# # Move to the Fourier root once, then use absolute paths everywhere.
# cd ../..
# FOURIER_DIR="$PWD"

# ARC_FILE="$FOURIER_DIR/my_cases/inputs/conduction.center-source.arc"
# OUTPUT_DIR="$FOURIER_DIR/my_cases/outputs"
# LOG_FILE="$OUTPUT_DIR/logs_SS.txt"
# TIMER_FILE="$OUTPUT_DIR/StrongScaling.txt"

# # --------------------------- Environment ---------------------------

# module load craype-x86-trento
# module load craype-accel-amd-gfx90a
# module load PrgEnv-amd
# module load cmake/3.27.9
# module load rocm/6.4.3
# module load cray-hdf5-parallel/1.14.3.7

# cd "$FOURIER_DIR"

# mkdir -p "$OUTPUT_DIR"

# if [[ ! -x "$FOURIER_DIR/Fourier" ]]; then
#   echo "Error: executable not found: $FOURIER_DIR/Fourier" >&2
#   exit 1
# fi

# if [[ ! -f "$ARC_FILE" ]]; then
#   echo "Error: input file not found: $ARC_FILE" >&2
#   exit 1
# fi

# # Empty the output files before starting the benchmark.
# : > "$LOG_FILE"
# : > "$TIMER_FILE"

# # ------------------------- Fourier command -------------------------

# run_fourier()
# {
#   local nb_procs="$1"
#   local part_x="$2"
#   local part_y="$3"
#   local cells="$4"
#   local extra_petsc_flags="${5:-}"

#   srun \
#     --jobid="$JOB_ID" \
#     --ntasks="$nb_procs" \
#     --cpus-per-task=1 \
#     ./Fourier "$ARC_FILE" \
#     -A,//fem/petsc-flags="-ksp_monitor $extra_petsc_flags \
#     -ksp_max_it $MAX_IT -ksp_rtol $R_TOL \
#     -ksp_view -pc_type $PC -pc_gamg_threshold $AMG_THRESHOLD \
#     -ksp_type $SOLVER -ksp_initial_guess_nonzero" \
#     -A,//meshes/mesh/generator/nb-part-x="$part_x" \
#     -A,//meshes/mesh/generator/nb-part-y="$part_y" \
#     -A,//meshes/mesh/generator/x/n="$cells" \
#     -A,//meshes/mesh/generator/x/length=1.0 \
#     -A,//meshes/mesh/generator/y/n="$cells" \
#     -A,//meshes/mesh/generator/y/length=1.0 \
#     -A,//fem/linear-system/@name=PetscLinearSystem
# }

# # ------------------------------ Warm-up ----------------------------

# echo "========== Warm-up =========="

# # Use the first configuration with a smaller mesh.
# read -r WARMUP_PROCS WARMUP_X WARMUP_Y <<< "${CASES[0]}"
# run_fourier "$WARMUP_PROCS" "$WARMUP_X" "$WARMUP_Y" 1000 "-log_view" 2>&1 | tee -a "$LOG_FILE"
# # run_fourier "$WARMUP_PROCS" "$WARMUP_X" "$WARMUP_Y" 1000 >/dev/null 2>&1

# # ----------------------------- Benchmark ---------------------------

# echo "========== Strong scaling =========="

# for case_config in "${CASES[@]}"; do
#   read -r NB_PROCS PART_X PART_Y <<< "$case_config"

#   if (( PART_X * PART_Y != NB_PROCS )); then
#     echo "Error: $PART_X x $PART_Y != $NB_PROCS processes" >&2
#     exit 1
#   fi

#   echo "Running $NB_PROCS processes ($PART_X x $PART_Y)"

#   printf '# procs=%s part_x=%s part_y=%s\n' \
#     "$NB_PROCS" "$PART_X" "$PART_Y" >> "$TIMER_FILE"

#   run_fourier "$NB_PROCS" "$PART_X" "$PART_Y" "$N" "-log_view" \
#     2>&1 |
#     tee -a "$LOG_FILE" |
#     awk '/ArcaneFem-Timer/' >> "$TIMER_FILE"

#   printf '\n' >> "$TIMER_FILE"
# done

# echo
# echo "Benchmark completed."
# echo "Full log:   $LOG_FILE"
# echo "Timer data: $TIMER_FILE"