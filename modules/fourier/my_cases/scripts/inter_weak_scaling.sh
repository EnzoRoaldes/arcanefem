#!/usr/bin/env bash

# Depuis Fourier/my_cases/scripts/ :
#   bash inter_weak_scaling_hypre.sh JOB_ID [--gpu]

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

# Paramètres
N_LOCAL=1000
M_LOCAL=1000
LOCAL_LENGTH_X=1.0
LOCAL_LENGTH_Y=1.0
WARMUP_SIZE=1000

MAX_IT=500
R_TOL=1e-10
PC="amg"
SOLVER="cg"
AMG_THRESHOLD=0.25
AMG_COARSENER=8
AMG_INTERP_TYPE=6
AMG_SMOOTHER=6

# Format : "rangs_MPI part_x part_y".
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

SRUN_FLAGS=(--cpus-per-task=1)
ARCANE_FLAGS=()

if [[ "$MODE" == "gpu" ]]; then
  CASES=(
    # "32 8 4"
    # "16 4 4"
    # "8 4 2"
    # "4 2 2"
    # "2 2 1"
    "1 1 1"
  )

  SRUN_FLAGS=(
    --gpus-per-task=1
    --gpu-bind=single:1
    --ntasks-per-node=8
    --cpus-per-task=8
  )
  ARCANE_FLAGS=(
    -A,AcceleratorRuntime=hip
    -A,//fem/matrix-format=AF-BSR
  )

  export MPICH_GPU_SUPPORT_ENABLED=1
  export HSA_XNACK=1
  export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
fi

OUTPUT_FLAGS=(
  -A,//arcane-post-processing/output-period=0
  -A,//arcane-post-processing/output-frequency=0
  -A,//arcane-post-processing/save-init=false
  -A,//arcane-post-processing/save-final-time=false
  -A,//arcane-post-processing/end-execution-output=false
)

# Chemins depuis Fourier/my_cases/scripts/.
cd ../..
FOURIER_DIR="$PWD"

ARC_FILE="$FOURIER_DIR/my_cases/inputs/conduction.center-source.arc"
OUTPUT_DIR="$FOURIER_DIR/my_cases/outputs/$MODE"
LOG_FILE="$OUTPUT_DIR/logs_WS.txt"
TIMER_FILE="$OUTPUT_DIR/WeakScaling.txt"
WARMUP_LOG="$OUTPUT_DIR/warmup_WS.log"

# Environnement
module load craype-x86-trento
module load craype-accel-amd-gfx90a
module load PrgEnv-amd
module load cmake/3.27.9
module load rocm/6.4.3
module load cray-hdf5-parallel/1.14.3.7

mkdir -p "$OUTPUT_DIR"

if [[ ! -x "$FOURIER_DIR/Fourier" ]]; then
  echo "Executable introuvable : $FOURIER_DIR/Fourier" >&2
  exit 1
fi
if [[ ! -f "$ARC_FILE" ]]; then
  echo "Fichier introuvable : $ARC_FILE" >&2
  exit 1
fi

for case_config in "${CASES[@]}"; do
  read -r NB_PROCS PART_X PART_Y <<< "$case_config"
  if (( NB_PROCS <= 0 || PART_X <= 0 || PART_Y <= 0 || PART_X * PART_Y != NB_PROCS )); then
    echo "Configuration invalide : $case_config" >&2
    exit 1
  fi
done

echo "Mode=$MODE, PC=$PC, solveur=$SOLVER"
echo "Cellules par rang : $N_LOCAL x $M_LOCAL = $(( N_LOCAL * M_LOCAL ))"

: > "$LOG_FILE"
: > "$TIMER_FILE"

run_fourier()
{
  local nb_procs="$1"
  local part_x="$2"
  local part_y="$3"
  local mesh_n="$4"
  local mesh_m="$5"
  local length_x length_y
  local node_flags=()

  # Pas spatial constant, y compris pour le warm-up.
  length_x="$(LC_ALL=C awk -v l="$LOCAL_LENGTH_X" -v n="$mesh_n" -v n0="$N_LOCAL" \
    'BEGIN { printf "%.15g", l * n / n0 }')"
  length_y="$(LC_ALL=C awk -v l="$LOCAL_LENGTH_Y" -v n="$mesh_m" -v n0="$M_LOCAL" \
    'BEGIN { printf "%.15g", l * n / n0 }')"

  # Limite chaque étape GPU au nombre de nœuds nécessaire (8 GCD/nœud).
  if [[ "$MODE" == "gpu" ]]; then
    node_flags=(--nodes="$(( (nb_procs + 7) / 8 ))")
  fi

  srun \
    --jobid="$JOB_ID" \
    --ntasks="$nb_procs" \
    --threads-per-core=1 \
    --distribution=block:block \
    --cpu-bind=verbose,cores \
    "${SRUN_FLAGS[@]}" \
    "${node_flags[@]}" \
    ./Fourier "${ARCANE_FLAGS[@]}" "$ARC_FILE" \
    "${OUTPUT_FLAGS[@]}" \
    -A,//fem/linear-system/@name=HypreLinearSystem \
    -A,//fem/linear-system/solver="$SOLVER" \
    -A,//fem/linear-system/preconditioner="$PC" \
    -A,//fem/linear-system/rtol="$R_TOL" \
    -A,//fem/linear-system/atol=0 \
    -A,//fem/linear-system/max-iter="$MAX_IT" \
    -A,//fem/linear-system/verbosity=2 \
    -A,//fem/linear-system/amg-threshold="$AMG_THRESHOLD" \
    -A,//fem/linear-system/amg-coarsener="$AMG_COARSENER" \
    -A,//fem/linear-system/amg-interp-type="$AMG_INTERP_TYPE" \
    -A,//fem/linear-system/amg-smoother="$AMG_SMOOTHER" \
    -A,//fem/linear-system/amg-near-null-space=false \
    -A,//meshes/mesh/generator/nb-part-x="$part_x" \
    -A,//meshes/mesh/generator/nb-part-y="$part_y" \
    -A,//meshes/mesh/generator/x/n="$mesh_n" \
    -A,//meshes/mesh/generator/x/length="$length_x" \
    -A,//meshes/mesh/generator/y/n="$mesh_m" \
    -A,//meshes/mesh/generator/y/length="$length_y"
}

# Warm-up séparé des mesures.
echo "========== Warm-up =========="
read -r WARMUP_PROCS WARMUP_X WARMUP_Y <<< "${CASES[0]}"

if run_fourier "$WARMUP_PROCS" "$WARMUP_X" "$WARMUP_Y" \
    "$WARMUP_SIZE" "$WARMUP_SIZE" >"$WARMUP_LOG" 2>&1; then
  echo "Warm-up terminé."
else
  STATUS=$?
  tail -n 40 "$WARMUP_LOG" >&2
  echo "Échec du warm-up. Log : $WARMUP_LOG" >&2
  exit "$STATUS"
fi

echo "========== Weak scaling =========="

for case_config in "${CASES[@]}"; do
  read -r NB_PROCS PART_X PART_Y <<< "$case_config"

  # Cellules par rang et pas spatial constants ; le domaine grandit.
  MESH_N=$(( N_LOCAL * PART_X ))
  MESH_M=$(( M_LOCAL * PART_Y ))

  echo "$NB_PROCS rangs ($PART_X x $PART_Y), maillage $MESH_N x $MESH_M"

  printf '# procs=%s part_x=%s part_y=%s mesh_n=%s mesh_m=%s\n' \
    "$NB_PROCS" "$PART_X" "$PART_Y" "$MESH_N" "$MESH_M" |
    tee -a "$LOG_FILE" >> "$TIMER_FILE"

  if run_fourier "$NB_PROCS" "$PART_X" "$PART_Y" "$MESH_N" "$MESH_M" \
      2>&1 |
      tee -a "$LOG_FILE" |
      awk '
        /\[ArcaneFem-Timer\]/ ||
        /\[Hypre-Timer\]/ ||
        /\[Hypre-Info\][[:space:]]+Converged in[[:space:]]+[0-9]+[[:space:]]+iterations/
      ' >> "$TIMER_FILE"; then
    printf '\n' >> "$TIMER_FILE"
    printf '\n' >> "$LOG_FILE"
  else
    STATUS=$?
    echo "Échec à $NB_PROCS rangs. Log : $LOG_FILE" >&2
    exit "$STATUS"
  fi
done

echo
echo "Benchmark terminé."
echo "Log complet : $LOG_FILE"
echo "Mesures     : $TIMER_FILE"

