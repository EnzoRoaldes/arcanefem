#!/usr/bin/env bash

module load craype-x86-trento
module load craype-accel-amd-gfx90a
module load PrgEnv-amd
module load cmake/3.27.9
module load rocm/6.4.3
module load cray-hdf5-parallel/1.14.3.7

echo "======== Paths Creation ========"
FOURIER_DIR="./../.."
INPUT_DIR="my_cases/inputs"
OUTPUT_HDF_DIR="output/depouillement/vtkhdfv2"
ARC_FILE="$INPUT_DIR/conduction.heterogeneous.arc"
LOG_FILE="my_cases/outputs/logs_WS.txt"
N=10000
MAX_IT=1000
AMG_THRESHOLD=0.01
R_TOL=1e-10
PC="gamg"
SOLVER="cg"
JOB_ID="$1"

cd "$FOURIER_DIR"




echo "======= Output Creation ======="
: > "my_cases/outputs/WeakScaling.txt"
: > "$LOG_FILE"




echo "========== Warmup =========="

echo "== refined mesh 192 =="
srun --jobid="$JOB_ID" --ntasks=192 --cpus-per-task=1 ./Fourier "$ARC_FILE" \
    -A,//fem/petsc-flags="-ksp_monitor -ksp_max_it $MAX_IT -ksp_rtol $R_TOL \
    -ksp_view -pc_type $PC -pc_gamg_threshold $AMG_THRESHOLD -ksp_type $SOLVER" \
    -A,//meshes/mesh/generator/nb-part-x=16  -A,//meshes/mesh/generator/nb-part-y=12 \
    -A,//meshes/mesh/generator/x/n=$N -A,//meshes/mesh/generator/x/length=192.0 \
    -A,//meshes/mesh/generator/y/n=$M -A,//meshes/mesh/generator/y/length=192.0 \
    -A,//fem/linear-system/@name=PetscLinearSystem \
    >/dev/null




echo "========== Execution =========="

echo "== refined mesh 192 =="
srun --jobid="$JOB_ID" --ntasks=192 --cpus-per-task=1 ./Fourier "$ARC_FILE" \
    -A,//fem/petsc-flags="-ksp_monitor -ksp_max_it $MAX_IT -ksp_rtol $R_TOL \
    -ksp_view -pc_type $PC -pc_gamg_threshold $AMG_THRESHOLD -ksp_type $SOLVER" \
    -A,//meshes/mesh/generator/nb-part-x=16  -A,//meshes/mesh/generator/nb-part-y=12 \
    -A,//meshes/mesh/generator/x/n=$N -A,//meshes/mesh/generator/x/length=192.0 \
    -A,//meshes/mesh/generator/y/n=$M -A,//meshes/mesh/generator/y/length=192.0 \
    -A,//fem/linear-system/@name=PetscLinearSystem \
    2>&1 |
    tee -a "$LOG_FILE" |
    {
        grep 'ArcaneFem-Timer'
        printf '\n'
    } >> "my_cases/outputs/WeakScaling.txt"

echo >> "my_cases/outputs/WeakScaling.txt"


echo "== refined mesh 96 =="
srun --jobid="$JOB_ID" --ntasks=96 --cpus-per-task=1 ./Fourier "$ARC_FILE" \
    -A,//fem/petsc-flags="-ksp_monitor -ksp_max_it $MAX_IT -ksp_rtol $R_TOL \
    -ksp_view -pc_type $PC -pc_gamg_threshold $AMG_THRESHOLD -ksp_type $SOLVER" \
    -A,//meshes/mesh/generator/nb-part-x=8  -A,//meshes/mesh/generator/nb-part-y=12 \
    -A,//meshes/mesh/generator/x/n=$N -A,//meshes/mesh/generator/x/length=96.0 \
    -A,//meshes/mesh/generator/y/n=$M -A,//meshes/mesh/generator/y/length=192.0 \
    -A,//fem/linear-system/@name=PetscLinearSystem \
    2>&1 |
    tee -a "$LOG_FILE" |
    {
        grep 'ArcaneFem-Timer'
        printf '\n'
    } >> "my_cases/outputs/WeakScaling.txt"

echo >> "my_cases/outputs/WeakScaling.txt"


echo "== refined mesh 48 =="
srun --jobid="$JOB_ID" --ntasks=48 --cpus-per-task=1 ./Fourier "$ARC_FILE" \
    -A,//fem/petsc-flags="-ksp_monitor -ksp_max_it $MAX_IT -ksp_rtol $R_TOL \
    -ksp_view -pc_type $PC -pc_gamg_threshold $AMG_THRESHOLD -ksp_type $SOLVER" \
    -A,//meshes/mesh/generator/nb-part-x=4  -A,//meshes/mesh/generator/nb-part-y=4 \
    -A,//meshes/mesh/generator/x/n=$N -A,//meshes/mesh/generator/x/length=1.0 \
    -A,//meshes/mesh/generator/y/n=$M -A,//meshes/mesh/generator/y/length=1.0 \
    -A,//fem/linear-system/@name=PetscLinearSystem \
    2>&1 |
    tee -a "$LOG_FILE" |
    {
        grep 'ArcaneFem-Timer'
        printf '\n'
    } >> "my_cases/outputs/WeakScaling.txt"

echo >> "my_cases/outputs/WeakScaling.txt"


# echo "== refined mesh 24 =="
# srun --jobid="$JOB_ID" --ntasks=24 --cpus-per-task=1 ./Fourier "$ARC_FILE" \
#     -A,//fem/petsc-flags="-ksp_monitor -ksp_max_it $MAX_IT -ksp_rtol $R_TOL \
#     -ksp_view -pc_type $PC -pc_gamg_threshold $AMG_THRESHOLD -ksp_type $SOLVER" \
#     -A,//meshes/mesh/generator/nb-part-x=4  -A,//meshes/mesh/generator/nb-part-y=4 \
#     -A,//meshes/mesh/generator/x/n=$N -A,//meshes/mesh/generator/x/length=1.0 \
#     -A,//meshes/mesh/generator/y/n=$M -A,//meshes/mesh/generator/y/length=1.0 \
#     -A,//fem/linear-system/@name=PetscLinearSystem \
#     2>&1 |
#     tee -a "$LOG_FILE" |
#     {
#         grep 'ArcaneFem-Timer'
#         printf '\n'
#     } >> "my_cases/outputs/WeakScaling.txt"

# echo >> "my_cases/outputs/WeakScaling.txt"


echo "== refined mesh 12 =="
srun --jobid="$JOB_ID" --ntasks=12 --cpus-per-task=1 ./Fourier "$ARC_FILE" \
    -A,//fem/petsc-flags="-ksp_monitor -ksp_max_it $MAX_IT -ksp_rtol $R_TOL \
    -ksp_view -pc_type $PC -pc_gamg_threshold $AMG_THRESHOLD -ksp_type $SOLVER" \
    -A,//meshes/mesh/generator/nb-part-x=1  -A,//meshes/mesh/generator/nb-part-y=12 \
    -A,//meshes/mesh/generator/x/n=$N -A,//meshes/mesh/generator/x/length=1.0 \
    -A,//meshes/mesh/generator/y/n=$((M/2)) -A,//meshes/mesh/generator/y/length=1.0 \
    -A,//fem/linear-system/@name=PetscLinearSystem \
    2>&1 |
    tee -a "$LOG_FILE" |
    {
        grep 'ArcaneFem-Timer'
        printf '\n'
    } >> "my_cases/outputs/WeakScaling.txt"

echo >> "my_cases/outputs/WeakScaling.txt"


echo "== refined mesh 8 =="
srun --jobid="$JOB_ID" --ntasks=8 --cpus-per-task=1 ./Fourier "$ARC_FILE" \
    -A,//fem/petsc-flags="-ksp_monitor -ksp_max_it $MAX_IT -ksp_rtol $R_TOL \
    -ksp_view -pc_type $PC -pc_gamg_threshold $AMG_THRESHOLD -ksp_type $SOLVER" \
    -A,//meshes/mesh/generator/nb-part-x=1  -A,//meshes/mesh/generator/nb-part-y=8 \
    -A,//meshes/mesh/generator/x/n=$N -A,//meshes/mesh/generator/x/length=1.0 \
    -A,//meshes/mesh/generator/y/n=$((M/2)) -A,//meshes/mesh/generator/y/length=1.0 \
    -A,//fem/linear-system/@name=PetscLinearSystem \
    2>&1 |
    tee -a "$LOG_FILE" |
    {
        grep 'ArcaneFem-Timer'
        printf '\n'
    } >> "my_cases/outputs/WeakScaling.txt"

echo >> "my_cases/outputs/WeakScaling.txt"


echo "== initial mesh =="
srun --jobid="$JOB_ID" --ntasks=1 --cpus-per-task=1 ./Fourier "$ARC_FILE" \
    -A,//fem/petsc-flags="-ksp_monitor -ksp_max_it $MAX_IT -ksp_rtol $R_TOL \
    -ksp_view -pc_type $PC -pc_gamg_threshold $AMG_THRESHOLD -ksp_type $SOLVER" \
    -A,//meshes/mesh/generator/nb-part-x=1  -A,//meshes/mesh/generator/nb-part-y=1 \
    -A,//meshes/mesh/generator/x/n=$((N/4)) -A,//meshes/mesh/generator/x/length=1.0 \
    -A,//meshes/mesh/generator/y/n=$((M/4)) -A,//meshes/mesh/generator/y/length=1.0 \
    -A,//fem/linear-system/@name=PetscLinearSystem \
    2>&1 |
    tee -a "$LOG_FILE" |
    {
        grep 'ArcaneFem-Timer'
        printf '\n'
    } >> "my_cases/outputs/WeakScaling.txt"

echo >> "my_cases/outputs/WeakScaling.txt"
