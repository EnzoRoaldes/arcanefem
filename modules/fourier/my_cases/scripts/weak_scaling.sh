#!/usr/bin/env bash


echo "======== Paths Creation ========"
FOURIER_DIR="./../.."
INPUT_DIR="my_cases/inputs"
OUTPUT_HDF_DIR="output/depouillement/vtkhdfv2"
ARC_FILE="$INPUT_DIR/conduction.center-source.arc"
LOG_FILE="my_cases/outputs/logs_WS.txt"
N=4000
M=4000 
MAX_IT=5000
AMG_THRESHOLD=0.01
R_TOL=1e-15
PC="gamg"
SOLVER="cg"

cd "$FOURIER_DIR"


echo "======= Output Creation ======="
: > "my_cases/outputs/WeakScaling.txt"
: > "$LOG_FILE"




echo "========== Warmup =========="

mpiexec -n 8 ./Fourier "$ARC_FILE" \
    -A,//fem/petsc-flags="-ksp_monitor -ksp_max_it $MAX_IT -ksp_rtol $R_TOL \
    -ksp_view -pc_type $PC -pc_gamg_threshold $AMG_THRESHOLD -ksp_type $SOLVER -ksp_initial_guess_nonzero" \
    -A,//meshes/mesh/generator/nb-part-x=1  -A,//meshes/mesh/generator/nb-part-y=8 \
    -A,//meshes/mesh/generator/x/n=1000 -A,//meshes/mesh/generator/x/length=1.0 \
    -A,//meshes/mesh/generator/y/n=1600 -A,//meshes/mesh/generator/y/length=1.0 \
    -A,//fem/linear-system/@name=PetscLinearSystem \
    >/dev/null





echo "========== Execution =========="

# echo "== refined mesh 5 =="
# mpiexec -n 16 ./Fourier "$ARC_FILE" \
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

echo "== refined mesh 4 =="
mpiexec -n 8 ./Fourier "$ARC_FILE" \
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



# mpiexec -n 1 ./Fourier "$HOME/local/build/opt-arcanefem-cuda-Release/modules/fourier/my_cases/inputs/bench_square.arc" \
#     -A,//fem/petsc-flags="-ksp_monitor -ksp_max_it $MAX_IT -ksp_rtol $R_TOL \
#     -ksp_view -pc_type $PC -pc_gamg_threshold $AMG_THRESHOLD -ksp_type $SOLVER" \
#     -A,//meshes/mesh/generator/nb-part-x=1  -A,//meshes/mesh/generator/nb-part-y=1 \
#     -A,//meshes/mesh/generator/x/n=100 -A,//meshes/mesh/generator/x/length=1.0 \
#     -A,//meshes/mesh/generator/y/n=100 -A,//meshes/mesh/generator/y/length=1.0 \
#     -A,//fem/linear-system/@name=PetscLinearSystem \
#     -A,AcceleratorRuntime=cuda

    


echo "== refined mesh 3 =="
mpiexec -n 4 ./Fourier "$ARC_FILE" \
    -A,//fem/petsc-flags="-ksp_monitor -ksp_max_it $MAX_IT -ksp_rtol $R_TOL \
    -ksp_view -pc_type $PC -pc_gamg_threshold $AMG_THRESHOLD -ksp_type $SOLVER" \
    -A,//meshes/mesh/generator/nb-part-x=1  -A,//meshes/mesh/generator/nb-part-y=4 \
    -A,//meshes/mesh/generator/x/n=$((N/2)) -A,//meshes/mesh/generator/x/length=1.0 \
    -A,//meshes/mesh/generator/y/n=$((M/2)) -A,//meshes/mesh/generator/y/length=1.0 \
    -A,//fem/linear-system/@name=PetscLinearSystem \
    2>&1 |
    tee -a "$LOG_FILE" |
    {
        grep 'ArcaneFem-Timer'
        printf '\n'
    } >> "my_cases/outputs/WeakScaling.txt"

echo >> "my_cases/outputs/WeakScaling.txt"


echo "== refined mesh 2 =="
mpiexec -n 2 ./Fourier "$ARC_FILE" \
    -A,//fem/petsc-flags="-ksp_monitor -ksp_max_it $MAX_IT -ksp_rtol $R_TOL \
    -ksp_view -pc_type $PC -pc_gamg_threshold $AMG_THRESHOLD -ksp_type $SOLVER" \
    -A,//meshes/mesh/generator/nb-part-x=1  -A,//meshes/mesh/generator/nb-part-y=2 \
    -A,//meshes/mesh/generator/x/n=$((N/2)) -A,//meshes/mesh/generator/x/length=1.0 \
    -A,//meshes/mesh/generator/y/n=$((M/4)) -A,//meshes/mesh/generator/y/length=1.0 \
    -A,//fem/linear-system/@name=PetscLinearSystem \
    2>&1 |
    tee -a "$LOG_FILE" |
    {
        grep 'ArcaneFem-Timer'
        printf '\n'
    } >> "my_cases/outputs/WeakScaling.txt"

echo >> "my_cases/outputs/WeakScaling.txt"


echo "== initial mesh =="
mpiexec -n 1 ./Fourier "$ARC_FILE" \
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
