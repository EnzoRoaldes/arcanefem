#!/bin/bash

#SBATCH --account=genden15
#SBATCH --job-name=SS_fourier
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --constraint GENOA
#SBATCH --time=02:00:00
#SBATCH --output=#SBATCH --output=/lus/work/RES1/genden15/eroaldes/software/install/fork_arcanefem-release/modules/fourier/my_cases/slurm_SS_%j.log


# export openmp_per_???=1

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
ARC_FILE="$INPUT_DIR/conduction.center-source.arc"
LOG_FILE="my_cases/outputs/logs_SS.txt"
N=7000
MAX_IT=5000
AMG_THRESHOLD=0.01
R_TOL=1e-15
PC="gamg"
SOLVER="cg"

cd "$FOURIER_DIR"


echo "======= Output Creation ======="
: > "my_cases/outputs/StrongScaling.txt"
: > "$LOG_FILE"


echo "========== Warmup =========="

srun --ntasks=192 --cpus-per-task=1 ./Fourier "$ARC_FILE" \
    -A,//fem/petsc-flags="-ksp_monitor -ksp_max_it $MAX_IT -ksp_rtol $R_TOL \
    -ksp_view -pc_type $PC -pc_gamg_threshold $AMG_THRESHOLD -ksp_type $SOLVER -ksp_initial_guess_nonzero" \
    -A,//meshes/mesh/generator/nb-part-x=16  -A,//meshes/mesh/generator/nb-part-y=12 \
    -A,//meshes/mesh/generator/x/n=$N -A,//meshes/mesh/generator/x/length=1.0 \
    -A,//meshes/mesh/generator/y/n=$N -A,//meshes/mesh/generator/y/length=1.0 \
    -A,//fem/linear-system/@name=PetscLinearSystem \
    >/dev/null



echo "========== Execution =========="


echo "== refined mesh 192 =="
srun --ntasks=192 --cpus-per-task=1 ./Fourier "$ARC_FILE" \
    -A,//fem/petsc-flags="-ksp_monitor -ksp_max_it $MAX_IT -ksp_rtol $R_TOL \
    -ksp_view -pc_type $PC -pc_gamg_threshold $AMG_THRESHOLD -ksp_type $SOLVER -ksp_initial_guess_nonzero" \
    -A,//meshes/mesh/generator/nb-part-x=16  -A,//meshes/mesh/generator/nb-part-y=12 \
    -A,//meshes/mesh/generator/x/n=$N -A,//meshes/mesh/generator/x/length=1.0 \
    -A,//meshes/mesh/generator/y/n=$N -A,//meshes/mesh/generator/y/length=1.0 \
    -A,//fem/linear-system/@name=PetscLinearSystem \
    2>&1 |
    tee -a "$LOG_FILE" |
    {
        grep 'ArcaneFem-Timer'
        printf '\n'
    } >> "my_cases/outputs/StrongScaling.txt"

echo >> "my_cases/outputs/StrongScaling.txt"


# echo "== refined mesh 96 =="
# srun --ntasks=96 --cpus-per-task=1 ./Fourier "$ARC_FILE" \
#     -A,//fem/petsc-flags="-ksp_monitor -ksp_max_it $MAX_IT -ksp_rtol $R_TOL \
#     -ksp_view -pc_type $PC -pc_gamg_threshold $AMG_THRESHOLD -ksp_type $SOLVER -ksp_initial_guess_nonzero" \
#     -A,//meshes/mesh/generator/nb-part-x=8  -A,//meshes/mesh/generator/nb-part-y=12 \
#     -A,//meshes/mesh/generator/x/n=$N -A,//meshes/mesh/generator/x/length=1.0 \
#     -A,//meshes/mesh/generator/y/n=$N -A,//meshes/mesh/generator/y/length=1.0 \
#     -A,//fem/linear-system/@name=PetscLinearSystem \
#     2>&1 |
#     tee -a "$LOG_FILE" |
#     {
#         grep 'ArcaneFem-Timer'
#         printf '\n'
#     } >> "my_cases/outputs/StrongScaling.txt"

# echo >> "my_cases/outputs/StrongScaling.txt"


echo "== refined mesh 48 =="
srun --ntasks=48 --cpus-per-task=1 ./Fourier "$ARC_FILE" \
    -A,//fem/petsc-flags="-ksp_monitor -ksp_max_it $MAX_IT -ksp_rtol $R_TOL \
    -ksp_view -pc_type $PC -pc_gamg_threshold $AMG_THRESHOLD -ksp_type $SOLVER -ksp_initial_guess_nonzero" \
    -A,//meshes/mesh/generator/nb-part-x=6  -A,//meshes/mesh/generator/nb-part-y=8 \
    -A,//meshes/mesh/generator/x/n=$N -A,//meshes/mesh/generator/x/length=1.0 \
    -A,//meshes/mesh/generator/y/n=$N -A,//meshes/mesh/generator/y/length=1.0 \
    -A,//fem/linear-system/@name=PetscLinearSystem \
    2>&1 |
    tee -a "$LOG_FILE" |
    {
        grep 'ArcaneFem-Timer'
        printf '\n'
    } >> "my_cases/outputs/StrongScaling.txt"

echo >> "my_cases/outputs/StrongScaling.txt"


# echo "== refined mesh 24 =="
# srun --ntasks=24 --cpus-per-task=1 ./Fourier "$ARC_FILE" \
#     -A,//fem/petsc-flags="-ksp_monitor -ksp_max_it $MAX_IT -ksp_rtol $R_TOL \
#     -ksp_view -pc_type $PC -pc_gamg_threshold $AMG_THRESHOLD -ksp_type $SOLVER -ksp_initial_guess_nonzero" \
#     -A,//meshes/mesh/generator/nb-part-x=4  -A,//meshes/mesh/generator/nb-part-y=6 \
#     -A,//meshes/mesh/generator/x/n=$N -A,//meshes/mesh/generator/x/length=1.0 \
#     -A,//meshes/mesh/generator/y/n=$N -A,//meshes/mesh/generator/y/length=1.0 \
#     -A,//fem/linear-system/@name=PetscLinearSystem \
#     2>&1 |
#     tee -a "$LOG_FILE" |
#     {
#         grep 'ArcaneFem-Timer'
#         printf '\n'
#     } >> "my_cases/outputs/StrongScaling.txt"

# echo >> "my_cases/outputs/StrongScaling.txt"


echo "== refined mesh 12 =="
srun --ntasks=12 --cpus-per-task=1 ./Fourier "$ARC_FILE" \
    -A,//fem/petsc-flags="-ksp_monitor -ksp_max_it $MAX_IT -ksp_rtol $R_TOL \
    -ksp_view -pc_type $PC -pc_gamg_threshold $AMG_THRESHOLD -ksp_type $SOLVER -ksp_initial_guess_nonzero" \
    -A,//meshes/mesh/generator/nb-part-x=3  -A,//meshes/mesh/generator/nb-part-y=4 \
    -A,//meshes/mesh/generator/x/n=$N -A,//meshes/mesh/generator/x/length=1.0 \
    -A,//meshes/mesh/generator/y/n=$N -A,//meshes/mesh/generator/y/length=1.0 \
    -A,//fem/linear-system/@name=PetscLinearSystem \
    2>&1 |
    tee -a "$LOG_FILE" |
    {
        grep 'ArcaneFem-Timer'
        printf '\n'
    } >> "my_cases/outputs/StrongScaling.txt"

echo >> "my_cases/outputs/StrongScaling.txt"


# echo "== refined mesh 8 =="
# srun --ntasks=8 --cpus-per-task=1 ./Fourier "$ARC_FILE" \
#     -A,//fem/petsc-flags="-ksp_monitor -ksp_max_it $MAX_IT -ksp_rtol $R_TOL \
#     -ksp_view -pc_type $PC -pc_gamg_threshold $AMG_THRESHOLD -ksp_type $SOLVER -ksp_initial_guess_nonzero" \
#     -A,//meshes/mesh/generator/nb-part-x=2  -A,//meshes/mesh/generator/nb-part-y=4 \
#     -A,//meshes/mesh/generator/x/n=$N -A,//meshes/mesh/generator/x/length=1.0 \
#     -A,//meshes/mesh/generator/y/n=$N -A,//meshes/mesh/generator/y/length=1.0 \
#     -A,//fem/linear-system/@name=PetscLinearSystem \
#     2>&1 |
#     tee -a "$LOG_FILE" |
#     {
#         grep 'ArcaneFem-Timer'
#         printf '\n'
#     } >> "my_cases/outputs/StrongScaling.txt"

# echo >> "my_cases/outputs/StrongScaling.txt"


echo "== refined mesh 4 =="
srun --ntasks=4 --cpus-per-task=1 ./Fourier "$ARC_FILE" \
    -A,//fem/petsc-flags="-ksp_monitor -ksp_max_it $MAX_IT -ksp_rtol $R_TOL \
    -ksp_view -pc_type $PC -pc_gamg_threshold $AMG_THRESHOLD -ksp_type $SOLVER -ksp_initial_guess_nonzero" \
    -A,//meshes/mesh/generator/nb-part-x=2  -A,//meshes/mesh/generator/nb-part-y=2 \
    -A,//meshes/mesh/generator/x/n=$N -A,//meshes/mesh/generator/x/length=1.0 \
    -A,//meshes/mesh/generator/y/n=$N -A,//meshes/mesh/generator/y/length=1.0 \
    -A,//fem/linear-system/@name=PetscLinearSystem \
    2>&1 |
    tee -a "$LOG_FILE" |
    {
        grep 'ArcaneFem-Timer'
        printf '\n'
    } >> "my_cases/outputs/StrongScaling.txt"

echo >> "my_cases/outputs/StrongScaling.txt"


# echo "== refined mesh 2 =="
# srun --ntasks=2 --cpus-per-task=1 ./Fourier "$ARC_FILE" \
#     -A,//fem/petsc-flags="-ksp_monitor -ksp_max_it $MAX_IT -ksp_rtol $R_TOL \
#     -ksp_view -pc_type $PC -pc_gamg_threshold $AMG_THRESHOLD -ksp_type $SOLVER -ksp_initial_guess_nonzero" \
#     -A,//meshes/mesh/generator/nb-part-x=1  -A,//meshes/mesh/generator/nb-part-y=2 \
#     -A,//meshes/mesh/generator/x/n=$N -A,//meshes/mesh/generator/x/length=1.0 \
#     -A,//meshes/mesh/generator/y/n=$N -A,//meshes/mesh/generator/y/length=1.0 \
#     -A,//fem/linear-system/@name=PetscLinearSystem \
#     2>&1 |
#     tee -a "$LOG_FILE" |
#     {
#         grep 'ArcaneFem-Timer'
#         printf '\n'
#     } >> "my_cases/outputs/StrongScaling.txt"

# echo >> "my_cases/outputs/StrongScaling.txt"



echo "== initial mesh =="
srun --ntasks=1 --cpus-per-task=1 ./Fourier "$ARC_FILE" \
    -A,//fem/petsc-flags="-ksp_monitor -ksp_max_it $MAX_IT -ksp_rtol $R_TOL \
    -ksp_view -pc_type $PC -pc_gamg_threshold $AMG_THRESHOLD -ksp_type $SOLVER -ksp_initial_guess_nonzero" \
    -A,//meshes/mesh/generator/nb-part-x=1  -A,//meshes/mesh/generator/nb-part-y=1 \
    -A,//meshes/mesh/generator/x/n=$N -A,//meshes/mesh/generator/x/length=1.0 \
    -A,//meshes/mesh/generator/y/n=$N -A,//meshes/mesh/generator/y/length=1.0 \
    -A,//fem/linear-system/@name=PetscLinearSystem \
    2>&1 |
    tee -a "$LOG_FILE" |
    {
        grep 'ArcaneFem-Timer'
        printf '\n'
} >> "my_cases/outputs/StrongScaling.txt"


echo >> "my_cases/outputs/StrongScaling.txt"


# srun --ntasks=1 --cpus-per-task=1 ./Fourier ./my_cases/inputs/conduction.heterogeneous.arc -A,//fem/petsc-flags="-ksp_monitor -ksp_max_it 500 -ksp_rtol 1e-15  -ksp_view -pc_type gamg -pc_gamg_threshold 0.05 -ksp_type cg -ksp_initial_guess_nonzero"   -A,//meshes/mesh/generator/nb-part-x=1  -A,//meshes/mesh/generator/nb-part-y=1  -A,//meshes/mesh/generator/x/n=500 -A,//meshes/mesh/generator/x/length=1.0   -A,//meshes/mesh/generator/y/n=500 -A,//meshes/mesh/generator/y/length=1.0   -A,//fem/linear-system/@name=PetscLinearSystem

# srun --ntasks=96 --cpus-per-task=1 ./Fourier ./my_cases/inputs/conduction.heterogeneous.arc -A,//fem/petsc-flags="-ksp_monitor -ksp_max_it 500 -ksp_rtol 1e-15  -ksp_view -pc_type gamg -pc_gamg_threshold 0.05 -ksp_type cg -ksp_initial_guess_nonzero"   -A,//meshes/mesh/generator/nb-part-x=8  -A,//meshes/mesh/generator/nb-part-y=12  -A,//meshes/mesh/generator/x/n=7000 -A,//meshes/mesh/generator/x/length=70.0   -A,//meshes/mesh/generator/y/n=7000 -A,//meshes/mesh/generator/y/length=70.0   -A,//fem/linear-system/@name=PetscLinearSystem