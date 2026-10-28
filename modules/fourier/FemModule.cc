// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FemModule.cc                                                (C) 2000-2026 */
/*                                                                           */
/* Simple module to solve Fourier's equation using FEM.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "FemModule.h"
#include "ElementMatrix.h"
#include "ElementMatrixHexQuad.h"
// enzo
#include <arcane/core/IParallelMng.h>
#include <arcane/core/IMeshMng.h>
#include <arcane/core/MeshHandle.h>

/*---------------------------------------------------------------------------*/
/**
 * @brief Initializes the FemModuleFourier at the start of the simulation.
 *
 *  - initializes degrees of freedom (DoFs) on nodes.
 *  - builds support for manufactured test case (optional).
 */
/*---------------------------------------------------------------------------*/

void FemModuleFourier::
startInit()
{
  info() << "[ArcaneFem-Info] Started module startInit()";
  Real elapsedTime = platform::getRealTime();

  m_assemble_linear_system = options()->assembleLinearSystem();
  m_solve_linear_system = options()->solveLinearSystem();
  m_cross_validation = options()->hasSolutionComparisonFile();

  auto* mesh_mng = subDomain()->meshMng();
  IMesh* fine_mesh = mesh_mng->findMeshHandle("FineMesh").mesh();
  IMesh* coarse_mesh = mesh_mng->findMeshHandle("CoarseMesh").mesh();
  if (!fine_mesh || !coarse_mesh || fine_mesh == coarse_mesh)
    ARCANE_FATAL("Expected two distinct meshes: FineMesh and CoarseMesh");

  auto queue = subDomain()->acceleratorMng()->queue();
  m_fine = std::make_unique<MeshState>(fine_mesh, traceMng(), queue, "Fine");
  m_coarse = std::make_unique<MeshState>(coarse_mesh, traceMng(), queue, "Coarse");
  
  for (MeshState* s : {m_fine.get(), m_coarse.get()}) {
    s->dofs_on_nodes.initialize(s->mesh, 1);
    s->u.fill(0.0);
  }

  // Maillage fin
  m_fine->matrix_format = options()->matrixFormatF();
  m_fine->petsc_flags = options()->petscFlagsF();

  // Maillage grossier
  m_coarse->matrix_format = options()->matrixFormatC();
  m_coarse->petsc_flags = options()->petscFlagsC();

  
  // Check if the mesh is a quad or hex mesh by examining the number of nodes in the first cell
  // On suppose les deux maillage du même type
  UnstructuredMeshConnectivityView connectivity(m_fine->mesh);
  const Int32 nb_node = connectivity.cellNode().nbNode(CellLocalId(0));
  if (m_fine->mesh->dimension() == 2)
    m_is_quad4_mesh = (nb_node == 4);
  else if (m_fine->mesh->dimension() == 3)
    m_is_hexa8_mesh = (nb_node == 8);

  BC::IArcaneFemBC* bc = options()->boundaryConditions();

  for (BC::IManufacturedSolution* bs : bc->manufacturedSolutions()) {
    if (bs->getManufacturedSource()) {
      ICaseFunction* opt_function_source = bs->getManufacturedSourceFunction();
      IStandardFunction* scf_source = bs->getManufacturedSourceStandardFunction();
      if (!scf_source)
        ARCANE_FATAL("No standard case function for option 'manufactured-source-condition'");
      auto* functorS = scf_source->getFunctorRealReal3ToReal();
      if (!functorS)
        ARCANE_FATAL("Standard function '{0}' is not convertible to f(Real,Real3) -> Real", opt_function_source->name());
      m_manufactured_source = functorS;
    }

    if (bs->getManufacturedDirichlet()) {
      ICaseFunction* opt_function = bs->getManufacturedDirichletFunction();
      IStandardFunction* scf = bs->getManufacturedDirichletStandardFunction();
      if (!scf)
        ARCANE_FATAL("No standard case function for option 'manufactured-dirichlet-condition'");
      auto* functor = scf->getFunctorRealReal3ToReal();
      if (!functor)
        ARCANE_FATAL("Standard function '{0}' is not convertible to f(Real,Real3) -> Real", opt_function->name());
      m_manufactured_dirichlet = functor;
    }
  }

  elapsedTime = platform::getRealTime() - elapsedTime;
  ArcaneFemFunctions::GeneralFunctions::printArcaneFemTime(traceMng(),"initialize", elapsedTime);
}

/*---------------------------------------------------------------------------*/
/**
 * @brief Performs the main computation for the FemModuleFourier.
 *
 * - Stops the time loop after 1 iteration since the equation is steady state.
 * - Resets, configures, and initializes the linear system.
 * - Executes the stationary solve.
 */
/*---------------------------------------------------------------------------*/

void FemModuleFourier::
compute()
{
  info() << "[ArcaneFem-Info] Started module compute()";
  Real elapsedTime = platform::getRealTime();

  // Stop code after computations
  if (m_global_iteration() > 0)
    subDomain()->timeLoopMng()->stopComputeLoop(true);

  const String service_name = options()->linearSystem.serviceName();

  // cas m_fine
  m_fine->linear_system.reset();
  m_fine->linear_system.setLinearSystemFactory(options()->linearSystem());
  m_fine->linear_system.initialize(subDomain(),  acceleratorMng()->defaultRunner(),  m_fine->dofs_on_nodes.dofFamily(), "Solver");

  if (m_fine->petsc_flags != NULL){
    CommandLineArguments args = ArcaneFemFunctions::GeneralFunctions::getPetscFlagsFromCommandline(m_fine->petsc_flags);
    m_fine->linear_system.setSolverCommandLineArguments(args);
  }
  if (m_fine->matrix_format == "BSR" || m_fine->matrix_format == "AF-BSR") {
    bool use_csr_in_linear_system_fine =
    options()->linearSystem.serviceName() == "HypreLinearSystem" ||
    options()->linearSystem.serviceName() == "AlienLinearSystem" ||
    options()->linearSystem.serviceName() == "PetscLinearSystem";
    if (m_fine->matrix_format == "BSR")
      m_fine->bsr_format.initialize(mesh(), 1, use_csr_in_linear_system_fine, 0);
    else
      m_fine->bsr_format.initialize(mesh(), 1, use_csr_in_linear_system_fine, 1);
    m_fine->bsr_format.computeSparsity();
  }
  // cas m_coarse
  m_coarse->linear_system.reset();
  m_coarse->linear_system.setLinearSystemFactory(options()->linearSystem());
  m_coarse->linear_system.initialize(subDomain(),  acceleratorMng()->defaultRunner(),  m_coarse->dofs_on_nodes.dofFamily(), "Solver");

  if (m_coarse->petsc_flags != NULL){
    CommandLineArguments args = ArcaneFemFunctions::GeneralFunctions::getPetscFlagsFromCommandline(m_coarse->petsc_flags);
    m_coarse->linear_system.setSolverCommandLineArguments(args);
  }
  if (m_coarse->matrix_format == "BSR" || m_coarse->matrix_format == "AF-BSR") {
    bool use_csr_in_linear_system_coarse =
    options()->linearSystem.serviceName() == "HypreLinearSystem" ||
    options()->linearSystem.serviceName() == "AlienLinearSystem" ||
    options()->linearSystem.serviceName() == "PetscLinearSystem";
    if (m_coarse->matrix_format == "BSR")
      m_coarse->bsr_format.initialize(mesh(), 1, use_csr_in_linear_system_coarse, 0);
    else
      m_coarse->bsr_format.initialize(mesh(), 1, use_csr_in_linear_system_coarse, 1);
    m_coarse->bsr_format.computeSparsity();
  }

  _doStationarySolve(*m_fine);
  _doStationarySolve(*m_coarse);

  elapsedTime = platform::getRealTime() - elapsedTime;
  ArcaneFemFunctions::GeneralFunctions::printArcaneFemTime(traceMng(), "compute", elapsedTime);
}

/*---------------------------------------------------------------------------*/
/**
 * @brief Performs a stationary solve for the FEM system.
 *
 * This method follows via the following steps:
 *   1. _getMaterialParameters()     Retrieves material parameters
 *   2. _assembleBilinearOperator()  Assembles the FEM  matrix 𝐀
 *   3. _assembleLinearOperator()    Assembles the FEM RHS vector 𝐛
 *   4. _solve()                     Solves for solution vector 𝐮 = 𝐀⁻¹𝐛
 *   5. _updateVariables()           Updates FEM variables 𝐮 = 𝐱
 *   6. _validateResults()           Regression test
 */
/*---------------------------------------------------------------------------*/

void FemModuleFourier::
_doStationarySolve(MeshState& s)
{
  _getMaterialParameters(s);
  if(m_assemble_linear_system){
    _assembleBilinearOperator(s);
    _assembleLinearOperator(s);
  }

  if(m_solve_linear_system){
    _solve(s);
    _updateVariables(s);
  }

  if(m_cross_validation){
    _validateResults(s);
  }
}

/*---------------------------------------------------------------------------*/
/**
 * @brief Retrieves and sets the material parameters for the simulation.
 *
 * This method initializes:
 *  - material properties:
 *       # thermal conductivity coefficient (`lambda`)
 *       # heat source term (`qdot`)
 */
/*---------------------------------------------------------------------------*/

void FemModuleFourier::
_getMaterialParameters(MeshState& s)
{
  info() << "[ArcaneFem-Info] Started module _getMaterialParameters()" << " mesh=" << s.mesh->name();
  Real elapsedTime = platform::getRealTime();

  lambda = options()->lambda();
  qdot = options()->qdot();

  s.cell_lambda.fill(lambda);

  if (options()->useSkyscraper()) {
    // Skyscrapper behaviour
    if (s.mesh->dimension() != 2)
      ARCANE_FATAL("The skyscraper material distribution requires a 2D mesh");

    ENUMERATE_ (Cell, icell, s.mesh->allCells()) {
      Cell cell = *icell;

      Real3 center = Real3::zero();

      for (Node node : cell.nodes())
        center += s.node_coord[node];

      const Real nb_nodes =
          static_cast<Real>(std::size(cell.nodes()));

      center /= nb_nodes;

      const Int32 ix =
          static_cast<Int32>(std::floor(10.0 * center.x));

      const Int32 iy =
          static_cast<Int32>(std::floor(10.0 * center.y));

      if ((ix % 2 == 1) && (iy % 2 == 1))
        s.cell_lambda[cell] = 100000.0 * (ix + iy + 1);
    }
  }
  else {
    // Standard behaviour used by existing tests
    for (const auto& bs : options()->materialProperty()) {
      const String group_name = bs->volume().name();
      CellGroup group = s.mesh->cellFamily()->findGroup(group_name);
      
      const Real value = bs->lambda();

      info() << "Lambda for group=" << group.name()
            << " v=" << value;

      ENUMERATE_ (Cell, icell, group) {
        s.cell_lambda[icell] = value;
      }
    }
  }

  elapsedTime = platform::getRealTime() - elapsedTime;
  ArcaneFemFunctions::GeneralFunctions::printArcaneFemTime(traceMng(), "get-material-params", elapsedTime);
}

void FemModuleFourier::
_assembleLinearOperator(MeshState& s)
{
  if (options()->linearSystem.serviceName() == "HypreLinearSystem" ||
      options()->linearSystem.serviceName() == "PetscLinearSystem")
    _assembleLinearOperatorGpu(s);
  else
    _assembleLinearOperatorCpu(s);
}

/*---------------------------------------------------------------------------*/
/**
 * @brief FEM linear operator for the current simulation step.      
 *
 * This method constructs the linear  system by  assembling the LHS matrix 
 * and  RHS vector, applying various boundary conditions and source terms.
 *
 * Steps involved:
 *  1. The RHS vector is initialized to zero before applying any conditions.
 *  2. If a constant source term is specified (`qdot`), apply it to the RHS.
 *  3. If Neumann BC are specified applied to the RHS.
 *  4. If Dirichlet BC are specified apply to the LHS & RHS. 
 *  5. If manufactured source conditions is specified, apply to RHS.
 *  6. If  manufactured Dirichlet BC are specified apply to the LHS & RHS. 
 */
/*---------------------------------------------------------------------------*/

void FemModuleFourier::
_assembleLinearOperatorCpu(MeshState& s)
{
  info() << "[ArcaneFem-Info] Started module _assembleLinearOperator()" << " mesh=" << s.mesh->name();
  Real elapsedTime = platform::getRealTime();

  if (s.matrix_format == "BSR" || s.matrix_format == "AF-BSR")
    s.bsr_format.toLinearSystem(s.linear_system);

  VariableDoFReal& rhs_values(s.linear_system.rhsVariable()); // Temporary variable to keep values for the RHS
  rhs_values.fill(0.0);

  auto node_dof(m_dofs_on_nodes.nodeDoFConnectivityView());

  if (options()->qdot.isPresent())
    ArcaneFemFunctions::BoundaryConditions::applyConstantSourceToRhs(qdot, s.mesh, node_dof, s.node_coord, rhs_values);

  // Helper lambda to apply boundary conditions
  auto applyBoundaryConditions = [&](auto BCFunctions) {
    BC::IArcaneFemBC* bc = options()->boundaryConditions();
    if (bc) {

      // Neumann
      for (BC::INeumannBoundaryCondition* bs : bc->neumannBoundaryConditions())
        ArcaneFemFunctions::BoundaryConditions::applyNeumannToRhs(bs, s.mesh, node_dof, s.node_coord, rhs_values);

      // Dirichlet
      for (BC::IDirichletBoundaryCondition* bs : bc->dirichletBoundaryConditions())
        ArcaneFemFunctions::BoundaryConditions::applyDirichletToLhsAndRhs(bs, node_dof, s.linear_system, rhs_values);

      // Manufactured boundary conditions
      for (BC::IManufacturedSolution* bs : bc->manufacturedSolutions()) {
        if (bs->getManufacturedSource()) {
          ARCANE_CHECK_POINTER(m_manufactured_source);
          info() << "Apply manufactured Source condition to all cells";
          BCFunctions.applyManufacturedSourceToRhs(m_manufactured_source, s.mesh, node_dof, s.node_coord, rhs_values);
        }
        if (bs->getManufacturedDirichlet()) {
          ARCANE_CHECK_POINTER(m_manufactured_dirichlet);
          info() << "Apply manufactured dirichlet condition to all borders";
          FaceGroup group = s.mesh->outerFaces();
          BCFunctions.applyManufacturedDirichletToLhsAndRhs(m_manufactured_dirichlet, lambda, group, bs, node_dof, s.node_coord, s.linear_system, rhs_values);
        }
      }
    }
  };

   if (s.mesh->dimension() == 3)
     applyBoundaryConditions(ArcaneFemFunctions::BoundaryConditions3D());
   else
     applyBoundaryConditions(ArcaneFemFunctions::BoundaryConditions2D());

  elapsedTime = platform::getRealTime() - elapsedTime;
  ArcaneFemFunctions::GeneralFunctions::printArcaneFemTime(traceMng(), "rhs-vector-assembly", elapsedTime);
}

/*---------------------------------------------------------------------------*/
/**
 * @brief FEM linear operator for the current simulation step.
 * GPU compatible. Currently working with HypreDoFLinearSystem.
 *
 * This method constructs the linear  system by  assembling the LHS matrix
 * and  RHS vector, applying various boundary conditions and source terms.
 *
 * Steps involved:
 *  1. The RHS vector is initialized to zero before applying any conditions.
 *  2. If Neumann BC are specified applied to the RHS.
 *  3. If Dirichlet BC/Point are specified apply to the LHS & RHS.
 */
/*---------------------------------------------------------------------------*/

void FemModuleFourier::_assembleLinearOperatorGpu(MeshState& s)
{
  info() << "[ArcaneFem-Info] Started module _assembleLinearOperatorGpu()" << " mesh=" << s.mesh->name();
  Real elapsedTime = platform::getRealTime();

  s.bsr_format.toLinearSystem(s.linear_system);

  auto& rhs_values(s.linear_system.rhsVariable());
  rhs_values.fill(0.0);

  // Because of Dirichlet (Penalty) implementation in Hypre.
  s.linear_system.getForcedInfo().fill(false);
  auto queue = subDomain()->acceleratorMng()->defaultQueue();

  if (options()->qdot.isPresent())
    FemUtils::Gpu::BoundaryConditions::applyConstantSourceToRhs(qdot, s.dofs_on_nodes, s.node_coord, rhs_values, s.mesh, queue);

  BC::IArcaneFemBC* bc = options()->boundaryConditions();
  if (bc) {
    for (BC::INeumannBoundaryCondition* bs : bc->neumannBoundaryConditions())
      FemUtils::Gpu::BoundaryConditions::applyNeumannToRhs(bs, s.dofs_on_nodes, s.node_coord, rhs_values, s.mesh, queue);

    for (BC::IDirichletBoundaryCondition* bs : bc->dirichletBoundaryConditions())
      FemUtils::Gpu::BoundaryConditions::applyDirichletToLhsAndRhs(bs, s.dofs_on_nodes, s.linear_system, s.mesh, queue);
  }

  elapsedTime = platform::getRealTime() - elapsedTime;
  ArcaneFemFunctions::GeneralFunctions::printArcaneFemTime(traceMng(),"rhs-vector-assembly-gpu", elapsedTime);
}

/*---------------------------------------------------------------------------*/
/**
 * @brief Calls the right function for LHS assembly given as mesh type.
 */
/*---------------------------------------------------------------------------*/

void FemModuleFourier::
_assembleBilinearOperator(MeshState& s)
{
  info() << "[ArcaneFem-Info] Started module _assembleBilinearOperator()"  << " mesh=" << s.mesh->name();
  Real elapsedTime = platform::getRealTime();

  if (s.matrix_format == "BSR" || s.matrix_format == "AF-BSR") {
    UnstructuredMeshConnectivityView m_connectivity_view(s.mesh);
    auto cn_cv = m_connectivity_view.cellNode();
    auto m_queue = subDomain()->acceleratorMng()->defaultQueue();
    auto command = makeCommand(m_queue);
    auto in_node_coord = ax::viewIn(command, s.node_coord);
    auto in_cell_lambda = ax::viewIn(command, s.cell_lambda);

    if (s.mesh->dimension() == 2) {
      if (s.matrix_format == "BSR") {
        if(m_is_quad4_mesh)
          s.bsr_format.assembleBilinearAtomic([=] ARCCORE_HOST_DEVICE(CellLocalId cell_lid) { return _computeElementMatrixQuad4Gpu(cell_lid, cn_cv, in_node_coord, in_cell_lambda); });
        else
          s.bsr_format.assembleBilinearAtomic([=] ARCCORE_HOST_DEVICE(CellLocalId cell_lid) { return computeElementMatrixTria3Gpu(cell_lid, cn_cv, in_node_coord, in_cell_lambda); });
      }
      else {
        if(m_is_quad4_mesh)
            s.bsr_format.assembleBilinearAtomicFree([=] ARCCORE_HOST_DEVICE(CellLocalId cell_lid, Int32 node_lid) { return computeElementVectorQuad4Gpu(cell_lid, cn_cv, in_node_coord, in_cell_lambda, node_lid); });
          else
            s.bsr_format.assembleBilinearAtomicFree([=] ARCCORE_HOST_DEVICE(CellLocalId cell_lid, Int32 node_lid) { return computeElementVectorTria3Gpu(cell_lid, cn_cv, in_node_coord, in_cell_lambda, node_lid); });      
      }
    }
      
    if (s.mesh->dimension() == 3) {
      if (s.matrix_format == "BSR") {
        if(m_is_hexa8_mesh)
          s.bsr_format.assembleBilinearAtomic([=] ARCCORE_HOST_DEVICE(CellLocalId cell_lid) { return _computeElementMatrixHexa8Gpu(cell_lid, cn_cv, in_node_coord, in_cell_lambda); });
        else
          s.bsr_format.assembleBilinearAtomic([=] ARCCORE_HOST_DEVICE(CellLocalId cell_lid) { return computeElementMatrixTetra4Gpu(cell_lid, cn_cv, in_node_coord, in_cell_lambda); });
      }
      else {
        if(m_is_hexa8_mesh)
            s.bsr_format.assembleBilinearAtomicFree([=] ARCCORE_HOST_DEVICE(CellLocalId cell_lid, Int32 node_lid) { return computeElementVectorHexa8Gpu(cell_lid, cn_cv, in_node_coord, in_cell_lambda, node_lid); });
          else
            s.bsr_format.assembleBilinearAtomicFree([=] ARCCORE_HOST_DEVICE(CellLocalId cell_lid, Int32 node_lid) { return computeElementVectorTetra4Gpu(cell_lid, cn_cv, in_node_coord, in_cell_lambda, node_lid); });      
      }
    }
  }

  if (s.matrix_format == "DOK") {
    if (s.mesh->dimension() == 3) {
      if (m_is_hexa8_mesh)
        _assembleBilinear<8>(s, [this, &s](const Cell& cell) { return _computeElementMatrixHexa8(s, cell); });
      else
        _assembleBilinear<4>(s, [this, &s](const Cell& cell) { return _computeElementMatrixTetra4(s, cell); });
    }

    if (s.mesh->dimension() == 2) {
      if (m_is_quad4_mesh)
        _assembleBilinear<4>(s, [this, &s](const Cell& cell) { return _computeElementMatrixQuad4(s, cell); });
      else
        _assembleBilinear<3>(s, [this, &s](const Cell& cell) { return _computeElementMatrixTria3(s, cell); });
    }
  }

  elapsedTime = platform::getRealTime() - elapsedTime;
  ArcaneFemFunctions::GeneralFunctions::printArcaneFemTime(traceMng(), "lhs-matrix-assembly", elapsedTime);
}

/*---------------------------------------------------------------------------*/
/**
 * @brief Assembles the bilinear operator matrix for the FEM linear system.
 *
 * The method performs the following steps:
 *   1. For each cell, retrieves the cell-specific constant `lambda`.
 *   2. Computes element matrix using provided `compute_element_matrix` function.
 *   3. Assembles global matrix by adding contributions from each cell's element 
 *      matrix to the corresponding entries in the global matrix.
 */
/*---------------------------------------------------------------------------*/

template <int N>
void FemModuleFourier::
_assembleBilinear(MeshState& s, const std::function<RealMatrix<N, N>(const Cell&)>& compute_element_matrix)
{
  auto node_dof(s.dofs_on_nodes.nodeDoFConnectivityView());

  ENUMERATE_ (Cell, icell, s.mesh->allCells()) {
    Cell cell = *icell;

    lambda = s.cell_lambda[cell]; // lambda is always considered cell constant
    auto K_e = compute_element_matrix(cell); // element matrix based on the provided function
    Int32 n1_index = 0;
    for (Node node1 : cell.nodes()) {
      Int32 n2_index = 0;
      for (Node node2 : cell.nodes()) {
        Real v = K_e(n1_index, n2_index);
        if (node1.isOwn()) {
          s.linear_system.matrixAddValue(node_dof.dofId(node1, 0), node_dof.dofId(node2, 0), v);
        }
        ++n2_index;
      }
      ++n1_index;
    }
  }
}

/*---------------------------------------------------------------------------*/
/**
 * @brief Solves the linear system and updates the solution vector.
 *
 * This method performs the following actions:
 *   1. Solves the linear system to compute the solution.
 *   2. Copies the computed solution from the DoF to the node values.
 *   3. Synchronizes the updated node values.
 */
/*---------------------------------------------------------------------------*/

void FemModuleFourier::
_solve(MeshState& s)
{
  info() << "[ArcaneFem-Info] Started module _solve()" << " mesh=" << s.mesh->name();

  const Int32 nb_repetitions = options()->solveRepetitions();

  auto* pm = s.dofs_on_nodes.dofFamily()->parallelMng();
  auto* queue = subDomain()->acceleratorMng()->defaultQueue();
  
  auto& solution = s.linear_system.solutionVariable();

  s.linear_system.applyMatrixTransformation();
  s.linear_system.applyRHSTransformation();
  queue->barrier();

  for (Int32 i = 0; i < nb_repetitions; ++i) {
    solution.fill(0.0);
    queue->barrier();
    pm->barrier();

    const Real start = platform::getRealTime();
    s.linear_system.solve();
    queue->barrier();
    const Real local_time = platform::getRealTime() - start;
    const Real time = pm->reduce(Parallel::ReduceMax, local_time);

    const String name = String::format("solve-linear-system-{0}", i + 1);
    ArcaneFemFunctions::GeneralFunctions::printArcaneFemTime(traceMng(), name, time);
  }

}

/*---------------------------------------------------------------------------*/
/**
 * @brief Update the FEM variables.
 *
 * This method performs the following actions:
 *   1. Fetches values of solution from solved linear system to FEM variables,
 *      i.e., it copies RHS DOF to u.
 *   2. Performs synchronize of FEM variables across subdomains.
 */
/*---------------------------------------------------------------------------*/

void FemModuleFourier::
_updateVariables(MeshState& s)
{
  info() << "[ArcaneFem-Info] Started module _updateVariables()" << " mesh=" << s.mesh->name();
  Real elapsedTime = platform::getRealTime();

  { // copies solution (and optionally exact solution) to FEM output
    VariableDoFReal& dof_u(s.linear_system.solutionVariable());
    auto node_dof(s.dofs_on_nodes.nodeDoFConnectivityView());

    ENUMERATE_ (Node, inode, s.mesh->ownNodes()) {
      Node node = *inode;
      Real v = dof_u[node_dof.dofId(node, 0)];
      s.u[node] = v;
    }

    for (BC::IManufacturedSolution* bs : options()->boundaryConditions()->manufacturedSolutions())
      ENUMERATE_ (Node, inode, s.mesh->ownNodes()) {
        Node node = *inode;
        s.u_exact[node] = m_manufactured_dirichlet->apply(lambda, s.node_coord[node]);
      }
  }

  s.u.synchronize();

  for (BC::IManufacturedSolution* bs : options()->boundaryConditions()->manufacturedSolutions())
    s.u_exact.synchronize();

  elapsedTime = platform::getRealTime() - elapsedTime;
  ArcaneFemFunctions::GeneralFunctions::printArcaneFemTime(traceMng(), "update-variables", elapsedTime);
}

/*---------------------------------------------------------------------------*/
/**
 * @brief Validates and prints the results of the FEM computation.
 *
 * This method performs the following actions:
 *   1. Prints the computed values for each node.
 *   2. Retrieves the filename for the result file from options.
 *   3. If a filename is provided, checks the computed results against result file.
 *
 * @note The result comparison uses a tolerance of 1.0e-4.
 */
/*---------------------------------------------------------------------------*/

void FemModuleFourier::
_validateResults(MeshState& s)
{
  info() << "[ArcaneFem-Info] Started module _validateResults()" << " mesh=" << s.mesh->name();
  Real elapsedTime = platform::getRealTime();

  ENUMERATE_ (Node, inode, s.mesh->allNodes()) {
    Node node = *inode;
    info() << "u[" << node.uniqueId() << "] = " << s.u[node];
  }

  const bool is_fine = (&s == m_fine.get());
  const String filename = is_fine
      ? options()->solutionComparisonFileF()
      : options()->solutionComparisonFileC();
  
  checkNodeResultFile(traceMng(), filename, s.u, 1.0e-4);

  elapsedTime = platform::getRealTime() - elapsedTime;
  ArcaneFemFunctions::GeneralFunctions::printArcaneFemTime(traceMng(), "result-validation", elapsedTime);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_FEM(FemModuleFourier);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
