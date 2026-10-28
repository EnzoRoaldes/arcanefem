// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FemModule.h                                                 (C) 2000-2026 */
/*                                                                           */
/* FemModuleFourier class definition.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef FEMMODULES_H
#define FEMMODULES_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arcane/utils/PlatformUtils.h>
#include <arcane/utils/NumArray.h>
#include <arcane/utils/CommandLineArguments.h>
#include <arcane/utils/Real3.h>
#include <arcane/ITimeLoopMng.h>
#include <arcane/IMesh.h>
#include <arcane/IItemFamily.h>
#include <arcane/ItemGroup.h>
#include <arcane/ICaseMng.h>
#include <arcane/core/IStandardFunction.h>
#include <arcane/accelerator/core/IAcceleratorMng.h>
#include <arcane/accelerator/VariableViews.h>

#include "IArcaneFemBC.h"
#include "IDoFLinearSystemFactory.h"
#include "Fem_axl.h"
#include "FemUtils.h"
#include "DoFLinearSystem.h"
#include "FemDoFsOnNodes.h"
#include "ArcaneFemFunctions.h"
#include "ArcaneFemFunctionsGpu.h"
#include "BSRFormat.h"

///
#include <memory>
#include <arcane/core/VariableBuildInfo.h>
#include <arcane/accelerator/core/RunQueue.h>
///

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace Arcane::FemUtils;
namespace ax = Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/**
 * @brief A module for finite element method.
 *
 * This class handles the initialization and computation for finite element
 * method (FEM) simulations, providing methods to  set  up and solve linear
 * systems, assemble FEM operators, and perform result checks.
 */
/*---------------------------------------------------------------------------*/


///
struct MeshState
{
  IMesh* mesh;

  VariableNodeReal u;
  VariableNodeReal u_exact;
  VariableNodeReal3 node_coord;
  VariableCellReal cell_lambda;
  String petsc_flags;
  String matrix_format="DOK";
  FemDoFsOnNodes dofs_on_nodes;
  BSRFormat bsr_format;
  DoFLinearSystem linear_system;

  MeshState(IMesh* mesh_ptr,
            ITraceMng* trace_mng,
            ax::RunQueue& queue,
            const String& suffix)
  : mesh(mesh_ptr)
  , u(VariableBuildInfo(mesh_ptr, "U" + suffix))
  , u_exact(VariableBuildInfo(mesh_ptr, "UExact" + suffix))
  , node_coord(mesh_ptr->nodesCoordinates())
  , cell_lambda(VariableBuildInfo(mesh_ptr, "CellLambda" + suffix))
  , dofs_on_nodes(trace_mng)
  , bsr_format(trace_mng, queue, dofs_on_nodes)
  {}

  MeshState(const MeshState&) = delete;
  MeshState(MeshState&&) = delete;
  MeshState& operator=(MeshState&&) = delete;
  MeshState& operator=(const MeshState&) = delete;
};
///




class FemModuleFourier
: public ArcaneFemObject
{
 public:

  explicit FemModuleFourier(const ModuleBuildInfo& mbi)
  : ArcaneFemObject(mbi)
  , m_dofs_on_nodes(mbi.subDomain()->traceMng())
  , m_bsr_format(mbi.subDomain()->traceMng(), *(mbi.subDomain()->acceleratorMng()->defaultQueue()), m_dofs_on_nodes)
  {
    ICaseMng* cm = mbi.subDomain()->caseMng();
    cm->setTreatWarningAsError(true);
    cm->setAllowUnkownRootElelement(false);
  }

  void startInit() override; //! Method called at the beginning of the simulation
  void compute() override; //! Method called at each iteration
  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 0); }

  void _assembleBilinearOperator(MeshState& s);
  void _assembleLinearOperatorGpu(MeshState& s);

 private:

  ///
  std::unique_ptr<MeshState> m_fine;
  std::unique_ptr<MeshState> m_coarse;
  ///

  Real lambda;
  Real qdot;
  Real ElementNodes;

  String m_petsc_flags;
  String m_matrix_format = "DOK";

  bool m_assemble_linear_system = true;
  bool m_solve_linear_system = true;
  bool m_cross_validation = false;
  bool m_is_quad4_mesh = false;
  bool m_is_hexa8_mesh = false;

  DoFLinearSystem m_linear_system;
  IItemFamily* m_dof_family = nullptr;
  FemDoFsOnNodes m_dofs_on_nodes;
  BSRFormat m_bsr_format;

  void _doStationarySolve(MeshState& s);
  void _getMaterialParameters(MeshState& s);
  void _solve(MeshState& s);
  void _assembleLinearOperator(MeshState& s);
  void _assembleLinearOperatorCpu(MeshState& s);
  void _validateResults(MeshState& s);
  void _updateVariables(MeshState& s);

  RealMatrix<3, 3> _computeElementMatrixTria3(MeshState& s, Cell cell);
  RealMatrix<4, 4> _computeElementMatrixTetra4(MeshState& s, Cell cell);
  RealMatrix<4, 4> _computeElementMatrixQuad4(MeshState& s, Cell cell);
  RealMatrix<8, 8> _computeElementMatrixHexa8(MeshState& s, Cell cell);
  IBinaryMathFunctor<Real, Real3, Real>* m_manufactured_dirichlet = nullptr;
  IBinaryMathFunctor<Real, Real3, Real>* m_manufactured_source = nullptr;

  template<int N>
  void _assembleBilinear(MeshState& s, const std::function<RealMatrix<N, N>(const Cell&)>& compute_element_matrix);
};

#endif
