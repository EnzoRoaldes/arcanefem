<?xml version="1.0"?>
<case codename="PoissonTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Cut circle 2D</title>
    <timeloop>PoissonTestLoop</timeloop>
  </arcane>

  <arcane-post-processing>
   <output-period>1</output-period>
   <!-- <format name="VtkHdfV2PostProcessor" /> -->
   <output>
     <variable>U</variable>
   </output>
  </arcane-post-processing>

  <meshes>
    <!-- coarse mesh (CPU) -->
    <mesh>
      <generator name="Cartesian2D">
        <x>
          <n>400</n>
          <length>1.0</length>
        </x>

        <y>
          <n>400</n>
          <length>1.0</length>
        </y>

        <origin>0.0 0.0</origin>

        <nb-part-x>4</nb-part-x>
        <nb-part-y>2</nb-part-y>

        <generate-sod-groups>true</generate-sod-groups>

      </generator>
    </mesh>

    <!-- fine mesh (GPU) -->
    <mesh>
      <generator name="Cartesian2D">
        <x>
          <n>200</n>
          <length>1.0</length>
        </x>

        <y>
          <n>200</n>
          <length>1.0</length>
        </y>

        <origin>0.0 0.0</origin>

        <nb-part-x>4</nb-part-x>
        <nb-part-y>2</nb-part-y>

        <generate-sod-groups>true</generate-sod-groups>

      </generator>
      <!-- <mesh-builder-customiser name="ArcaneParallelMngSplitterMeshBuilderCustomiser">
        <subset-ratio>2</subset-ratio>
      </mesh-builder-customiser> -->
    </mesh>
  </meshes>

  <fem>
    <f>5.5</f>
    <boundary-conditions-c>
      <dirichlet>
        <enforce-Dirichlet-method>RowColumnElimination</enforce-Dirichlet-method>
        <surface>XMIN</surface>
        <value>0.5</value>
      </dirichlet>
      <dirichlet>
        <enforce-Dirichlet-method>RowColumnElimination</enforce-Dirichlet-method>
        <surface>XMAX</surface>
        <value>10</value>
      </dirichlet>
    </boundary-conditions-c>
    <boundary-conditions-f>
      <dirichlet>
        <enforce-Dirichlet-method>RowColumnElimination</enforce-Dirichlet-method>
        <surface>XMIN</surface>
        <value>0.5</value>
      </dirichlet>
      <dirichlet>
        <enforce-Dirichlet-method>RowColumnElimination</enforce-Dirichlet-method>
        <surface>XMAX</surface>
        <value>10</value>
      </dirichlet>
    </boundary-conditions-f>
  </fem>
</case>
