<?xml version="1.0"?>
<case codename="PoissonTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>PoissonTest manufactured solution.</title>
    <timeloop>PoissonTestLoop</timeloop>
  </arcane>

  <arcane-post-processing>
   <output-period>1</output-period>
   <format name="VtkHdfV2PostProcessor" />
   <output>
     <variable>U</variable>
     <variable>UExact</variable>
   </output>
  </arcane-post-processing>

  <meshes>
    <mesh>
      <filename>meshes/unit_square.msh</filename>
    </mesh>
  </meshes>

  <fem>
    <matrix-format>DOK</matrix-format>
    <manufactured-solution>sine</manufactured-solution>
    <manufactured-solution-tolerance>8.0e-2</manufactured-solution-tolerance>
  </fem>
</case>
