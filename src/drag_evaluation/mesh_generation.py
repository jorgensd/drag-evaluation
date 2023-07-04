import gmsh
from mpi4py import MPI
import meshio
from typing import Dict, List
from pathlib import Path
import numpy as np

__all__ = ["generate_cylinder_mesh", "create_mesh"]


def create_mesh(mesh: meshio.Mesh, cell_type: str, prune_z: bool = False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(
        points=points,
        cells={cell_type: cells},
        cell_data={"name_to_read": [cell_data]},
    )
    return out_mesh


def generate_cylinder_mesh(
    filename: Path,
    L1: float,
    L2: float,
    width: float,
    c_x: float,
    c_y: float,
    r: float,
    res_min: float,
    markers: Dict[str, int],
):
    """
    Generate a square mesh (-L1, L2)x(-width/2, width/2) with a circular whole or radius r
    at (0,0).

    Markers is a dictionary mapping boundary markers (integers) to four possible boundaries;
    - 'inlet'
    - 'outlet'
    - 'walls'
    - 'obstacle'
    """
    for key in markers.keys():
        assert key in ["inlet", "outlet", "walls", "obstacle"]

    gmsh.initialize()
    order = 1
    gdim = 2
    tmp_file = Path("tmp.msh")
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    if mesh_comm.rank == model_rank:
        rectangle = gmsh.model.occ.addRectangle(
            -L1, -width / 2, 0, L2 + L1, width, tag=1
        )
        obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
        gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
        gmsh.model.occ.synchronize()
        volumes = gmsh.model.getEntities(dim=gdim)
        assert len(volumes) == 1
        gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], 1)
        gmsh.model.setPhysicalName(volumes[0][0], 1, "Fluid")

    boundary_tags: Dict[int, List[int]] = {markers[key]: [] for key in markers.keys()}
    if mesh_comm.rank == model_rank:
        boundaries = gmsh.model.getBoundary(volumes, oriented=False)
        for boundary in boundaries:
            center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
            if np.allclose(center_of_mass, [-L1, 0, 0]):
                boundary_tags[markers["inlet"]].append(boundary[1])
            elif np.allclose(center_of_mass, [L2, 0, 0]):
                boundary_tags[markers["outlet"]].append(boundary[1])
            elif np.allclose(
                center_of_mass, [(L2 + L1) / 2 - L1, -width / 2, 0]
            ) or np.allclose(center_of_mass, [(L2 + L1) / 2 - L1, width / 2, 0]):
                boundary_tags[markers["walls"]].append(boundary[1])
            else:
                boundary_tags[markers["obstacle"]].append(boundary[1])

        for group, g_markers in boundary_tags.items():
            gmsh.model.addPhysicalGroup(1, g_markers, group)
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(
            distance_field, "EdgesList", boundary_tags[markers["obstacle"]]
        )
        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 10 * res_min)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 3 * r)
        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
        # gmsh.option.setNumber("Mesh.Algorithm", 8)
        # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        # gmsh.option.setNumber("Mesh.RecombineAll", 1)
        # gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.model.mesh.generate(gdim)
        gmsh.model.mesh.setOrder(order)
        gmsh.model.mesh.optimize("Netgen")

        gmsh.write(str(tmp_file))
    gmsh.finalize()
    mesh_comm.Barrier()
    filename = Path(filename)
    facet_file_name = filename.with_stem(filename.stem + "_facets").with_suffix(".xdmf")
    mesh_file_name = filename.with_suffix(".xdmf")

    if MPI.COMM_WORLD.rank == 0:
        in_mesh = meshio.read(str(tmp_file))
        tmp_file.unlink()

        line_mesh = create_mesh(in_mesh, "line", prune_z=True)
        meshio.write(facet_file_name, line_mesh)

        triangle_mesh = create_mesh(in_mesh, "triangle", prune_z=True)
        meshio.write(mesh_file_name, triangle_mesh)
    mesh_comm.Barrier()
    return mesh_file_name, facet_file_name
