import time
from pathlib import Path

from fenics import (
    Constant,
    HDF5File,
    Mesh,
    MeshValueCollection,
    XDMFFile,
    cpp,
    set_log_level,
)

from drag_evaluation.mesh_generation import generate_cylinder_mesh
from drag_evaluation.solvers import NavierStokes
from drag_evaluation.utils import CR1P0, P2P0

set_log_level(20)


L1 = 12.8
L2 = 128
Width = 6.4 * 2


bc_dict = {1: Constant((0, 0)), 3: Constant((1, 0)), 4: Constant((1, 0))}


func_spaces = [P2P0, CR1P0]


Ns = [8]


Rs = [1, 10, 50, 100]

meshes, ffs = [], []

for N in Ns:
    mesh_file = Path(f"mesh_{1./N:.3e}.xdmf")
    mesh_file, facet_file = generate_cylinder_mesh(
        mesh_file,
        12.8,
        128,
        12.8,
        0,
        0,
        1,
        1.0 / N,
        {"walls": 3, "inlet": 3, "outlet": 4, "obstacle": 1},
    )
    mesh = Mesh()
    with XDMFFile(str(mesh_file.absolute())) as xdmf:
        xdmf.read(mesh)
    mvc = MeshValueCollection("size_t", mesh, 1)
    with XDMFFile(str(facet_file.absolute())) as xdmf:
        xdmf.read(mvc, "name_to_read")
    ff = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    meshes.append(mesh)
    ffs.append(ff)

ix = 0

for func_space in func_spaces:
    print("\nFunction space: ", func_space.__name__)

    for ix, N in enumerate(Ns):
        print(f"N = {N}")

        mesh = meshes[ix]
        ff = ffs[ix]

        qp0 = None

        print("R   p_drag  visc_drag  beta_drag  omega_drag  eps_diff")

        for R in Rs:
            nu = 2 / R

            desc = f"{func_space.__name__}_N{N:g}_R{R:g}"

            W = func_space(mesh)
            model = NavierStokes(W, nu=nu, ff=ff, bc_dict=bc_dict)
            start = time.time()
            qp0 = model.solve(qp0)
            u0, p0, foo = qp0.split(deepcopy=True)
            solver_time = time.time() - start

            # Save mesh and solution to .h5
            h5 = HDF5File(mesh.mpi_comm(), f"baseflows/qp0_{desc}.h5", "w")
            h5.write(mesh, "mesh")
            h5.write(qp0, "u")
            h5.close()

            ix += 1

        print("")
