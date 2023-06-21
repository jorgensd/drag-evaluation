from fenics import *
set_log_level(20)
from mshr import *
import sys
sys.path.append('../')
from utils import mark_boundaries_cylinder
from utils import *

from solvers import *

    
L1 = 12.8
L2 = 128
Width = 6.4*2


diam = 2

bc_dict = { 1: Constant((0,0)), 3: Constant((1,0)), 4: Constant((1,0))}


func_spaces = [P2P0, CR1P0]

experiments = {}

Ns = [256]


Rs = [1, 10, 50, 100]

meshes, ffs = [], []

for N in Ns:
    domain = Rectangle(Point(-L1, -Width/2), Point(L2, Width/2))-Circle(Point(0,0), 1, 8*N)
    mesh = generate_mesh(domain, N)
    ff = mark_boundaries_cylinder(mesh)

    meshes.append(mesh)
    ffs.append(ff)

ix = 0
    
for func_space in func_spaces:
    
    print('\nFunction space: ', func_space.__name__)
    
    for ix, N in enumerate(Ns):
        print(f'N = {N}')
            
        mesh = meshes[ix]
        ff = ffs[ix]
        
        qp0 = None

        print(f'R   p_drag  visc_drag  beta_drag  omega_drag  eps_diff')
            
        for R in Rs:
            nu = 2/R
            
            desc = f'{func_space.__name__}_N{N:g}_R{R:g}'
            
            W = func_space(mesh)
            model = NavierStokes(W, nu=nu, ff=ff, bc_dict=bc_dict)
            import time
            start = time.time()
            qp0 = model.solve(qp0)
            u0, p0, foo = qp0.split(deepcopy=True)
            solver_time = time.time() - start
            
            
            # Save mesh and solution to .h5
            h5 = HDF5File(mesh.mpi_comm(), f'baseflows/qp0_{desc}.h5', 'w')
            h5.write(mesh, 'mesh')
            h5.write(qp0, 'u')
            h5.close()

            ix += 1
        
        print('')