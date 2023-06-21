from fenics import *
set_log_level(30)
from xii import *
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

import sys
table_type = 'full'#sys.argv[1]


Ns = [16, 32, 64, 128]

Rs = [1, 10, 50, 100]

drag_ref_values = {}
drag_ref_values[1] = 26.074
drag_ref_values[10] = 4.099
drag_ref_values[50] = 1.868
drag_ref_values[100] = 1.471

pressure_ref_values = {}
pressure_ref_values[1] = 13.63193
pressure_ref_values[10] = 2.36579   
pressure_ref_values[50] = 1.28824
pressure_ref_values[100] = 1.10257

visc_ref_values = {}
visc_ref_values[1] = 12.44233
visc_ref_values[10] = 1.73098
visc_ref_values[50] = 0.57811
visc_ref_values[100] = 0.36826

meshes, ffs = [], []
for N in Ns:
    domain = Rectangle(Point(-L1, -Width/2), Point(L2, Width/2))-Circle(Point(0,0), 1, 8*N)
    mesh = generate_mesh(domain, N)
    ff = mark_boundaries_cylinder(mesh)

    meshes.append(mesh)
    ffs.append(ff)

ix = 0
    
for func_space in func_spaces:
    
    vdrag_ffile = File(f'plots/vdrag_{func_space.__name__}.pvd')
    pdrag_ffile = File(f'plots/pdrag_{func_space.__name__}.pvd')
    
    u_ffile = File(f'plots/u_{func_space.__name__}.pvd')
    p_ffile = File(f'plots/p_{func_space.__name__}.pvd')
    
    print('\nFunction space: ', func_space.__name__)
    
    for ix_R, R in enumerate(Rs):
        print(f'R = {R}')
        
        
        if table_type == 'full':
            print(f'N & $p_{{drag}}$ & $u_{{drag}}$ & $\\beta_{{drag}}$ & $\\omega_{{drag}}$ & $\\epsilon_{{diff}}$ \\\\')
        else:
            print(f'R & $$\\omega_{{drag}}$ & $\\epsilon_{{diff}}$  & M \\\\')
            
        for ix, N in enumerate(Ns):
            mesh = meshes[ix]
            ff = ffs[ix]
            
            nu = 2/R
            
            desc = f'{func_space.__name__}_N{N:g}_R{R:g}'
            
            W = func_space(mesh)
            
            # read solution
            h5 = HDF5File(mesh.mpi_comm(), f'baseflows/qp0_{desc}.h5', 'r')
            qp0 = Function(W)
            h5.read(qp0, 'u')
            h5.close()
            
            u0, p0, foo = qp0.split(deepcopy=True)
            
            omega_drag  = drag_omega(u0, p0, nu, ff)
            p_drag, u_drag, beta_drag = drag_beta(u0, p0, nu)
            
            eps_diff = 10**2*(omega_drag - beta_drag)/omega_drag
            
            error_omega_drag = 100*abs(drag_ref_values[R] - omega_drag)/drag_ref_values[R]
            error_beta_drag = 100*abs(drag_ref_values[R] - beta_drag)/drag_ref_values[R]
            
            error_pressure_drag = 100*abs(pressure_ref_values[R] - p_drag)/pressure_ref_values[R]
            
            error_visc_drag = abs(100*(visc_ref_values[R] - u_drag)/visc_ref_values[R])
            
            if table_type == 'full':
                print(f'{N:3.0f}& {p_drag:5.3f} \scriptsize ({error_pressure_drag:1.1f}\\%) &  {u_drag:5.3f} \scriptsize ({error_visc_drag:1.1f}\\%) & {beta_drag:5.3f} & {omega_drag:5.3f} &  {eps_diff:1.3f} \\\\')
            else:
                print(f'{R:3.0f}& {omega_drag:5.3f} ({error_omega_drag:1.1f}\\%) &  {eps_diff:1.3f} & {N:3.0f} \\\\')

        # compute normal stress
        XD = Constant((1,0))
        n = Expression(('x[0]', 'x[1]'), degree=1)
        submesh = EmbeddedMesh(ff, 1)
        ns_v = project(dot((grad(u0)+grad(u0).T)*XD, n), FunctionSpace(mesh, 'CG', 1))
        ns_v = Trace(ns_v, submesh)
        ns_v = interpolate(ns_v, FunctionSpace(submesh, 'CG', 1))
        ns_v.rename('$(D(u)\chi)\cdot n$', '0')
        
        ns_p = -p0*dot(n,XD)
        ns_p = project(ns_p, FunctionSpace(mesh, 'CG', 1))
        ns_p = Trace(ns_p, submesh)
        ns_p = interpolate(ns_p, FunctionSpace(submesh, 'CG', 1))
        ns_p.rename('$p(n \cdot \chi)$', '0')
                
        vdrag_ffile << (ns_v, ix_R)
        pdrag_ffile << (ns_p, ix_R)
        
        if func_space is TaylorHood:
            u0 = interpolate(u0, VectorFunctionSpace(refine(mesh), 'CG', 1, 2))
        u0.rename('u', 'u')
        p0.rename('p', 'p')
        u_ffile << (u0, ix_R)
        p_ffile << (p0, ix_R)
            
            
        print('')