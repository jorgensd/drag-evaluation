import os

import numpy as np
from fenics import *

from .solvers import D, Stokes

__all__ = ["drag_beta", "drag_omega", "TaylorHood", "CR1P0", "P2P0", "ScottVogelius", "read_up",
           "read_h5_mesh", "read_h5_mesh", "read_h5_func", "write_HDF5file", "mark_boundaries_cylinder", "mark_boundaries_expduct", "get_computed_nus_cylinder", "get_closest_computed_Rs_expduct"
]

def drag_beta(u, p, nu, XD=Constant((1, 0))):
    """Compute drag force on cylinder

    Args:
        u (Function): velocity
        p (Function): pressure
        nu (float): _
        XD (_type_, optional): _description_. Defaults to Constant((1, 0)).

    Returns:
        _type_: _description_
    """
    
    mesh = u.function_space().mesh()
    ff = mark_boundaries_cylinder(mesh)

    ds = Measure("ds", domain=mesh, subdomain_data=ff, subdomain_id=1)

    n = Expression(('x[0]', 'x[1]'), degree=4)
    #tau = as_vector([n[1], -n[0]])

    visc_drag = Constant(nu)*dot((grad(u)+grad(u).T)*XD, n)*ds  # viscous drag
    p_drag = -p*dot(n, XD)*ds  # pressure drag

    visc_drag = assemble(visc_drag)
    p_drag = assemble(p_drag)
    total_drag = visc_drag + p_drag
    
    return p_drag, visc_drag, total_drag





def drag_omega(u, p, nu, ff):

    W = TaylorHood(u.function_space().mesh())
    bc_dict = {1: Constant((1, 0)), 3: Constant((0, 0)), 4: Constant((0, 0))}
    model = Stokes(W, nu=1, bc_dict=bc_dict, ff=ff)
    xi = model.solve()
    v, foo, bar = xi.split(deepcopy=True)

    drag = Constant(0.5*nu)*inner(D(u), D(v))*dx
    drag += inner(grad(u)*u, v)*dx - div(v)*p*dx
    drag_value = -assemble(drag)

    return drag_value


def TaylorHood(mesh):
    # Make FE space with Taylor-Hood elements with given degree
    P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    R = FiniteElement("R", mesh.ufl_cell(), 0)
    TH = MixedElement([P2, P1, R])
    return FunctionSpace(mesh, TH)


def CR1P0(mesh):
    CR = VectorElement("Crouzeix-Raviart", mesh.ufl_cell(), 1)
    P0 = FiniteElement("DG", mesh.ufl_cell(), 0)
    R = FiniteElement("R", mesh.ufl_cell(), 0)
    CRP0 = MixedElement([CR, P0, R])
    return FunctionSpace(mesh, CRP0)


def P2P0(mesh):
    P2 = VectorElement("CG", mesh.ufl_cell(), 2)
    P0 = FiniteElement("DG", mesh.ufl_cell(), 0)
    R = FiniteElement("R", mesh.ufl_cell(), 0)
    P2P0 = MixedElement([P2, P0, R])
    return FunctionSpace(mesh, P2P0)

def ScottVogelius(mesh):
    P4 = VectorElement("Lagrange", mesh.ufl_cell(), 4)
    P3 = FiniteElement("DG", mesh.ufl_cell(), 3)
    R = FiniteElement("R", mesh.ufl_cell(), 0)
    SV = MixedElement([P4, P3, R])

    W = FunctionSpace(mesh, SV)

    return W


def read_up(mpicomm, loc, desc):
    # We use an adaptive solver so the mesh might have changed from the initial one
    meshf = HDF5File(mpicomm, loc + 'mesh' + desc + '.h5', "r")
    mesh = Mesh()
    meshf.read(mesh, "/mesh", False)
    meshf.close()

    ff = mark_boundaries_cylinder(mesh)

    # Make FE space with Taylor-Hood elements with given degree
    P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    LM = FiniteElement('R', mesh.ufl_cell(), 0)

    TH = MixedElement([P2, P1, LM])
    W = FunctionSpace(mesh, TH)

    # Read in closest solution wrt nu
    upf = HDF5File(mesh.mpi_comm(), loc + 'up0' + desc + '.h5', "r")
    up = Function(W)
    upf.read(up, "/up")
    upf.close()
    return up


def read_h5_mesh(mpi, loc, varname):
    meshf = HDF5File(mpi, loc, "r")
    mesh = Mesh()
    meshf.read(mesh, varname, False)
    meshf.close()
    return mesh

def read_h5_func(W, loc, varname):
    upf = HDF5File(W.mesh().mpi_comm(), loc, "r")
    up = Function(W)
    upf.read(up, varname)
    upf.close()
    return up


def write_HDF5file(var, mesh, fname, varname):
    file = HDF5File(mesh.mpi_comm(), fname, "w")
    file.write(var, varname)
    file.close()


def mark_boundaries_cylinder(mesh):
    '''
    make facetfunction ff with boundary marking for expanding duct mesh
    with inlet=3, outlet=4 and the rest=1

    Args: 
        mesh (fenics mesh): domain mesh

    Returns:
        fenics facetfunction

    '''

    ff = MeshFunction('size_t', mesh, mesh.topology().dim() - 1, 0)

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], np.max(mesh.coordinates()[:, 0])) and on_boundary

    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], np.min(mesh.coordinates()[:, 0])) and on_boundary

    class Top(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], np.max(mesh.coordinates()[:, 1])) and on_boundary

    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], np.min(mesh.coordinates()[:, 1])) and on_boundary

    class Circle(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0]**2.0+x[1]**2.0, 1.0, 0.5*mesh.hmin())

    right = Right()
    left = Left()
    bottom = Bottom()
    right = Right()
    top = Top()
    circle = Circle()

    top.mark(ff, 3)
    bottom.mark(ff, 3)
    left.mark(ff, 3)
    right.mark(ff, 4)
    circle.mark(ff, 1)

    if mesh.topology().dim() == 3:
        class CubeTop(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[2], np.max(mesh.coordinates()[:, 2])) and on_boundary

        class CubeBottom(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[2], np.min(mesh.coordinates()[:, 2])) and on_boundary

        ctop = CubeTop()
        cbot = CubeBottom()
        ctop.mark(ff, 3)
        cbot.mark(ff, 4)

    return ff


def mark_boundaries_expduct(mesh):
    '''
    make facetfunction ff with boundary marking for expanding duct mesh
    with inlet=3, outlet=4 and the rest=1

    Args: 
        mesh (fenics mesh): domain mesh

    Returns:
        fenics facetfunction

    '''

    ff = MeshFunction('size_t', mesh, mesh.topology().dim() - 1, 0)

    class Boundary(SubDomain):
        def inside(self, x, on_boundary): return on_boundary

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], np.max(mesh.coordinates()[:, 0])) and on_boundary

    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], np.min(mesh.coordinates()[:, 0])) and on_boundary

    bound = Boundary()
    right = Right()
    left = Left()

    bound.mark(ff, 1)
    left.mark(ff, 3)
    right.mark(ff, 4)

    return ff


def get_computed_nus_cylinder(loc, params):
    '''
    Chech which values of R we've already computed the solution for on this particular
    domain and mesh
    '''
    computed_nus = []
    beta = params["beta"]
    disc_desc = '_L{L:n}_W{W:n}_N_{N:n}'.format(**params)

    if not os.path.exists(loc):
        os.makedirs(loc)

    for filename in os.listdir(loc):
        right_beta = (filename.find('up0_beta%g_' % beta) >= 0)
        right_disc = (filename.find(disc_desc) >= 0)

        if right_beta and right_disc:
            nu_str_ix_start = filename.find('nu') + 2
            nu_str_ix_end = filename.find('_L')
            computed_nus.append(filename[nu_str_ix_start:nu_str_ix_end])
    computed_nus = np.array(computed_nus, dtype=float)
    return computed_nus


def get_closest_computed_Rs_expduct(loc, R, params, larger=False):
    '''
    Check which values of R we've already computed the solution for on this particular
    domain and mesh

    Args:
        loc (str): folder to look in
        params (dict): problem parameters

    Returns: 
        closest smaller R (float), or None if none we found

    '''

    computed_Rs = []

    if not os.path.exists(loc):
        os.makedirs(loc)

    for filename in os.listdir(loc):

        mesh_name = '_L1_{L1:n}_L2_{L2:n}_H1_{H1:n}_H2_{H2:n}'.format(**params)
        disc_desc = '_alpha_{alpha:n}_k_{k:n}_DG_{DG:n}_N_{N:n}'.format(
            **params)

        right_domain = (filename.find(mesh_name) >= 0)
        right_discretization = (filename.find(disc_desc) >= 0)

        if right_domain and right_discretization:
            R_str_ix_start = filename.find('R') + 2
            R_str_ix_end = filename.find('_alpha')
            computed_Rs.append(filename[R_str_ix_start:R_str_ix_end])
    computed_Rs = np.unique(np.array(computed_Rs, dtype=float))

    if len(computed_Rs) < 1:
        return None

    ix_previously_computed_smaller_Rs = np.where(computed_Rs-R < 0)[0]
    if larger:
        ix_previously_computed_smaller_Rs = np.where(computed_Rs-R > 0)[0]
    if len(ix_previously_computed_smaller_Rs) > 0:
        previously_computed_smaller_Rs = computed_Rs[ix_previously_computed_smaller_Rs]
        closest_R_ix = np.argmin(
            np.abs(np.array(previously_computed_smaller_Rs-R, dtype=float) - R))
        closest_R = previously_computed_smaller_Rs[closest_R_ix]
    else:
        closest_R = None

    return closest_R
