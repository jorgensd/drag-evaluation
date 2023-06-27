from fenics import *

'''
Functions solving Navier-Stokes equations with/without slip
'''
__all__ = ["Stokes", "NO_SLIP", "NavierStokes"]

NO_SLIP = 99

class Stokes:
    '''
    Input args:
        ff: facet function with boundary markers
        bound_conds (dict): dict with the key giving the marker (int) and the value (Function)
        degree (int): polynomial degree
        W (df.FunctionSpace): Function space for the solution
    '''
    
    def __init__(self, W, nu, ff, bc_dict):
        self.W = W
        self.nu = nu        
        self.ff = ff
        self.bc_dict = bc_dict
    
    def a(self):
        # Define variational problem
        (u, p, rho) = TrialFunctions(self.W)
        (v, q, phi) = TestFunctions(self.W)
        
        dx = Measure('dx', domain=self.W.mesh())

        a =  Constant(0.5*self.nu)*inner(D(u), D(v))*dx
        a += - div(v)*p*dx - q*div(u)*dx
        a +=  + rho*q*dx + phi*p*dx

        return a
    
    def L(self):
        (v, q, rho) = TestFunctions(self.W)
        
        zero_function = Constant((0.0,) * self.W.mesh().topology().dim())
        dx = Measure('dx', domain=self.W.mesh())
        L =  dot(zero_function,v)*dx
        
        return L
    
    def get_bcs(self):
        bcs = []
        for marker, func in self.bc_dict.items():
            bcs.append( DirichletBC(self.W.sub(0), func, self.ff, marker) )
        return bcs

    def solve(self):

        up = Function(self.W)     
        solve(self.a() == self.L(), up, self.get_bcs(),solver_parameters={'linear_solver':'mumps'})
        return up
    
class NavierStokes(Stokes):
    
    def a(self, up):
        # Define variational problem
        (u, p, rho) = split(up)
        (v, q, phi) = TestFunctions(self.W)
        
        dx = Measure('dx', domain=self.W.mesh())

        a =  Constant(0.5*self.nu)*inner(D(u), D(v))*dx
        a += inner(grad(u)*u, v)*dx - div(v)*p*dx - q*div(u)*dx 

        a +=  + rho*q*dx + phi*p*dx
        
        Velm, Pelm, foo = self.W.ufl_element().sub_elements()
        if Velm.family() == "Crouzeix-Raviart":
            hK = avg(CellDiameter(self.W.mesh()))
            gamma = Constant(1)
            a += (gamma/hK)*inner(jump(u), jump(v))*dS
            
        elif Velm.family() == "Lagrange" and Pelm.family() == "Lagrange" and Velm.degree() == Pelm.degree() == 1:
            hK = CellDiameter(self.W.mesh())
            gamma = -Constant(0.5)
            a += gamma*hK**2*inner(grad(p), grad(q))*dx
        return a

    def stokes_solve(self):
        
        return Stokes(self.W, self.nu, self.ff, self.bc_dict).solve()
        
    def solve(self, up=None):
            
        # Set up initial guess for Newton solver
        if up is None: 
            up = self.stokes_solve()
        else: 
            up = interpolate(up, self.W)
        
        PETScOptions.set("mat_mumps_icntl_14", 600)
        
        F = self.a(up) - self.L() 
        JF = derivative(F, up, TrialFunction(self.W))

        bcs = self.get_bcs()
        
        problem = NonlinearVariationalProblem(F, up, bcs, JF)
        
        solver = NonlinearVariationalSolver(problem)
        prm = solver.parameters["newton_solver"]

        prm['absolute_tolerance'] = 5E-4
        prm['relative_tolerance'] = 5E-5
        prm['maximum_iterations'] = 15
        prm['relaxation_parameter'] = 1.0
        prm["linear_solver"] = "mumps"
        
        solver.solve()
            
        return up


    
def D(u):
    gradu = grad(u)
    return gradu + gradu.T
