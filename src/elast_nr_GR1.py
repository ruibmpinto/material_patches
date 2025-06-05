# Import necessary modules from FEniCSx and standard libraries
from mpi4py import MPI
# For PETSc solver options
from petsc4py import PETSc 
from dolfinx import fem, mesh, plot, log
# For NewtonSolver
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import ufl
import numpy as np

import basix
from basix.ufl import element

# Set up logging (optional, to see solver progress)
# log.set_log_level(log.LogLevel.INFO) # Uncomment for more verbose output

# 1. Create a distributed mesh (a unit cube)
comm = MPI.COMM_WORLD
domain = mesh.create_unit_cube(comm, 10, 10, 10,
                               cell_type=mesh.CellType.hexahedron)
tdim = domain.topology.dim # Topological dimension (3 for a cube)
gdim = domain.geometry.dim # Geometrical dimension (3 for 3D space)

# 2. Define the function space for vector displacements
# Use Lagrange elements of order 1 for each component of the displacement vector
vector_element = element("Lagrange", domain.basix_cell(), 1, shape=(gdim,))
V = fem.FunctionSpace(domain, vector_element)

# 3. Create a fem.Function to store the solution
# This function will be updated by the Newton solver.
u = fem.Function(V)
u.name = "Displacement"

# 4. Define material properties (isotropic linear elastic)
# Young's modulus
E_val = 1.0e5  
# Poisson's ratio
nu_val = 0.3   

E = fem.Constant(domain, fem.petsc.ScalarType(E_val))
nu = fem.Constant(domain, fem.petsc.ScalarType(nu_val))

mu = E / (2 * (1 + nu))
lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))

# 5. Define constitutive relations (strain and stress)
def epsilon(u_):
    """Strain tensor"""
    return ufl.sym(ufl.grad(u_)) # sym(A) = 0.5 * (A + A.T)

def sigma(u_):
    """Stress tensor (Hooke's law for isotropic material)"""
    return lambda_ * ufl.tr(epsilon(u_)) * ufl.Identity(gdim) + \
        2 * mu * epsilon(u_)









# 6. Define the variational problem forms for the Newton solver
# Trial function for the Jacobian (represents the increment delta_u)
u_trial = ufl.TrialFunction(V) 
# Test function
v_test = ufl.TestFunction(V)  

# Body forces (assumed zero)
f_body = fem.Constant(domain, fem.petsc.ScalarType((0, 0, 0)))
L_form = ufl.dot(f_body, v_test) * ufl.dx # Part of the residual

# Residual F(u; v_test) = inner(sigma(u), epsilon(v_test))*dx - L(v_test)
# Here, 'u' is the fem.Function representing the current solution candidate
F_form = ufl.inner(sigma(u), epsilon(v_test)) * ufl.dx - L_form

# Jacobian J(u_trial; v_test) = derivative of F_form with respect to u 
# in direction u_trial
# For this linear problem, J is inner(sigma(u_trial), epsilon(v_test))*dx
J_form = ufl.inner(sigma(u_trial), epsilon(v_test)) * ufl.dx








# 7. Define and apply prescribed displacement boundary conditions
# (This part is the same as in the linear solver version)

class PrescribedDisplacement:
    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=fem.petsc.ScalarType)
        values[0] = 0.1 * x[0]
        values[1] = 0.05 * x[1]
        return values

# fem.Function to hold prescribed values
u_D_fem = fem.Function(V) #
u_D_fem.interpolate(PrescribedDisplacement())

domain.topology.create_connectivity(tdim - 1, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
boundary_dofs = fem.locate_dofs_topological(V, tdim - 1, boundary_facets)
bc = fem.dirichletbc(u_D_fem, boundary_dofs)





# Done

# 8. Set up the NonlinearProblem and NewtonSolver
# For dolfinx versions >= 0.6, NonlinearProblem is often in dolfinx.fem
# For older versions, it might be in dolfinx.fem.petsc
# Let's try dolfinx.fem.petsc.NonlinearProblem first, then dolfinx.fem.NonlinearProblem
try:
    problem = fem.petsc.NonlinearProblem(F_form, u, bcs=[bc], J=J_form)
except AttributeError:
    # Try the other common location if the above fails
    problem = fem.NonlinearProblem(F_form, u, bcs=[bc], J=J_form)


solver = NewtonSolver(domain.comm, problem)

# Set solver parameters
solver.convergence_criterion = "incremental" # Check based on the norm of the solution increment
solver.rtol = 1e-8  # Relative tolerance
solver.atol = 1e-10 # Absolute tolerance
solver.max_it = 10  # Maximum number of iterations (should be 1 or 2 for linear problem)
solver.report = True # Print convergence reason

# Configure the KSP (Krylov Subspace Method) solver used within Newton
# For this linear problem, a direct solver (LU) is efficient and robust.
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly" # Use KSP preonly for direct solver
opts[f"{option_prefix}pc_type"] = "lu"       # Use LU factorization as preconditioner (effectively direct solve)
# For very large systems, one might use iterative solvers like GMRES with a preconditioner:
# opts[f"{option_prefix}ksp_type"] = "gmres"
# opts[f"{option_prefix}pc_type"] = "sor" # or "hypre", "ilu", etc.
ksp.setFromOptions()








# 9. Solve the problem
# Set initial guess for u if desired (default is zero, which is fine)
# u.x.array[:] = 0.0 # Explicitly set to zero (already default for new fem.Function)

num_iterations, converged = solver.solve(u) # u is updated in-place

if comm.rank == 0:
    if converged:
        print(f"Newton-Raphson solver converged in {num_iterations} iteration(s).")
    else:
        print(f"Newton-Raphson solver DID NOT converge after {num_iterations} iterations.")










# 10. Post-processing (optional: save results to XDMF for ParaView)
try:
    from dolfinx.io import XDMFFile
    with XDMFFile(domain.comm, "linear_elastic_cube_newton.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        # Make sure to pass the updated u function
        xdmf.write_function(u)
    if comm.rank == 0:
        print("Solution saved to linear_elastic_cube_newton.xdmf")
except ImportError:
    if comm.rank == 0:
        print("XDMFFile not available. Skipping saving solution to file.")

# Verification: Approximate max displacement
if comm.rank == 0:
    u_magnitude = ufl.sqrt(ufl.dot(u, u))
    V_scalar = fem.FunctionSpace(domain, ("Lagrange", 1))
    u_mag_expr = fem.Expression(u_magnitude, V_scalar.element.interpolation_points())
    u_mag_func = fem.Function(V_scalar)
    u_mag_func.interpolate(u_mag_expr)
    
    max_disp_local = np.max(u_mag_func.x.array) if u_mag_func.x.array.size > 0 else 0.0
    max_disp_global = domain.comm.allreduce(max_disp_local, op=MPI.MAX)
    print(f"Approximate maximum displacement magnitude (Newton solution): {max_disp_global:.4e}")