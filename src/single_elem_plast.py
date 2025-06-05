#%% --------------------------------- Imports ---------------------------------

# Compatbility issues between v0.8.0 and v0.9.0
# https://fenicsproject.discourse.group/t/how-did-the-function-object-change-in-dolfinx-v0-9-0/16085
# https://github.com/FEniCS/web/pull/192/files

# Import FEnicSx/dolfinx
import dolfinx

# For numerical arrays
import numpy as np

# specific functions from ufl modules
import ufl
from ufl import TestFunction, TrialFunction, grad, tr, Identity, \
                inner, derivative, sqrt, dev, le, conditional, inv, det, dot

# For MPI-based parallelization
from mpi4py import MPI

# PETSc solvers
from petsc4py import PETSc

# specific functions from dolfinx modules
from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

# basix finite elements (necessary for dolfinx v0.8.0)
import basix
from basix.ufl import element

# Matplotlib for plotting
import matplotlib.pyplot as plt

#%% -------------------------------- Geometry ---------------------------------
domain = mesh.create_unit_cube(comm=MPI.COMM_WORLD, nx=10, ny=10, nz=10, 
                               cell_type=mesh.CellType.hexahedron)

# 3D 
dim = domain.topology.dim 

print(domain)

# Define the volume integration measure "dx" 
# also specify the number of volume quadrature points.
deg_quad = 1
dx = ufl.Measure('dx', domain=domain,
                 metadata={'quadrature_degree': deg_quad,
                           "quadrature_scheme": "default"})

#%% ------------------------------ Boundaries ---------------------------------
def left(x):
    return np.isclose(x[0], 0)
def right(x):
    return np.isclose(x[0], 1)

# mark facets
boundaries = [(1, left), (2, right)]

fdim = dim - 1
facet_indices, facet_markers = [], []

for (marker, locator) in boundaries:
    facets = mesh.locate_entities_boundary(domain, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))

# Format the facet indices and markers as required for use in dolfinx.
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)

# Add these marked facets as "mesh tags" for later use in BCs.
facet_tags = mesh.meshtags(domain, fdim, facet_indices[sorted_facets],
                           facet_markers[sorted_facets])

# Define the boundary integration measure "ds" using the facet tags,
# also specify the number of surface quadrature points.
ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tags,
                 metadata={'quadrature_degree':deg_quad})

#%% ----------------------------- Function spaces -----------------------------

# vector elements and function space
U2 = element("Lagrange", domain.basix_cell(), 2, shape=(3,))
V = fem.functionspace(domain, U2)

# tensor function space 
M1 = element("Lagrange", domain.basix_cell(), 1, shape=(3,3)) 
V1 = fem.functionspace(domain, M1)

u = fem.Function(V)
u_old = fem.Function(V)
T_old = fem.Function(V1)
Fp_inv_old = fem.Function(V1)

u_trial = TrialFunction(V)
u_test = TestFunction(V)

#%%  -------------------------------- Material --------------------------------

E = fem.Constant(domain, 70.0e3)
nu = fem.Constant(domain, 0.33)

k = fem.Constant(domain, 350.0)

lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)


def return_mapping_hypoelastic_von_mises(v, Fp_old_inv):
    Id = Identity(3)
    F_ = grad(v) + Id

    Fe_tr = F_*Fp_old_inv
    E_tr = 0.5*(Fe_tr.T*Fe_tr-Id)


    P2e_tr = lmbda * tr(E_tr) * Id + 2 * mu * E_tr
    sigma_bar_tr = sqrt(1.5*inner(dev(P2e_tr),dev(P2e_tr)))
    f_trial = sigma_bar_tr - k

    deqps_ = conditional(le(f_trial, 0.0),0.0, np.sqrt(1.5)*f_trial/(3*mu))
    Fp_new_inv = conditional(le(f_trial, 0.0),Fp_old_inv,Fp_old_inv*(
        Id - 1.5*f_trial/(3*mu)*P2e_tr/sigma_bar_tr))
    Fe_new = F_ * Fp_new_inv
    E_new = 0.5 * (Fe_new.T*Fe_new-Id)
    P2e_new = lmbda * tr(E_new) * Id + 2 * mu * E_new

    P1_new = Fe_new * P2e_new * Fp_new_inv.T

    return P1_new, Fp_new_inv, deqps_

P1, Fp_inv, deqps = return_mapping_hypoelastic_von_mises(u, Fp_inv_old)

F = grad(u) + Identity(3)
J = det(F)
Fe = F*Fp_inv
P2e = inv(Fe)*P1*inv(Fp_inv.T)
T = 1/J * Fe * P2e * Fe.T

#%% ------------------------------- Weak forms --------------------------------

Res = ufl.inner(P1, grad(u_test))*dx

#%% --------------------------- Initial conditions ----------------------------

# A function for constructing the identity matrix.
#
# To use the interpolate() feature, this must be defined as a 
# function of x.
def identity(x):
    values = np.zeros((domain.geometry.dim*domain.geometry.dim,
                      x.shape[1]), dtype=np.float64)
    values[0] = 1
    values[4] = 1
    values[8] = 1
    return values

# interpolate the identity onto the tensor-valued Cv function.
Fp_inv_old.interpolate(identity)  
#%% --------------------------- Boundary Conditions ---------------------------

def Ramp(t, time_total):
    if t < time_total/2:
        return 2 * t/time_total
    elif t < 3*time_total/4:
        return 2.0 - 2 * t/time_total
    else:
        return 2 * t/time_total - 1

disp_total = -0.1

# later set value to disp_total*Ramp(t,time_total)
disp_bc = fem.Constant(domain, PETSc.ScalarType(0.0))

left_dofs_u1 = fem.locate_dofs_topological(V.sub(0), facet_tags.dim,
                                           facet_tags.find(1))
left_dofs_u2 = fem.locate_dofs_topological(V.sub(1), facet_tags.dim,
                                           facet_tags.find(1))

right_dofs_u1 = fem.locate_dofs_topological(V.sub(0), facet_tags.dim,
                                            facet_tags.find(2))
right_dofs_u2 = fem.locate_dofs_topological(V.sub(1), facet_tags.dim,
                                            facet_tags.find(2))

bc_0 = fem.dirichletbc(0.0, right_dofs_u1, V.sub(0))
# bc_1 = fem.dirichletbc(0.0, right_dofs_u2, V.sub(1))
# bc_2 = fem.dirichletbc(0.0, left_dofs_u2, V.sub(1))
bc_3 = fem.dirichletbc(disp_bc, left_dofs_u1, V.sub(0))

bcs = [bc_0, bc_3]

#%% --------------------------------- Solver ----------------------------------

problem = NonlinearProblem(Res, u, bcs, derivative(Res, u, u_trial))

solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-8
solver.atol = 1e-8
solver.max_it = 50
solver.report = True

#  The Krylov solver parameters.
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
# opts[f"{option_prefix}ksp_type"] = "preonly" # "preonly" works equally well
# opts[f"{option_prefix}pc_type"] = "lu" # do not use 'gamg' pre-conditioner
# opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
# opts[f"{option_prefix}ksp_max_it"] = 30
ksp.setFromOptions()

#%% ------------------------------- Output file -------------------------------
out_file = "results/nonlinear_elasticity.xdmf"

# scalar function space to store output mises stress
S1 = element("Lagrange", domain.basix_cell(), 1) 
V2 = fem.functionspace(domain, S1)

U3 = element("Lagrange",  domain.basix_cell(), 1, shape=(3,))
V3 = fem.functionspace(domain, U3)

u_out = fem.Function(V3, name='u')
u_out.interpolate(u)

T_out = fem.Function(V1, name='T')
T_expr = fem.Expression(P1, V1.element.interpolation_points())
T_out.interpolate(T_expr)

Fp_inv_out = fem.Function(V1, name='Fp')
Fp_inv_expr = fem.Expression(Fp_inv, V1.element.interpolation_points())
Fp_inv_out.interpolate(Fp_inv_expr)

sigma_mises = fem.Function(V2, name='mises')
sigma_mises_expr = fem.Expression(sqrt(1.5*inner(dev(P2e),dev(P2e))),
                                  V2.element.interpolation_points())
sigma_mises.interpolate(sigma_mises_expr)

deqps_out = fem.Function(V2, name='deqps')
deqps_expr = fem.Expression(deqps, V2.element.interpolation_points())
deqps_out.interpolate(deqps_expr)

eqps_out = fem.Function(V2, name='eqps')

surface_stress = fem.form(T_out.sub(0)*ds(1))

with io.XDMFFile(domain.comm, out_file, "w") as xdmf:
    xdmf.write_mesh(domain)
with io.XDMFFile(domain.comm, out_file, "a") as xdmf:
    xdmf.write_function(u_out, 0)
    xdmf.write_function(T_out, 0)
    xdmf.write_function(sigma_mises, 0)

#%% ---------------------------- Solver iterations ----------------------------

# set u to zero for case of reexecution
u.x.petsc_vec.set(0.0)

time_current = 0.0
time_total = 1.0
num_steps = 300
dt = time_total/num_steps

hist_time = np.zeros(num_steps+1)
hist_disp = np.zeros(num_steps+1)
hist_force = np.zeros(num_steps+1)
hist_mises = np.zeros(num_steps+1)
hist_eqps = np.zeros(num_steps + 1)

ii = 0

while (round(time_current + dt, 9) <= time_total):
    time_current += dt
    ii += 1

    disp_bc.value = disp_total*Ramp(time_current,time_total)
    
    (iter, converged) = solver.solve(u)
    assert converged

    # Collect results from MPI ghost processes
    u.x.scatter_forward()
    u_out.interpolate(u)
    Fp_inv_out.interpolate(Fp_inv_expr)
    T_out.interpolate(T_expr)
    sigma_mises.interpolate(sigma_mises_expr)
    deqps_out.interpolate(deqps_expr)
    eqps_out.x.petsc_vec.axpy(1, deqps_out.x.petsc_vec)

    # Collect history output variables
    hist_time[ii] = time_current
    hist_disp[ii] = disp_total*Ramp(time_current,time_total)
    hist_force[ii] = domain.comm.gather(fem.assemble_scalar(surface_stress))[0]
    hist_mises[ii] = np.mean(sigma_mises.x.array)
    hist_eqps[ii] = np.mean(eqps_out.x.array)

    # Write outputs to file
    with io.XDMFFile(domain.comm, out_file, "a") as xdmf:
        xdmf.write_function(u_out, ii)
        xdmf.write_function(T_out, ii)
        xdmf.write_function(sigma_mises, ii)

    # update u_old
    # Update DOFs for next step
    u_old.x.array[:] = u.x.array
    T_old.x.array[:] = T_out.x.array
    Fp_inv_old.x.array[:] = Fp_inv_out.x.array

    print("Time step: ", ii, " | Iterations: ", iter, " | U: ",converged)

#%% ----------------------------- Post-processing -----------------------------
plt.plot(hist_time, -hist_disp)

plt.plot(hist_time,hist_force)

plt.plot(hist_time,hist_mises)

plt.plot(hist_time, hist_eqps)
