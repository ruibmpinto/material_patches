#%% --------------------------------- Imports ---------------------------------

# Compatbility issues between v0.8.0 and v0.9.0
# https://fenicsproject.discourse.group/t/how-did-the-function-object-change-in-dolfinx-v0-9-0/16085
# https://github.com/FEniCS/web/pull/192/files

import os

# Import FEnicSx/dolfinx
import dolfinx

# For numerical arrays
import numpy as np

# specific functions from ufl modules
import ufl
from ufl import TestFunction, TrialFunction, grad, tr, Identity, \
                inner, derivative, sqrt, dev, le, conditional, inv, det, dot, \
                nabla_div

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

import pickle as pkl

from fenicsx_plotly import plot

#%% ----------------------------- Material patch ------------------------------

filename = f"/Users/rbarreira/Desktop/machine_learning/material_patches/" + \
           f"2025_06_05/" + \
           f"material_patches_generation_2d_quad4/material_patch_0/" + \
           f"material_patch/material_patch_attributes.pkl"

with open(filename, 'rb') as file:
    patch = pkl.load(file)

for node_label, displacements in patch['mesh_boundary_nodes_disps'].items():
    # print(f"  Node {node_label}, displacement = {displacements}")

    node_coords = patch['mesh_nodes_coords_ref'][node_label]
    # print(f'Node coords: {node_coords}')
#%% -------------------------------- Geometry ---------------------------------

domain = mesh.create_unit_square(comm=MPI.COMM_WORLD, nx=3, ny=3, 
                               cell_type=mesh.CellType.quadrilateral)


# 3D 
# Topological dimension (3 for a cube)
tdim = domain.topology.dim 
print(f'Topological dimension: {tdim}')
# Geometrical dimension (3 for 3D space)
gdim = domain.geometry.dim
print(f'Geometrical dimension: {gdim}')

# Define the volume integration measure "dx" 
# also specify the number of volume quadrature points.
deg_quad = 2
dx = ufl.Measure('dx', domain=domain,
                 metadata={'quadrature_degree': deg_quad,
                           "quadrature_scheme": "default"})


# Define the boundary integration measure "ds" using the facet tags,
# also specify the number of surface quadrature points.
ds = ufl.Measure('ds', domain=domain)
        # metadata={'quadrature_degree':deg_quad,
        # "quadrature_scheme": "default"})
#%% ----------------------------- Function spaces -----------------------------

# vector elements and function space
vector_element = element(family="Lagrange", cell=domain.basix_cell(), degree=1,
                         shape=(gdim,))
V = fem.functionspace(domain, vector_element)

# tensor function space 
tensor_element = element(family="Lagrange", cell=domain.basix_cell(), degree=1,
                          shape=(gdim,gdim)) 
V1 = fem.functionspace(domain, tensor_element)

u = fem.Function(V)
u.name = "displacement"

# u_old = fem.Function(V)

u_trial = TrialFunction(V)
v_test = TestFunction(V)

#%%  -------------------------------- Material --------------------------------
E = fem.Constant(domain, 70.0e3)
nu = fem.Constant(domain, 0.33)

lmbda =  E * nu / ((1 + nu) * (1 - 2 * nu))
mu =  E / (2 * (1 + nu))

#%% ---------------------------- Constitutive law -----------------------------

# strain and stress
def epsilon(u_):
    """
    Strain tensor:
    
    Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
    """
    return ufl.sym(ufl.grad(u_)) 

def sigma(u_):
    """Stress tensor (Hooke's law for isotropic material) for plane strain"""

    
    # return lmbda * ufl.nabla_div(u_) * ufl.Identity(gdim) + \
    #    2 * mu * epsilon(u_)

    print(f'IMPLEMENT PLANE STRAIN!')
    return ufl.tr(epsilon(u_)) * lmbda * ufl.Identity(gdim) + \
       2 * mu * epsilon(u_)





#%% ------------------------------- Weak forms --------------------------------
# Linear functional: body forces (assumed zero)
f_body = fem.Constant(domain, np.zeros(gdim))
l_form = ufl.dot(f_body, v_test) * dx 

# Bilinear functional
# Residual F(u; v) = inner(sigma(u), epsilon(v))*dx - L(v)
# Here, 'u' is the fem.Function representing the current solution candidate
a_form = ufl.inner(sigma(u), epsilon(v_test)) * dx 


# Derivative of the residual with respect to u, in the direction of u_trial
# J_form = ufl.inner(sigma(u_trial), epsilon(v_test)) * dx
j_gateaux_der = ufl.derivative(a_form, u, u_trial)

residual = a_form - l_form

# Res = ufl.inner(sigma(u_trial), grad(v_test))*dx

#%% --------------------------- Initial conditions ----------------------------

# A function for constructing the identity matrix.
#
# To use the interpolate() feature, this must be defined as a 
# function of x.
def identity(x):
    """
    values: dim 1x9 [1, 0, 0, 0, 1, 0, 0, 0, 1]
    """
    values = np.zeros((domain.geometry.dim*domain.geometry.dim,
                      x.shape[1]), dtype=np.float64)
    values[0] = 1
    values[4] = 1
    values[8] = 1
    return values


#%% --------------------------- Boundary Conditions ---------------------------

# List to hold the DirichletBC objects - it will be updated in the loop
bcs = []

# Function to find DOFs based on coordinates
def find_dofs_and_displacements(v_space, coords, tolerance=1e-6):
    """
    Finds the global DOFs in function space V_space whose coordinates match
    those in coords_array.
    coords_array should be a single (x,y,z) coordinate array.
    Returns a list of global DOF indices.
    """
    # FEniCSx 0.9.0 changed the API for locating DOFs!
    # print(f'v_space.tabulate_dof_coordinates():
    # {v_space.tabulate_dof_coordinates()}')
    dof_coords = np.array(v_space.tabulate_dof_coordinates(), dtype=np.float64)
    
    # for idx_node, node_coord in enumerate(dof_coords):
    #     print(f'FENICSX:     Node {idx_node}: coords({node_coord})')
    #     # print(f'coords {idx_node}: coords({coords})')
        
    # dof_indices = []
    # for idx_node, node_coord in enumerate(dof_coords):
    #     # print(f'Node {idx_node}: coords({node_coord})')
    #     # print(f'coords {idx_node}: coords({coords})')
    #     if np.allclose(node_coord[:gdim], coords, atol=tolerance):
    #         # For a vector space V, dof_coords typically has shape
    #         # (N_dofs, gdim)
    #         # and each row corresponds to a single DOF
    #         # (e.g., x-component of a node).
    #         # The indices are already global.
    #         dof_indices.append(idx_node)
    #         break
    dofs = fem.locate_dofs_geometrical(V, lambda x: np.logical_and(
        np.isclose(x[0], coords[0]), np.isclose(x[1], coords[1])) )

    print(f'    FEniCSX mesh DOF: {dofs}, coords: ({dof_coords[dofs]})')
    # Function fem.dirichletBC only accepts numpy arrays
    # dof_indices = np.array(dof_indices)
    # print(f'dof_indices:{dof_indices}')

    return dofs # dof_indices

def apply_displacement_bc(v_space, coords, displacement_values):
    """
    Apply displacement boundary condition at given coordinates.
    """
    # Find DOFs at the specified coordinates
    dofs = find_dofs_and_displacements(v_space, coords)
    
    if len(dofs) == 0:
        return None
    
    # Create function to hold the displacement values
    # u_bc = fem.Function(v_space)
    
    # For a 2D vector space, we have 2 DOFs per node 
    # (x and y components)
    # The DOFs are ordered as:
    
    # Apply displacement values to the corresponding DOFs
    # for idx_dof, dof in enumerate(dofs):
    #     if idx_dof < len(displacement_values):
    #         u_bc.x.array[dof] = displacement_values[idx_dof % gdim]
    # print(f'type of displacement_values: {type(displacement_values)}')
    print(f'displacement_values: {displacement_values}')

    # print(f'type of dofs: {type(dofs)}')
    # print(f'dofs: {dofs}')

    # dofs = np.array(dofs, dtype=np.int32)

    # https://jsdokken.com/dolfinx-tutorial/chapter2/linearelasticity_code.html
    return  fem.dirichletbc(displacement_values, dofs, V)


# # mark facets
# boundaries = [(1, left), (2, right)]

# fdim = gdim - 1
# facet_indices, facet_markers = [], []

# for (marker, locator) in boundaries:
#     facets = mesh.locate_entities_boundary(domain, fdim, locator)
#     facet_indices.append(facets)
#     facet_markers.append(np.full_like(facets, marker))

# # Format the facet indices and markers as required for use in dolfinx.
# facet_indices = np.hstack(facet_indices).astype(np.int32)
# facet_markers = np.hstack(facet_markers).astype(np.int32)
# sorted_facets = np.argsort(facet_indices)

# # Add these marked facets as "mesh tags" for later use in BCs.
# facet_tags = mesh.meshtags(domain, fdim, facet_indices[sorted_facets],
#                            facet_markers[sorted_facets])


# disp_bc = fem.Constant(domain, PETSc.ScalarType(0.0))

# left_dofs_u1 = fem.locate_dofs_topological(V.sub(0), facet_tags.dim,
#                                            facet_tags.find(1))
# left_dofs_u2 = fem.locate_dofs_topological(V.sub(1), facet_tags.dim,
#                                            facet_tags.find(1))

# right_dofs_u1 = fem.locate_dofs_topological(V.sub(0), facet_tags.dim,
#                                             facet_tags.find(2))
# right_dofs_u2 = fem.locate_dofs_topological(V.sub(1), facet_tags.dim,
#                                             facet_tags.find(2))

# bc_0 = fem.dirichletbc(0.0, right_dofs_u1, V.sub(0))
# # bc_1 = fem.dirichletbc(0.0, right_dofs_u2, V.sub(1))
# # bc_2 = fem.dirichletbc(0.0, left_dofs_u2, V.sub(1))
# bc_3 = fem.dirichletbc(disp_bc, left_dofs_u1, V.sub(0))

# bcs = [bc_0, bc_3]

#%% --------------------------------- Solver ----------------------------------


# F = grad(u) + Identity(3)
# J = det(F)


problem = NonlinearProblem(F=residual, u=u, bcs=bcs,
                           J=derivative(residual, u, u_trial))



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
opts[f"{option_prefix}ksp_type"] = "preonly" 
opts[f"{option_prefix}pc_type"] = "lu"
# opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
# opts[f"{option_prefix}ksp_max_it"] = 30
ksp.setFromOptions()

#%%  -------------------------------- Solution --------------------------------

# Incremental loading
num_increments = 5

# Lists to store results for plotting
u_magnitudes_applied = []
u_max_values_domain = []
u_history = []
# Dictionary to store reaction forces: 
# {increment_idx: {node_label: [Rx, Ry, Rz]}}
forces_internal = {}



# Create output directory
output_dir = "output_incremental_disp"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)




# ----------------------------- Timestepping Loop -----------------------------
print("\n--- Starting Incremental Displacement Loading ---")
for idx_inc in range(num_increments):
    print(f"Increment {idx_inc + 1}/{num_increments}:")

    # Clear any previous boundary conditions
    bcs.clear()

    # Apply boundary conditions based on patch data for this increment
    # Read node labels as keys from dictionary 
    # patch['mesh_boundary_nodes_disps']

    # Does the node numbering match??!!
    # Can I import the mesh directly into FEniCSx?
    for node_label, displacements in \
        patch['mesh_boundary_nodes_disps'].items():
        
        # Retrieve nodal coordinates from dictionary
        # patch['mesh_nodes_coords_ref']
        node_coords = patch['mesh_nodes_coords_ref'][node_label]  
        # print(f'    Node label: {node_label}, coords:({node_coords})')


        # Find node labels in dofs in the FEniCSx FE mesh by the nodal 
        # coordinates from the dictionary patch._mesh_nodes_coords_ref
        # dofs = find_dofs_by_coordinates(V, node_coords)

        disp_array = np.array(displacements, dtype=np.float64)
        # print(f'disp_array: {disp_array}')
        displacement_values = disp_array
        # Time series data
        # if idx_inc < disp_array.shape[0]:
        #     # Use correct time value
        #     displacement_values = disp_array[idx_inc, :]
        # else:
        #     # Use last value if beyond range
        #     displacement_values = disp_array[-1, :]  

        # Apply boundary condition
        bcs.append(apply_displacement_bc(V, node_coords, displacement_values))

        # if bc is not None:
        #     bcs.append(bc)
        #     print(f'   Node {node_label}: ' + \
        #             f'coords=({node_coords[0]:.3f},{node_coords[1]:.3f}), ' + \
        #             f'disp=({displacement_values[0]:.6f}, ' + \
        #             f'{displacement_values[1]:.6f})')


    # if len(dofs) > 0:
    #     # Create a function to represent the displacement 
    #     # at this node for this increment
    #     # Get the displacement for this increment
    #     displacement_value = displacements #[:, idx_inc] 
    #     print(f'displacements: {displacements}')

    #     u_dirichlet = fem.Function(V)
    #     print(f'np.shape(u_dirichlet.x.array[dofs]): {np.shape(
    #         u_dirichlet.x.array[dofs])}')
    #     # How to impose (x,y)-displacements at a given node.
    #     # Different DOFS?? What is the DOFs order???
    #     u_dirichlet.x.array[dofs] = displacement_value
        
    #     bcs.append(fem.dirichletbc(u_dirichlet, dofs))

    #     print(f"   Node {node_label}, displacement = {displacement_value}")
    # else:
    #     print(f"   Warning: Node {node_label} not found in FEniCSx mesh.")
    #     print(f"   Warning: Node coords {node_coords}.")


    # Update the problem with the new boundary conditions
    problem.bcs = bcs

    # Solve the nonlinear problem
    try:
        num_iterations, converged = solver.solve(u)
        if not converged:
            print(f"Solver did not converge at increment {idx_inc + 1}. " + \
                  f"Stopping.")
            break

        print(f"    Converged in {num_iterations} iterations.")

        # Save solution
        u.x.scatter_forward() 

        with io.XDMFFile(
            domain.comm,
            f"{output_dir}/displacement_increment_{idx_inc+1:03d}.xdmf", "w"
            ) as xdmf:

            xdmf.write_mesh(domain)
            xdmf.write_function(u)

    except Exception as exc:
        print(f'   Error at increment {idx_inc+1}: {exc}.')
        break

print('------------- Incremental Displacement Loading Complete --------------')



#%% ------------------------------- Output file -------------------------------
out_file = "results/linear_elasticity.xdmf"

u_out = fem.Function(V, name='u')
u_out.interpolate(u)


with io.XDMFFile(domain.comm, out_file, "w") as xdmf:
    xdmf.write_mesh(domain)
with io.XDMFFile(domain.comm, out_file, "a") as xdmf:
    xdmf.write_function(u_out, 0)

#%% ---------------------------- Solver iterations ----------------------------

# # set u to zero for case of reexecution
# u.x.petsc_vec.set(0.0)

# time_current = 0.0
# time_total = 1.0
# num_steps = 100
# dt = time_total/num_steps

# hist_time = np.zeros(num_steps+1)
# hist_disp = np.zeros(num_steps+1)
# hist_force = np.zeros(num_steps+1)
# hist_mises = np.zeros(num_steps+1)

# ii = 0

# while (round(time_current + dt, 9) <= time_total):
#     time_current += dt
#     ii += 1

#     disp_bc.value = disp_total*Ramp(time_current,time_total)
    
#     (iter, converged) = solver.solve(u)
#     assert converged

#     # Collect results from MPI ghost processes
#     u.x.scatter_forward()
#     u_out.interpolate(u)
#     sigma_mises.interpolate(sigma_mises_expr)

#     # Collect history output variables
#     hist_time[ii] = time_current
#     hist_disp[ii] = disp_total*Ramp(time_current,time_total)
#     hist_force[ii] = domain.comm.gather(fem.assemble_scalar(surface_stress))[0]
#     hist_mises[ii] = np.mean(sigma_mises.x.array)

#     # Write outputs to file
#     with io.XDMFFile(domain.comm, out_file, "a") as xdmf:
#         xdmf.write_function(u_out, ii)
#         xdmf.write_function(sigma_mises, ii)

#     # update u_old
#     # Update DOFs for next step
#     u_old.x.array[:] = u.x.array

#     print("Time step: ", ii, " | Iterations: ", iter, " | U: ",converged)

#%% ----------------------------- Post-processing -----------------------------
plt.plot(hist_time, -hist_disp)

plt.plot(hist_time,hist_force)

plt.plot(hist_time,hist_mises)
