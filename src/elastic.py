"""

Elasticity tutorials:

- https://bleyerj.github.io/comet-fenicsx/tours/linear_problems/isotropic_orthotropic_elasticity/isotropic_orthotropic_elasticity.html
- https://bleyerj.github.io/comet-fenicsx/intro/linear_elasticity/linear_elasticity.html
- https://jsdokken.com/dolfinx-tutorial/chapter2/linearelasticity_code.html
- https://newfrac.gitlab.io/newfrac-fenicsx-training/01-linear-elasticity/LinearElasticity.html
- https://newfrac.github.io/fenicsx-fracture/notebooks/linear-elasticity/01-LinearElasticity.html


Dirichlet BCs:

- https://jsdokken.com/dolfinx-tutorial/chapter3/multiple_dirichlet.html

Compatbility issues between v0.8.0 and v0.9.0
- https://fenicsproject.discourse.group/t/how-did-the-function-object-change-in-dolfinx-v0-9-0/16085
- https://github.com/FEniCS/web/pull/192/files

https://docs.fenicsproject.org/ufl/main/manual/form_language.html

Second-order elements:
- https://bleyerj.github.io/comet-fenicsx/tours/dynamics/elastodynamics_newmark/elastodynamics_newmark.html
(main issue: XDMF support only Lagrangian elements of first and second order)

Node ordering:
- https://jsdokken.com/FEniCS-workshop/src/mesh_generation.html#higher-order-meshes
- GMSH: https://www.manpagez.com/info/gmsh/gmsh-2.2.6/gmsh_65.php


"""

#                                                                      Modules
#%% ===========================================================================
import os

# Import FEnicSx/dolfinx
import dolfinx
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

# specific functions from ufl modules
import ufl

from ufl import (
    sqrt, dev, le, conditional, inv, dot, nabla_div, inner,
    as_matrix, dot, cos, sin, SpatialCoordinate, Identity, grad, ln, tr, det,
    variable, derivative, TestFunction, TrialFunction,
)

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


from fenicsx_plotly import plot

from utilities.nodal_quantities import apply_displacement_bc, \
    extract_nodal_forces, find_boundary_nodes, extract_boundary_displacements

from utilities.force_integration import compute_forces_residual, \
    compute_forces_stress_integration, validate_stress_integration_methods
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira Morais Pinto (rbarreira@ethz.ch, ' \
'rui.pinto@brown.edu)'
__credits__ = ['Rui Barreira Morais Pinto', ]
__status__ = 'development'
# =============================================================================
#
#%% ------------------------------ User inputs  -------------------------------
analysis = "2d"
element_type = 'quad4'
material_behavior = 'elastic'

num_increments = 1

#%% ----------------------------- Material patch ------------------------------
if analysis == "2d":
    if element_type == 'quad4':
        filename = f"/Users/rbarreira/Desktop/machine_learning/" + \
                   f"material_patches/2025_06_05/" + \
                   f"material_patches_generation_2d_quad4_mesh_3x3/" + \
                   f"material_patch_0/material_patch/" + \
                   f"material_patch_attributes.pkl"
        elem_order = 1
    elif element_type == 'quad8':
        filename = f"/Users/rbarreira/Desktop/machine_learning/" + \
                   f"material_patches/2025_06_05/" + \
                   f"material_patches_generation_2d_quad8_mesh_3x3/" + \
                   f"material_patch_0/material_patch/" + \
                   f"material_patch_attributes.pkl"
        elem_order = 2
elif analysis == "3d":
    if element_type == 'hex8':
        filename = f"/Users/rbarreira/Desktop/machine_learning/" + \
                   f"material_patches/2025_06_05/" + \
                   f"material_patches_generation_3d_hex8_mesh_3x3/" + \
                   f"material_patch_0/material_patch/" + \
                   f"material_patch_attributes.pkl"
        elem_order = 1
    elif element_type == 'hex20':
        filename = f"/Users/rbarreira/Desktop/machine_learning/" + \
                   f"material_patches/2025_06_05/" + \
                   f"material_patches_generation_3d_hex20_mesh_3x3/" + \
                   f"material_patch_0/material_patch/" + \
                   f"material_patch_attributes.pkl"
        elem_order = 2

        # MAYBE CHECK:
        # topological
        # https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.fem.html#dolfinx.fem.locate_dofs_topological

with open(filename, 'rb') as file:
    patch = pkl.load(file)

#%% ----------------------------- Geometry & mesh -----------------------------
if analysis == "2d":
    domain = mesh.create_unit_square(comm=MPI.COMM_WORLD, nx=3, ny=3, 
                               cell_type=mesh.CellType.quadrilateral)
    domain.topology.create_connectivity(1, 2) 
elif analysis == "3d":
    domain = mesh.create_unit_cube(comm=MPI.COMM_WORLD, nx=3, ny=3, nz=3, 
                               cell_type=mesh.CellType.hexahedron)
    domain.topology.create_connectivity(2, 3) 

mesh_degree = domain.geometry.cmap.degree
# assert mesh_degree == elem_order, \
#     f'Mesh degree: {mesh_degree} does not match element order: {elem_order}'

# 3D 
# Topological dimension (3 for a cube)
tdim = domain.topology.dim 
# Geometrical dimension (3 for 3D space)
gdim = domain.geometry.dim

# Define the volume integration measure "dx" 
# and the no. of volume quadrature points.
deg_quad = max(2, 2 * elem_order)
dx = ufl.Measure('dx', domain=domain,
                 metadata={'quadrature_degree': deg_quad,
                           "quadrature_scheme": "default"})

# Define the boundary integration measure "ds"
ds = ufl.Measure('ds', domain=domain)
        # metadata={'quadrature_degree':deg_quad,
        # "quadrature_scheme": "default"})
#%% ----------------------------- Function spaces -----------------------------
# vector elements and function space
vector_element = element(family="Lagrange", cell=domain.basix_cell(),
                         degree=elem_order, shape=(gdim,))
V = fem.functionspace(domain, vector_element)

# tensor function space 
tensor_element = element(family="Lagrange", cell=domain.basix_cell(), 
                         degree=elem_order, shape=(gdim,gdim)) 
V1 = fem.functionspace(domain, tensor_element)

u = fem.Function(V)
u.name = "displacement"

u_trial = TrialFunction(V)
v_test = TestFunction(V)

#%% ---------------------------- Constitutive law -----------------------------
if material_behavior == 'elastic':

    # ------------------------- Material properties ---------------------------
    E = fem.Constant(domain, 1.10e5)
    nu = fem.Constant(domain, 0.33)

    lmbda =  E * nu / ((1. + nu) * (1. - 2. * nu))
    mu =  E / (2. * (1. + nu))


    def epsilon(u_):
        """
        Strain tensor:
        
        Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
        """
        return ufl.sym(ufl.grad(u_)) 

    def sigma(u_):
        """
        Stress tensor:
        
        Isotropic material in plane strain
        """
        # return lmbda * ufl.nabla_div(u_) * ufl.Identity(gdim) + \
        #    2 * mu * epsilon(u_)

        return ufl.tr(epsilon(u_)) * lmbda * ufl.Identity(gdim) + \
        2 * mu * epsilon(u_)

    a_form_lhs = sigma(u)
    a_form_rhs = epsilon(v_test)

elif material_behavior == 'hyperelastic':
    # ------------------------- Material properties ---------------------------
    E = fem.Constant(domain, 1.10e5)
    nu = fem.Constant(domain, 0.33)

    lmbda =  E * nu / ((1. + nu) * (1. - 2. * nu))
    mu =  E / (2. * (1. + nu))

    # Identity tensor
    Id = Identity(gdim)

    # Deformation gradient
    F = variable(Id + grad(u))

    # Right Cauchy-Green tensor
    C = F.T * F

    # Invariants of deformation tensors
    I1 = tr(C)
    J = det(F)

    # Stored strain energy density (compressible neo-Hookean model)
    psi = mu / 2 * (I1 - 3 - 2 * ln(J)) + lmbda / 2 * (J - 1) ** 2

    # PK1 stress = d_psi/d_F
    pk_1 = ufl.diff(psi, F)
    # in linear elasticity: 
    # pk_1 = 2.0 * mu * ufl.sym(ufl.grad(u)) + \
    #     lmbda * ufl.tr(ufl.sym(ufl.grad(u))) * I

    E_pot = psi * dx

    a_form_lhs = pk_1
    a_form_rhs = grad(v_test)
#%% ------------------------------- Weak forms --------------------------------
# Linear functional: body forces (assumed zero)
f_body = fem.Constant(domain, np.zeros(gdim))
l_form = ufl.dot(f_body, v_test) * dx 

# Bilinear functional
# Residual F(u; v) = inner(sigma(u), epsilon(v))*dx - L(v)
# 'u' is the fem.Function representing the current solution candidate
a_form = ufl.inner(a_form_lhs, a_form_rhs) * dx 

# Residual
residual = a_form - l_form

# Derivative of the residual with respect to u, in the direction of u_trial
# J_form = ufl.inner(sigma(u_trial), epsilon(v_test)) * dx
j_gateaux_der = derivative(residual, u, u_trial)

#%% --------------------------- Boundary Conditions ---------------------------



num_bd_nodes_mat_patch = len(patch['mesh_boundary_nodes_disps'].items())

# List to hold the DirichletBC objects - it will be updated in the loop
bcs = []

# Boundary node coordinates read from FEniCSx
boundary_nodes = find_boundary_nodes(domain, V)
num_bd_nodes_fe_space = len(boundary_nodes)
assert num_bd_nodes_mat_patch == num_bd_nodes_fe_space, f'Different no. ' + \
    f'of boundary nodes in material patch ({num_bd_nodes_mat_patch}) and ' + \
        f'solver ({num_bd_nodes_fe_space})'

node_coords_fe_space = V.tabulate_dof_coordinates()
bd_node_coords_fe_space = {}
for node in boundary_nodes:
    if gdim == 2:
        bd_node_coords_fe_space[node] = [node_coords_fe_space[node, 0],
                                         node_coords_fe_space[node, 1]]
        print(f'Node {node}: ({node_coords_fe_space[node, 0]:.6f}, ' + \
              f'{node_coords_fe_space[node, 1]:.6f})')
    elif gdim == 3:
        bd_node_coords_fe_space[node] = [node_coords_fe_space[node, 0],
                                         node_coords_fe_space[node, 1],
                                         node_coords_fe_space[node, 2]]
        print(f'Node {node}: ({node_coords_fe_space[node, 0]:.6f}, ' + \
              f'{node_coords_fe_space[node, 1]:.6f}, '
              f'{node_coords_fe_space[node, 2]:.6f})')

# Boundary node coordinates read from material patch
bd_node_coords_matpatch = {}
for node_label in patch['mesh_boundary_nodes_disps'].keys():
    bd_node_coords_matpatch[node_label] = patch['mesh_nodes_coords_ref'][
        node_label]

# # TODO: CHECK THIS - geometrical vs topological
# boundary_facets = mesh.exterior_facet_indices(domain.topology)

# # Find boundary DOFs
# boundary_dofs = fem.locate_dofs_topological(V, entity_dim=tdim-1,
#                                             entities=boundary_facets)

# print(boundary_dofs)












#%% ----------------------- Internal Forces Computation -----------------------
# Dictionary to store internal forces for all time steps: 
forces_internal = {}
u_disp = {}

#%% ----------------------- Data Structure Initialization ---------------------
# Initialize the main data structure to save
simulation_data = {
    'boundary_nodes_coords': {},
    'boundary_nodes_disps_time_series': {},
    'boundary_nodes_forces_time_series': {}
}

# Initialize boundary node coordinates from fe_space coordinates
for node_idx in boundary_nodes:
    # Extract coordinates for this boundary node
    if gdim == 2:
        node_coords = np.array([node_coords_fe_space[node_idx, 0], 
                      node_coords_fe_space[node_idx, 1]])
    elif gdim == 3:
        node_coords = np.array([node_coords_fe_space[node_idx, 0], 
                      node_coords_fe_space[node_idx, 1],
                      node_coords_fe_space[node_idx, 2]])
    
    # Store coordinates in simulation data
    simulation_data['boundary_nodes_coords'][int(node_idx)] = node_coords
    
    # Initialize time series arrays for displacements and forces
    simulation_data['boundary_nodes_disps_time_series'][int(node_idx)] = []
    simulation_data['boundary_nodes_forces_time_series'][int(node_idx)] = []


#%% --------------------------------- Solver ----------------------------------
problem = NonlinearProblem(F=residual, u=u, bcs=bcs,
                           J=j_gateaux_der)

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

# Ensuring u is set to 0.
u.x.petsc_vec.set(0.0)

# Create output directory
output_dir = "output_incremental_disp"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ----------------------------- Timestepping Loop -----------------------------
print("\n----------------- Incremental displacement loading -----------------")
for idx_inc in range(num_increments):
    print(f"\n# Increment {idx_inc + 1}/{num_increments}:")

    # Clear any previous boundary conditions
    bcs.clear()

    # Apply boundary conditions based on patch data for this increment
    current_increment_disps = {}

    # Apply boundary conditions based on patch data for this increment
    # Read node labels as keys from dictionary 
    # patch['mesh_boundary_nodes_disps']
    for node_label, displacements in \
        patch['mesh_boundary_nodes_disps'].items():
        
        # Retrieve nodal coordinates from dictionary
        # patch['mesh_nodes_coords_ref']
        node_coords = patch['mesh_nodes_coords_ref'][node_label]  
        # print(f'    Node label: {node_label}, coords:({node_coords})')

        # Find node labels in dofs in the FEniCSx FE mesh by the nodal 
        # coordinates from the dictionary patch._mesh_nodes_coords_ref
        disp_array = np.array(displacements, dtype=np.float64)

        displacement_values = disp_array
        # Time series data
        # if idx_inc < disp_array.shape[0]:
        #     displacement_values = disp_array[idx_inc, :]

        # Apply boundary condition
        bcs.append(apply_displacement_bc(V, gdim, node_coords, 
                                         displacement_values))
        
    # Update the problem with the new boundary conditions
    problem.bcs = bcs
    print(bcs)

    # Solve the nonlinear problem
    try:
        num_iterations, converged = solver.solve(u)
        if not converged:
            print(f'Solver did not converge at increment {idx_inc + 1}. ' + \
                    f'Stopping.')
            break

        print(f'    Converged in {num_iterations} iterations.')

        # Updates ghost values for parallel computations
        u.x.scatter_forward() 

        # Store current displacement values
        boundary_displacements = extract_boundary_displacements(
            u, boundary_nodes, V, gdim)
        
        for node_label, disp in boundary_displacements.items():
            simulation_data['boundary_nodes_disps_time_series'][
                node_label].append(disp.tolist())


    except Exception as exc:
        print(f'   Error at increment {idx_inc+1}: {exc}.')
        break
    
    
    # ---------------------------- Internal forces ----------------------------
    print('Computing internal forces...')

    # ABAQUS interior displacements
    # u.x.array[3*2] = 0.0286878
    # u.x.array[3*2 + 1] = 0.0428083

    # u.x.array[5*2] = 0.0201981        
    # u.x.array[5*2 + 1] = 0.0237266

    # u.x.array[7*2] = 0.0287869
    # u.x.array[7*2 + 1] = 0.0371176

    # u.x.array[10*2] = 0.0151731         
    # u.x.array[10*2 + 1] = 0.024637

    # Compute the internal forces
    force_vec = compute_forces_residual(
        domain, u=u, dx=dx, V=V,
        material_behavior=material_behavior,
        strain_func=epsilon, stress_func=sigma)
    
    force_vec = compute_forces_stress_integration(
        domain, u=u, V=V, material_behavior=material_behavior, 
        strain_func=epsilon, stress_func=sigma)
    
    # Validate force values from different methods
    # is_valid = validate_stress_integration_methods(
    #     domain, u=u, V=V, dx=dx, material_behavior=material_behavior,
    # strain_func=epsilon, stress_func=sigma)
    # if is_valid:
    #     print(f"Stress integration validation: PASSED")
    # else:
    #     print(f"Stress integration validation: FAILED")

    # Extract internal forces at the boundary nodes
    nodal_forces = extract_nodal_forces(force_vec, bd_node_coords_fe_space,
                                         gdim, V)

    # Store force data  
    for node_label, force in nodal_forces.items():                    
        simulation_data['boundary_nodes_forces_time_series'][
            node_label].append(force.tolist())

    # --------------------------- Save solution ---------------------------
        # with io.XDMFFile(
        #     domain.comm,
        #     f"{output_dir}/displacement_increment_{idx_inc+1:03d}.xdmf", "w"
        #     ) as xdmf:

        #     xdmf.write_mesh(domain)
        #     xdmf.write_function(u)

    # Clean up PETSc vector
    force_vec.destroy()

print('------------- incremental displacement loading complete --------------')

#%% ------------------------------- Output file -------------------------------
# out_file = f"{output_dir}/linear_elasticity.xdmf"

# u_out = fem.Function(V, name='u')
# u_out.interpolate(u)

# with io.XDMFFile(domain.comm, out_file, "w") as xdmf:
#     xdmf.write_mesh(domain)
# with io.XDMFFile(domain.comm, out_file, "a") as xdmf:
#     xdmf.write_function(u_out, t=0)

#%% ------------------------------- Save Data -------------------------------

# Convert forces time series lists to NumPy arrays
for node_label, forces_list in simulation_data[
    'boundary_nodes_forces_time_series'].items():
    simulation_data['boundary_nodes_forces_time_series'][
        node_label] = np.array(forces_list)

# Convert displacements time series lists to NumPy arrays
for node_label, disps_list in simulation_data[
    'boundary_nodes_disps_time_series'].items():
    simulation_data['boundary_nodes_disps_time_series'][
        node_label] = np.array(disps_list)


# Save the complete simulation data to a pickle file
output_filename = f"material_patch_sim_data_{analysis}_{element_type}.pkl"

try:
    with open(output_filename, 'wb') as f:
        pkl.dump(simulation_data, f, protocol=pkl.HIGHEST_PROTOCOL)

except Exception as excp:
    print(f"Error saving simulation data: {excp}")
