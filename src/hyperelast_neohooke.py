""" 

References & tutorials:

Hyperelasticity:
- https://bleyerj.github.io/comet-fenicsx/intro/hyperelasticity/hyperelasticity.html
- https://jsdokken.com/dolfinx-tutorial/chapter2/hyperelasticity.html
- https://newfrac.gitlab.io/newfrac-fenicsx-training/02-finite-elasticity/finite-elasticity-I.html

Compatbility issues between v0.8.0 and v0.9.0
- https://fenicsproject.discourse.group/t/how-did-the-function-object-change-in-dolfinx-v0-9-0/16085
- https://github.com/FEniCS/web/pull/192/files
"""

#                                                                      Modules
#%% ===========================================================================
import os

# Import FEnicSx/dolfinx
import dolfinx

# For numerical arrays
import numpy as np

# specific functions from ufl modules
import ufl

from ufl import (
    sqrt, dev, le, conditional, inv, dot, nabla_div, inner,
    as_matrix,
    dot,
    cos,
    sin,
    SpatialCoordinate,
    Identity,
    grad,
    ln,
    tr,
    det,
    variable,
    derivative,
    TestFunction,
    TrialFunction,
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

# Matplotlib for plotting
import matplotlib.pyplot as plt

import pickle as pkl

from fenicsx_plotly import plot
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira Morais Pinto (rbarreira@ethz.ch, ' \
'rui.pinto@brown.edu)'
__credits__ = ['Rui Barreira Morais Pinto', ]
__status__ = 'development'
# =============================================================================
#

#%% ------------------------------ User inputs  -------------------------------
analysis = "3d"
element_type = 'hex8'

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

with open(filename, 'rb') as file:
    patch = pkl.load(file)

for node_label, displacements in patch['mesh_boundary_nodes_disps'].items():
    # print(f"  Node {node_label}, displacement = {displacements}")

    node_coords = patch['mesh_nodes_coords_ref'][node_label]
    # print(f'Node coords: {node_coords}')
#%% -------------------------------- Geometry ---------------------------------
if analysis == "2d":
    domain = mesh.create_unit_square(comm=MPI.COMM_WORLD, nx=3, ny=3, 
                               cell_type=mesh.CellType.quadrilateral)
elif analysis == "3d":
    domain = mesh.create_unit_cube(comm=MPI.COMM_WORLD, nx=3, ny=3, nz=3, 
                               cell_type=mesh.CellType.hexahedron)

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

#%%  -------------------------------- Material --------------------------------

# Elastic constants
E = fem.Constant(domain, 1.10e5)
nu = fem.Constant(domain, 0.33)

lmbda =  E * nu / ((1. + nu) * (1. - 2. * nu))
mu =  E / (2. * (1. + nu))

#%% ---------------------------- Constitutive law -----------------------------
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

#%% ------------------------------- Weak forms --------------------------------
# Linear functional: body forces (assumed zero)
f_body = fem.Constant(domain, np.zeros(gdim))
l_form = ufl.dot(f_body, v_test) * dx 

# Bilinear functional
# 'u' is the fem.Function representing the current solution candidate
a_form = ufl.inner(pk_1, grad(v_test)) * dx 

# Residual
residual = a_form - l_form
# or equivalently
residual = derivative(E_pot, u, v_test)  

# Derivative of the residual with respect to u, in the direction of u_trial
j_gateaux_der = derivative(residual, u, u_trial)

#%% --------------------------- Initial conditions ----------------------------

def identity(x):
    """

    A function for constructing the identity matrix.
    To use the interpolate() feature, this must be defined as a function of x.
    
    In 3D:
        values: dim 1x9 [1, 0, 0, 0, 1, 0, 0, 0, 1]
    In 2D:
        values: dim 1x4 [1, 0, 0, 1]
    """
    values = np.zeros((domain.geometry.dim*domain.geometry.dim,
                      x.shape[1]), dtype=np.float64)
    if gdim == 3:
        values[0] = 1
        values[4] = 1
        values[8] = 1
    elif gdim == 2:
        values[0] = 1
        values[3] = 1

    return values


#%% --------------------------- Boundary Conditions ---------------------------

# List to hold the DirichletBC objects - it will be updated in the loop
bcs = []

def find_dofs(coords):
    """
    Finds the global DOFs in function space V_space whose coordinates match
    those in coords_array.
    coords_array should be a single (x,y,z) coordinate array.
    Returns a list of global DOF indices.
    """
    # FEniCSx 0.9.0 changed the API for locating DOFs!
    # dof_coords = np.array(v_space.tabulate_dof_coordinates(),
    # dtype=np.float64)
    
    # for idx_node, node_coord in enumerate(dof_coords):
    #     print(f'FENICSX:     Node {idx_node}: coords({node_coord})')
    #     # print(f'coords {idx_node}: coords({coords})')

    if gdim == 3:
        dofs = fem.locate_dofs_geometrical(V, lambda x: np.logical_and(
            np.logical_and(np.isclose(x[0], coords[0]), 
                           np.isclose(x[1], coords[1])), 
            np.isclose(x[2], coords[2])))
    elif gdim == 2:
        dofs = fem.locate_dofs_geometrical(V, lambda x: np.logical_and(
            np.isclose(x[0], coords[0]), np.isclose(x[1], coords[1])) )

    # print(f'FEniCSX mesh DOF: {dofs}, coords: ({dof_coords[dofs]})')

    return dofs

def apply_displacement_bc(v_space, coords, displacement_values):
    """
    Apply displacement boundary condition at given coordinates.
    """
    # Find DOFs at the specified coordinates
    dofs = find_dofs(coords)
    
    if len(dofs) == 0:
        return None

    print(f'    dofs: {dofs}, displacement_values: {displacement_values}')

    # https://jsdokken.com/dolfinx-tutorial/chapter2/linearelasticity_code.html
    return fem.dirichletbc(2*displacement_values, dofs, v_space)

#%% ----------------------- Internal Forces Computation -----------------------
def compute_forces_by_residual(domain, u, V, bcs):
    """
    Compute reaction forces from the assembled system.
    This method assembles the stiffness matrix and computes R = K*u - f
    """
    # Alternative 1:
    # Create test and trial functions
    u_test = TestFunction(V)
    
    # Bilinear form (stiffness matrix)
    a_bilinear_form = inner(pk_1, grad(v_test)) * dx 
    
    # Linear form (load vector - assuming zero body forces)
    f_body = fem.Constant(domain, np.zeros(gdim))
    l_linear_form = dot(f_body, u_test) * dx
    
    residual = a_bilinear_form - l_linear_form

    # Assemble the internal force vector
    f_int_vec = fem.petsc.create_vector(fem.form(residual))
    fem.petsc.assemble_vector(f_int_vec, fem.form(residual))
    
    # Update ghost values for parallel computations
    f_int_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, 
                          mode=PETSc.ScatterMode.REVERSE)
    
    # # Alternative 2: from the definition of residual    
    # residual_form = fem.form(residual)
    # f_int_vec = fem.petsc.create_vector(fem.form(residual_form))
    # # Assemble the residual vector
    # fem.petsc.assemble_vector(f_int_vec, fem.form(residual_form))
    # f_int_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, 
    #                       mode=PETSc.ScatterMode.REVERSE)
    # # print(f'f_int_vec: {f_int_vec.getArray()}')

    return f_int_vec

def compute_forces_by_stress_integration_ref_config(domain, u, V, gdim):
    """
    In the reference configuration, internal forces are computed as:

    f_int = ∫ P : ∇v dV
    
    where P is the first Piola-Kirchhoff stress tensor,
    and v is the test function.
    """
    # Test function
    v_test = TestFunction(V)
    
    # Define the internal force as the residual of the weak form
    # (without external forces since we assume f_body = 0)
    f_body = fem.Constant(domain, np.zeros(gdim))
    
    # Internal forces: f_int = ∫ P : ∇v dV
    internal_force_form = ufl.inner(pk_1, grad(v_test)) * dx
    
    # Assemble the internal force vector
    f_int_vec = fem.petsc.create_vector(fem.form(internal_force_form))
    fem.petsc.assemble_vector(f_int_vec, fem.form(internal_force_form))
    
    # Update ghost values for parallel computations
    f_int_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, 
                         mode=PETSc.ScatterMode.REVERSE)
    
    return f_int_vec

def compute_forces_by_stress_integration_current_config(domain, u, V, gdim):
    """
    Compute internal forces using Cauchy stress in current configuration.
    
    For this approach, we need to transform the integral to the 
    current configuration:

    f_int = ∫ σ : ∇v dv = ∫ J σ F^(-T) : ∇v dV ,
    
    where σ is the Cauchy stress, v is the current volume,
    and V is the reference volume.
    """

    # Deformation gradient
    F_def = ufl.variable(Id + ufl.grad(u))

    # Invariants
    J_det = ufl.det(F_def)

    # Cauchy stress: σ = (1/J) P F^T
    sigma_cauchy = (1/J_det) * pk_1 * F_def.T
    
    # Create test function
    v_test = ufl.TestFunction(V)
    
    # Transform to reference configuration: ∫ J σ F^(-T) : ∇v dV
    # This is equivalent to ∫ P : ∇v dV (which is what we use above)
    F_inv_T = ufl.inv(F_def).T
    internal_force_form = ufl.inner(J_det * sigma_cauchy * F_inv_T, 
                                    ufl.grad(v_test)) * ufl.dx
    
    # Assemble the internal force vector
    f_int_vec = fem.petsc.create_vector(fem.form(internal_force_form))
    fem.petsc.assemble_vector(f_int_vec, fem.form(internal_force_form))
    
    # Update ghost values for parallel computations
    f_int_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, 
                         mode=PETSc.ScatterMode.REVERSE)
    
    return f_int_vec

def compute_stress_measures_at_points(domain, u, V, gdim, points=None):
    """
    Compute various stress measures at specific points in the domain.
    """
    # Identity tensor
    Id = ufl.Identity(gdim)
    
    # Deformation gradient
    F_def = Id + ufl.grad(u)
    
    # Right Cauchy-Green tensor
    C = F_def.T * F_def
    
    # Invariants
    J_det = ufl.det(F_def)
    
    # Cauchy stress
    sigma_cauchy = (1/J_det) * pk_1 * F_def.T  
    
    # Tensor function spaces for stress projection
    tensor_element = ufl.TensorElement("Lagrange", domain.ufl_cell(),
                                       1, shape=(gdim,gdim))
    
    V_tensor = fem.FunctionSpace(domain, tensor_element)
    
    # Project stresses to nodes
    pk1_proj = fem.Function(V_tensor, name="PK1_stress")
    sigma_proj = fem.Function(V_tensor, name="Cauchy_stress")
    
    # Projection using Expression class
    pk1_expr = fem.Expression(pk_1, V_tensor.element.interpolation_points())
    sigma_expr = fem.Expression(sigma_cauchy,
                                V_tensor.element.interpolation_points())
    
    pk1_proj.interpolate(pk1_expr)
    sigma_proj.interpolate(sigma_expr)
    
    return pk1_proj, sigma_proj

def validate_stress_integration_methods(domain, u, V, gdim):
    """
    Compare different methods for computing internal forces.
    This helps validate that the stress integration is working correctly.
    """
    print("Validating stress integration methods...")
    
    # Method 1: Using your existing residual approach
    force_vec_residual = compute_forces_by_residual(domain, u, V, [])
    
    # Method 2: Direct stress integration (PK1)
    force_vec_pk1 = compute_forces_by_stress_integration_ref_config(
        domain, u, V, gdim)
    
    # Method 3: Cauchy stress approach
    force_vec_cauchy = compute_forces_by_stress_integration_current_config(
        domain, u, V, gdim)
    
    # Compare results
    residual_array = force_vec_residual.getArray()
    pk1_array = force_vec_pk1.getArray()
    cauchy_array = force_vec_cauchy.getArray()
    
    # Compute differences
    diff_pk1_residual = np.linalg.norm(pk1_array - residual_array)
    diff_cauchy_residual = np.linalg.norm(cauchy_array - residual_array)
    diff_pk1_cauchy = np.linalg.norm(pk1_array - cauchy_array)
    
    print(f"  ||F_PK1 - F_residual|| = {diff_pk1_residual:.3e}")
    print(f"  ||F_Cauchy - F_residual|| = {diff_cauchy_residual:.3e}")
    print(f"  ||F_PK1 - F_Cauchy|| = {diff_pk1_cauchy:.3e}")
    
    # Clean up
    force_vec_residual.destroy()
    force_vec_pk1.destroy()
    force_vec_cauchy.destroy()
    
    return diff_pk1_residual < 1e-10

def extract_nodal_forces(force_vec, boundary_node_coords, V):
    """
    Extract reaction forces at specific boundary nodes from the global
      reaction vector.
    """
    force_output = {}
    
    # Convert to a numpy array
    force_array = force_vec.getArray()

    # print(f'np.shape(reaction_array): {np.shape(reaction_array)}')

    # print(f'reaction_array: {reaction_array}')
    
    for node_label, coords in boundary_node_coords.items():
        # Find DOFs at this node
        dofs = find_dofs(coords)
        # if len(dofs) == 0:
        #     continue
            
        # Extract reaction forces at these DOFs
        node_reactions = []
        for dof in dofs:
            if dof <= len(force_array):
                if gdim == 3:
                    # x component
                    node_reactions.append(force_array[3*dof])
                    # y component
                    node_reactions.append(force_array[3*dof+1] )
                    # z component
                    node_reactions.append(force_array[3*dof+2] )
                elif gdim == 2:
                    # x component
                    node_reactions.append(force_array[2*dof])
                    # y component
                    node_reactions.append(force_array[2*dof+1] )

        
        # For 2D, we expect 2 components (x and y)
        # if len(node_reactions) >= 2:
        if gdim == 3:
            force_output[dofs[0]] = np.array(
                [node_reactions[0], node_reactions[1], node_reactions[2]])
        elif gdim == 2:
            force_output[dofs[0]] = np.array(
                [node_reactions[0], node_reactions[1]])
        # elif len(node_reactions) == 1:
        #     force_output[node_label] = np.array([node_reactions[0], 0.0])
        # else:
        #     force_output[node_label] = np.array([0.0, 0.0])
    
    return force_output

#%% --------------------------------- Solver ----------------------------------

problem = NonlinearProblem(F=residual, u=u, bcs=bcs,
                           J=j_gateaux_der)

solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-4
solver.atol = 1e-4
solver.max_it = 50
solver.report = True

#  The Krylov solver parameters.
# ksp = solver.krylov_solver
# opts = PETSc.Options()
# option_prefix = ksp.getOptionsPrefix()
# opts[f"{option_prefix}ksp_type"] = "preonly" 
# opts[f"{option_prefix}pc_type"] = "lu"
# # opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
# # opts[f"{option_prefix}ksp_max_it"] = 30
# ksp.setFromOptions()

#%%  -------------------------------- Solution --------------------------------

# Incremental loading
num_increments = 1

# Ensuring u is set to 0.
u.x.petsc_vec.set(0.0)
# Dictionary to store reaction forces for all time steps: 
# structure: {increment_idx: {node_label: [Rx, Ry]}}
forces_internal = {}

# Create output directory
output_dir = "output_incremental_disp"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Boundary node coordinates dictionary for reaction force computation
boundary_node_coords = {}
for node_label in patch['mesh_boundary_nodes_disps'].keys():
    boundary_node_coords[node_label] = patch['mesh_nodes_coords_ref'][
        node_label]

print(f'boundary_node_coords: {boundary_node_coords}')
# ----------------------------- Timestepping Loop -----------------------------
print("\n----------------- Incremental displacement loading -----------------")
for idx_inc in range(num_increments):
    print(f"\n# Increment {idx_inc + 1}/{num_increments}:")

    # Clear any previous boundary conditions
    bcs.clear()

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
        #     # Use correct time value
        #     displacement_values = disp_array[idx_inc, :]
        # else:
        #     # Use last value if beyond range
        #     displacement_values = disp_array[-1, :]  

        # Apply boundary condition
        bcs.append(apply_displacement_bc(V, node_coords, displacement_values))

    # Update the problem with the new boundary conditions
    problem.bcs = bcs


    # Solve the nonlinear problem
    try:
        num_iterations, converged = solver.solve(u)
        if not converged:
            print(f'Solver did not converge at increment {idx_inc + 1}. ' + \
                  f'Stopping.')
            break

        print(f'    Converged in {num_iterations} iterations.')

        # Updates ghost values for parallel computations
        # https://bleyerj.github.io/comet-fenicsx/intro/hyperelasticity/hyperelasticity.html
        u.x.scatter_forward() 

    except Exception as exc:
        print(f'   Error at increment {idx_inc+1}: {exc}.')
        break

    # ---------------------------- Internal forces ----------------------------
    print('Computing internal forces...')

    # Compute the reaction forces
    force_vec = compute_forces_by_stress_integration_current_config(
        domain, u, V, gdim)
    # force_vec = compute_forces_by_stress_integration_ref_config(
    #     domain, u, V, gdim)
    # force_vec = compute_forces_by_residual(domain, u, V, bcs)

    is_valid = validate_stress_integration_methods(domain, u, V, gdim)
    print(f"Stress integration validation: {'PASSED' if is_valid else 'FAILED'}")

    nodal_forces = extract_nodal_forces(force_vec, boundary_node_coords, V)
    # Store reaction forces for this increment
    forces_internal[idx_inc] = nodal_forces
    
    # Print reaction forces
    # total_reaction = np.array([0.0, 0.0])
    for node_label, f_int in nodal_forces.items():
        if gdim == 3:
            print(f'    Node {node_label}: Rx = {f_int[0]:.3e}, ' + \
                    f'Ry = {f_int[1]:.3e}, Rz = {f_int[2]:.3e}, ')
        elif gdim == 2:
            print(f'    Node {node_label}: Rx = {f_int[0]:.3e}, ' + \
                    f'Ry = {f_int[1]:.3e}')
        # total_reaction += reaction
    
    # print(f'    Total reaction force: Rx = {total_reaction[0]:.3e}, ' + \
    #       f'Ry = {total_reaction[1]:.3e}')

    # --------------------------- Save solution ---------------------------
        with io.XDMFFile(
            domain.comm,
            f"{output_dir}/hyperelasticit_neohooke_displacement_increment_{idx_inc+1:03d}.xdmf", "w"
            ) as xdmf:

            xdmf.write_mesh(domain)
            xdmf.write_function(u)

    # Clean up PETSc vector
    force_vec.destroy()

print('------------- incremental displacement loading complete --------------')



#%% ------------------------------- Output file -------------------------------
out_file = f"{output_dir}/hyperelasticit_neohooke_.xdmf"

u_out = fem.Function(V, name='u')
u_out.interpolate(u)


with io.XDMFFile(domain.comm, out_file, "w") as xdmf:
    xdmf.write_mesh(domain)
with io.XDMFFile(domain.comm, out_file, "a") as xdmf:
    xdmf.write_function(u_out, 0)
