#                                                                      Modules
#%% ===========================================================================
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
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira Morais Pinto (rbarreira@ethz.ch, ' \
'rui.pinto@brown.edu)'
__credits__ = ['Rui Barreira Morais Pinto', ]
__status__ = 'development'
# =============================================================================
#
#%% ----------------------------- Material patch ------------------------------

filename = f"/Users/rbarreira/Desktop/machine_learning/material_patches/" + \
           f"2025_06_05/" + \
           f"material_patches_generation_2d_quad4/material_patch_0/" + \
           f"material_patch/material_patch_attributes.pkl"

filename = f"/Users/rbarreira/Desktop/machine_learning/material_patches/" + \
           f"2025_06_05/" + \
           f"material_patches_generation_3d_hex8_mesh_3x3/" + \
           f"material_patch_0/material_patch/material_patch_attributes.pkl"

with open(filename, 'rb') as file:
    patch = pkl.load(file)

for node_label, displacements in patch['mesh_boundary_nodes_disps'].items():
    # print(f"  Node {node_label}, displacement = {displacements}")

    node_coords = patch['mesh_nodes_coords_ref'][node_label]
    # print(f'Node coords: {node_coords}')
#%% -------------------------------- Geometry ---------------------------------

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
vector_element = element(family="Lagrange", cell=domain.basix_cell(), degree=1,
                         shape=(gdim,))
V = fem.functionspace(domain, vector_element)

# tensor function space 
tensor_element = element(family="Lagrange", cell=domain.basix_cell(), degree=1,
                          shape=(gdim,gdim)) 
V1 = fem.functionspace(domain, tensor_element)

u = fem.Function(V)
u.name = "displacement"

u_trial = TrialFunction(V)
v_test = TestFunction(V)

#%%  -------------------------------- Material --------------------------------
E = fem.Constant(domain, 1.10e5)
nu = fem.Constant(domain, 0.33)

lmbda =  E * nu / ((1. + nu) * (1. - 2. * nu))
mu =  E / (2. * (1. + nu))

#%% ---------------------------- Constitutive law -----------------------------

# strain and stress
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

#%% ------------------------------- Weak forms --------------------------------
# Linear functional: body forces (assumed zero)
f_body = fem.Constant(domain, np.zeros(gdim))
l_form = ufl.dot(f_body, v_test) * dx 

# Bilinear functional
# Residual F(u; v) = inner(sigma(u), epsilon(v))*dx - L(v)
# 'u' is the fem.Function representing the current solution candidate
a_form = ufl.inner(sigma(u), epsilon(v_test)) * dx 


# Derivative of the residual with respect to u, in the direction of u_trial
# J_form = ufl.inner(sigma(u_trial), epsilon(v_test)) * dx
j_gateaux_der = ufl.derivative(a_form, u, u_trial)

residual = a_form - l_form

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
    return fem.dirichletbc(displacement_values, dofs, v_space)

#%% ----------------------- Internal Forces Computation -----------------------
def compute_reaction_forces(domain, u, V, bcs):
    """
    Compute reaction forces from the assembled system.
    This method assembles the stiffness matrix and computes R = K*u - f
    """
    # Alternative 1:
    # Create test and trial functions
    u_test = TestFunction(V)
    u_trial = TrialFunction(V)
    
    # Bilinear form (stiffness matrix)
    a = ufl.inner(sigma(u_trial), epsilon(u_test)) * dx
    
    # Linear form (load vector - assuming zero body forces)
    f_body = fem.Constant(domain, np.zeros(gdim))
    L = ufl.dot(f_body, u_test) * dx
    
    # Assemble system
    A = fem.petsc.assemble_matrix(fem.form(a))#, bcs=bcs)
    A.assemble()
    
    # print(f'Stiffness matrix A: {A.view()}')

    b = fem.petsc.assemble_vector(fem.form(L))
    # fem.petsc.apply_lifting(b, [fem.form(a)], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    # fem.petsc.set_bc(b, bcs)
    print(f'b: {b.getArray()}')
    
    # Compute internal forces: f_int = K*u
    f_int_vec = A.createVecLeft()
    A.mult(u.x.petsc_vec, f_int_vec)
    # print(f'f_int_vec_i = A_ij u_j: {f_int_vec.getArray()}')
    f_int_vec.axpy(-1.0, b)
    # print(f'f_int_vec_i = A_ij u_j - b_i: {f_int_vec.getArray()}')
    # print(f'f_int_vec: {f_int_vec.getArray()}')
    print(f'displacement vector: {u.x.array}')

    # Alternative 2: from the definition of residual    
    # unconstrained_residual_form = fem.form(residual)
    # f_int_vec = fem.petsc.create_vector(unconstrained_residual_form)
    # fem.petsc.assemble_vector(f_int_vec, unconstrained_residual_form)
    # print(f'f_int_vec: {f_int_vec.getArray()}')

    return f_int_vec

def extract_nodal_reaction_forces(reaction_vec, boundary_node_coords, V):
    """
    Extract reaction forces at specific boundary nodes from the global
      reaction vector.
    """
    reaction_forces = {}
    
    # Get reaction vector as numpy array
    reaction_array = reaction_vec.getArray()

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
            if dof <= len(reaction_array):
                if gdim == 3:
                    # x component
                    node_reactions.append(reaction_array[3*dof])
                    # y component
                    node_reactions.append(reaction_array[3*dof+1] )
                    # z component
                    node_reactions.append(reaction_array[3*dof+2] )
                elif gdim == 2:
                    # x component
                    node_reactions.append(reaction_array[2*dof])
                    # y component
                    node_reactions.append(reaction_array[2*dof+1] )

        
        # For 2D, we expect 2 components (x and y)
        # if len(node_reactions) >= 2:
        if gdim == 3:
            reaction_forces[dofs[0]] = np.array(
                [node_reactions[0], node_reactions[1], node_reactions[2]])
        elif gdim == 2:
            reaction_forces[dofs[0]] = np.array(
                [node_reactions[0], node_reactions[1]])
        # elif len(node_reactions) == 1:
        #     reaction_forces[node_label] = np.array([node_reactions[0], 0.0])
        # else:
        #     reaction_forces[node_label] = np.array([0.0, 0.0])
    
    return reaction_forces

#%% --------------------------------- Solver ----------------------------------

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

    # -------------------------- Reaction forces --------------------------
    print('Computing reaction forces...')

    # Compute the reaction forces
    reaction_vec = compute_reaction_forces(domain, u, V, bcs)
    nodal_reactions = extract_nodal_reaction_forces(reaction_vec, 
                                                    boundary_node_coords, V)
    # Store reaction forces for this increment
    forces_internal[idx_inc] = nodal_reactions
    
    # Print reaction forces
    # total_reaction = np.array([0.0, 0.0])
    for node_label, reaction in nodal_reactions.items():
        if gdim == 3:
            print(f'    Node {node_label}: Rx = {reaction[0]:.3e}, ' + \
                    f'Ry = {reaction[1]:.3e}, Rz = {reaction[2]:.3e}, ')
        elif gdim == 2:
            print(f'    Node {node_label}: Rx = {reaction[0]:.3e}, ' + \
                    f'Ry = {reaction[1]:.3e}')
        # total_reaction += reaction
    
    # print(f'    Total reaction force: Rx = {total_reaction[0]:.3e}, ' + \
    #       f'Ry = {total_reaction[1]:.3e}')


    # --------------------------- Save solution ---------------------------
        with io.XDMFFile(
            domain.comm,
            f"{output_dir}/displacement_increment_{idx_inc+1:03d}.xdmf", "w"
            ) as xdmf:

            xdmf.write_mesh(domain)
            xdmf.write_function(u)

    # Clean up PETSc vector
    reaction_vec.destroy()

print('------------- incremental displacement loading complete --------------')



#%% ------------------------------- Output file -------------------------------
out_file = f"{output_dir}/linear_elasticity.xdmf"

u_out = fem.Function(V, name='u')
u_out.interpolate(u)


with io.XDMFFile(domain.comm, out_file, "w") as xdmf:
    xdmf.write_mesh(domain)
with io.XDMFFile(domain.comm, out_file, "a") as xdmf:
    xdmf.write_function(u_out, 0)
