import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from dolfinx import fem, mesh, io
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
from petsc4py import PETSc

def compute_forces_residual(domain, u, dx, V, 
                            material_behavior,
                            strain_func=None,
                            stress_func=None):
    """
    Compute internal forces from the assembled system.
    
    This method assembles the stiffness matrix and computes R = K*u - f
    """
    # Create test and trial functions
    u_test = TestFunction(V)
    u_trial = TrialFunction(V)

    gdim = domain.geometry.dim

    if material_behavior == 'elastic':
       
        # Bilinear form (stiffness matrix)
        a = ufl.inner(stress_func(u_trial), strain_func(u_test)) * dx
        
        # Linear form (load vector - assuming zero body forces)
        f_body = fem.Constant(domain, np.zeros(gdim))
        L = dot(f_body, u_test) * dx
        
        # Assemble system
        A = fem.petsc.assemble_matrix(fem.form(a)) #, bcs=bcs)
        A.assemble()
        
        # DEBUG: 
        # Convert PETSc matrix to dense NumPy array
        # print(f'Stiffness matrix A: {A.view()}')
        # A_dense = A.convert("dense").getDenseArray()
        # Save to CSV file
        # np.savetxt('stiffness_matrix_fenicsx.csv', A_dense, delimiter=',', 
        # fmt='%.9e')

        b = fem.petsc.assemble_vector(fem.form(L))
        # fem.petsc.apply_lifting(b, [fem.form(a)], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                    mode=PETSc.ScatterMode.REVERSE)
        # fem.petsc.set_bc(b, bcs)
        # print(f'b: {b.getArray()}')
        
        # Compute internal forces: f_int = K*u
        f_int_vec = A.createVecLeft()
        A.mult(u.x.petsc_vec, f_int_vec)
        # print(f'f_int_vec_i = A_ij u_j: {f_int_vec.getArray()}')
        f_int_vec.axpy(-1.0, b)
        # print(f'f_int_vec_i = A_ij u_j - b_i: {f_int_vec.getArray()}')
        # print(f'f_int_vec: {f_int_vec.getArray()}')
        # print(f'displacement vector: {u.x.array}')

        # Alternative 2: from the definition of residual    
        # unconstrained_residual_form = fem.form(residual)
        # f_int_vec = fem.petsc.create_vector(unconstrained_residual_form)
        # fem.petsc.assemble_vector(f_int_vec, unconstrained_residual_form)
        # print(f'f_int_vec: {f_int_vec.getArray()}')

    elif material_behavior == 'hyperelastic':
        
        # Bilinear form (stiffness matrix)
        a_bilinear_form = inner(stress_func, grad(u_test)) * dx 
        
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
        
    return f_int_vec
        


# def compute_forces_stress(domain, u, stress_func, boundary_node_coords, V):
#     """
#     Compute interal forces at boundary nodes using the method of virtual work.
    
#     Parameters:
#     - domain: FEniCS mesh
#     - u: displacement solution function
#     - stress_func: stress function
#     - boundary_node_coords: dictionary of node coordinates
#     - V: function space
    
#     Returns:
#     - Dictionary with node labels and their internal forces
#     """
    
#     # Function space for the stress tensor
#     tensor_element = element(family="Lagrange", cell=domain.basix_cell(), 
#                            degree=1, shape=(gdim, gdim))
#     V_stress = fem.functionspace(domain, tensor_element)
    
#     # Project stress tensor to function space
#     stress_expr = stress_func(u)
#     stress_function = fem.Function(V_stress)
#     stress_function.interpolate(
#         fem.Expression(stress_expr, V_stress.element.interpolation_points()))
    
#     forces_internal = {}
    
#     for node_label, coords in boundary_node_coords.items():
#         # Find DOFs at this node
#         dofs = find_dofs(V, coords)
        
#         if len(dofs) == 0:
#             continue
            
#         # Test function that is 1 at this node and 0 elsewhere
#         # Create a function with delta function support
#         delta_func = fem.Function(V)
#         delta_func.x.array[:] = 0.0
        
#         # Set unit values at the DOFs corresponding to this node
#         for dof in dofs:
#             if dof < len(delta_func.x.array):
#                 delta_func.x.array[dof] = 1.0
        
#         # Compute internal force using virtual work principle
#         # R = ∫ σ : ε(δu) dΩ where δu is the delta function at the node
#         virtual_strain = epsilon(delta_func)
#         internal_form = ufl.inner(stress_expr, virtual_strain) * dx
        
#         # Assemble the form to get the internal force
#         force = fem.assemble_scalar(fem.form(internal_form))
    

#         forces_internal[node_label] = force
        
#     return forces_internal



def compute_forces_stress_integration(domain, u, V, stress_func, strain_func,
                                      material_behavior,
                                      configuration='reference'):
    """
    Compute internal forces by integrating stresses over elements.
    
    This method uses the principle: f_int = ∫ B^T σ dV
    where B is the strain-displacement matrix and σ is the stress tensor.
    
    Direct stress integration using the principle of virtual work.
    """
    
    # Test function
    v_test = ufl.TestFunction(V)

    gdim = domain.geometry.dim
    
    if material_behavior == 'elastic':
        # Compute stress from displacement
        stress = stress_func(u)

        # Internal force form (weak form of equilibrium): 
        # f_int = ∫ σ : ε(v) dV or f_int = ∫ B^T σ dV, 
        # with the strain-displacement matrix B
        internal_force_form = ufl.inner(stress, strain_func(v_test)) * ufl.dx
    
    elif material_behavior == 'hyperelastic':
        if configuration == 'reference':
            """
            In the reference configuration, internal forces are computed as:

            f_int = ∫ P : ∇v dV
            
            where P is the first Piola-Kirchhoff stress tensor,
            and v is the test function.

            stress_func: pk_1
            """
            # Test function
            v_test = TestFunction(V)
            
            # Define the internal force as the residual of the weak form
            # (without external forces since we assume f_body = 0)
            f_body = fem.Constant(domain, np.zeros(gdim))
            
            # Internal forces: f_int = ∫ P : ∇v dV
            internal_force_form = ufl.inner(stress_func, grad(v_test)) * ufl.dx
            
        elif configuration == 'current':
            """
            TODO - Check 

            Compute internal forces using Cauchy stress in current configuration.
            
            For this approach, we need to transform the integral to the 
            current configuration:

            f_int = ∫ σ : ∇v dv = ∫ J σ F^(-T) : ∇v dV ,
            
            where σ is the Cauchy stress, v is the current volume,
            and V is the reference volume.

            stress_func: pk_1
            """

            Id = Identity(gdim)

            # Deformation gradient
            F_def = ufl.variable(Id + ufl.grad(u))

            # Invariants
            J_det = ufl.det(F_def)

            # Cauchy stress: σ = (1/J) P F^T
            sigma_cauchy = (1/J_det) * stress_func * F_def.T
            
            # Create test function
            v_test = ufl.TestFunction(V)
            
            # Transform to reference configuration: ∫ J σ F^(-T) : ∇v dV
            # This is equivalent to ∫ P : ∇v dV (which is what we use above)
            F_inv_T = ufl.inv(F_def).T
            internal_force_form = ufl.inner(J_det * sigma_cauchy * F_inv_T, 
                                            ufl.grad(v_test)) * ufl.dx
            
    # Create internal force vector
    f_int_vec = fem.petsc.create_vector(fem.form(internal_force_form))
    # Assemble the internal force vector
    fem.petsc.assemble_vector(f_int_vec, fem.form(internal_force_form))
    
    # Update ghost values for parallel computations
    f_int_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, 
                        mode=PETSc.ScatterMode.REVERSE)
                    
    return f_int_vec


def validate_stress_integration_methods(domain, u, dx, V, 
                                        material_behavior,
                                        strain_func=None, stress_func=None):
    """
    Compare different methods for computing internal forces.
    This helps validate that the stress integration is working correctly.
    """
    print("Validating stress integration methods...")

    if material_behavior == 'elastic':
        # Method 1: residual
        force_vec_residual = compute_forces_residual(
            domain, u=u, dx=dx, V=V,
            strain_func=strain_func, stress_func=stress_func,
            material_behavior=material_behavior)
        
        # Method 3: Cauchy stress integration
        force_vec_cauchy = compute_forces_stress_integration(
            domain, u=u, V=V,
            strain_func=strain_func, stress_func=stress_func,
            material_behavior=material_behavior)
        
        # Compare results
        residual_array = force_vec_residual.getArray()
        cauchy_array = force_vec_cauchy.getArray()
        
        # Compute differences
        diff_cauchy_residual = np.linalg.norm(cauchy_array - residual_array)

        print(f"  ||F_Cauchy - F_residual|| = {diff_cauchy_residual:.3e}")
        
        # Clean up
        force_vec_residual.destroy()
        force_vec_cauchy.destroy()
    
        return diff_cauchy_residual < 1e-10

    elif material_behavior == 'hyperelastic':
        # Method 1: residual
        force_vec_residual = compute_forces_residual(
            domain, u=u, dx=dx, V=V,
            material_behavior=material_behavior, stress_func=stress_func)
        
        # Method 2: direct stress integration (PK1)
        force_vec_pk1 = compute_forces_stress_integration(
            domain, u=u, V=V, stress_func=stress_func,
            material_behavior=material_behavior, configuration='reference')
        
        # Method 3: Cauchy stress integration
        force_vec_cauchy = compute_forces_stress_integration(
            domain, u=u, V=V, stress_func=stress_func,
            material_behavior=material_behavior, configuration='current')
        
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
        
        return np.logical_and(diff_cauchy_residual < 1e-10, np.logical_and(
            diff_pk1_residual < 1e-10, diff_pk1_cauchy < 1e-10) )
