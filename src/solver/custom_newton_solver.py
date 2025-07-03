"""
References:

- https://jsdokken.com/dolfinx-tutorial/chapter4/newton-solver.html
- https://fenicsproject.discourse.group/t/custom-newton-solver-in-c/9471
- https://github.com/FEniCS/dolfinx/blob/9f55e8be10a3e3f6c3f2a2146395ddb2a332e5c1/cpp/dolfinx/nls/NewtonSolver.cpp#L148-L277
- https://docs.fenicsproject.org/dolfinx/main/python/_modules/dolfinx/nls/petsc.html#NewtonSolver

- https://fenicsproject.discourse.group/t/custom-newton-solver-problem-with-dirichlet-conditions/2329/12

- https://jsdokken.com/FEniCS-workshop/src/deep_dive/lifting.html
"""

#                                                                      Modules
#%% ===========================================================================

import numpy as np
import dolfinx
from dolfinx import fem
from petsc4py import PETSc
import ufl
from typing import Tuple, Optional
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira Morais Pinto (rbarreira@ethz.ch, ' \
'rui.pinto@brown.edu)'
__credits__ = ['Rui Barreira Morais Pinto', ]
__status__ = 'development'
# =============================================================================
#

class CustomNewtonSolver:
    """
    Custom Newton solver implementation in pure Python for FEniCSx.
    
    Newton's method for solving nonlinear problems
    of the form residual(u) = 0.
    """
    
    def __init__(self, 
                 problem,
                 max_iter: int = 50,
                 atol: float = 1e-7,
                 rtol: float = 1e-7,
                 conv_criterion: str = "residual",
                 relax_param: float = 0.0,
                 linear_solver: str = "default",
                 preconditioner: str = "default"):
        """
        Initialize the custom Newton solver.
        
        Args:
            problem: NonlinearPDE problem instance with F, J, bcs attributes
            max_iterations: Maximum number of Newton iterations
            absolute_tolerance: Absolute convergence tolerance
            relative_tolerance: Relative convergence tolerance  
            convergence_criterion: "residual" or "incremental"
            relaxation_parameter: Damping parameter for Newton updates
            linear_solver: Linear solver type for Newton corrections
            preconditioner: Preconditioner type
        """
        self.problem = problem
        self.max_iter = max_iter
        self.atol = atol
        self.rtol = rtol
        self.conv_criterion = conv_criterion
        self.relax_param = relax_param
        
        # Initialize linear solver for Newton corrections
        self._setup_linear_solver(linear_solver, preconditioner)
        
        # Convergence history
        self.residual_norms = []
        self.increment_norms = []
        
    def _setup_linear_solver(self, solver_type: str, pc_type: str):
        """
        Setup PETSc linear solver for Newton corrections.
        """
        self.ksp = PETSc.KSP().create()
        
        if solver_type != "default":
            self.ksp.setType(solver_type)
        
        pc = self.ksp.getPC()
        if pc_type != "default":
            pc.setType(pc_type)
        

        self.ksp.setTolerances(rtol=self.atol, atol=self.rtol, 
                               max_it=self.max_iter)
        
        
    def solve(self, u) -> Tuple[int, bool]:
        """
        Solve the nonlinear problem using Newton's method.
        
        Args:
            u: dolfinx.fem.Function containing the initial guess and solution
            
        Returns:
            Tuple of (number_of_iterations, converged_flag)
        """

        # Get the PETSc vector from the fem.Function
        u_vec = u.x.petsc_vec

        # Create Jacobian matrix
        jacobian_mat = fem.petsc.create_matrix(self.problem.J)
        # Create residual right hand side vector
        residual_vec = fem.petsc.create_vector(self.problem.F)
        # residual_vec = u_vec.duplicate()

        # Create increment vector
        delta_u = u_vec.duplicate()

        # Clear convergence history
        self.residual_norms.clear()
        self.increment_norms.clear()
        
        
        # Initial residual computation
        self._assemble_residual(residual_vec)
        residual_norm0 = 1E6 # residual_vec.norm()
        # residual_norm = residual_norm0
        
        # self.residual_norms.append(residual_norm)
        
        print(f'Newton solver starting') # : ||F||_0 = {residual_norm0:.3e}')
        
        # Loop control
        converged = False
        iteration = 0
        
        # Check if already converged
        # if residual_norm < self.atol:
        #     print(f"Newton solver converged in {iteration} iterations")
        #     return iteration, True
            
        # Newton iteration loop
        for iteration in range(1, self.max_iter + 1):
            # Assemble Jacobian matrix
            jacobian_mat = self._assemble_jacobian(jacobian_mat)
            self.ksp.setOperators(jacobian_mat)

            # Apply boundary conditions to the residual
            self._apply_boundary_conditions(residual_vec, u_vec)
            
            # Solve linear system: J * delta_u = -F
            # Change sign for correction
            residual_vec.scale(-1.0)  
            
            
            # Solve linear problem
            self.ksp.solve(residual_vec, delta_u)

            # Check linear solver convergence
            if self.ksp.getConvergedReason() < 0:
                print(f'Warning: Linear solver failed, iteration {iteration}')
                
            # # Apply relaxation and update solution
            # delta_u.scale(self.relax_param)
            # Update u_{i+1} = u_i + delta u_i
            u_vec += delta_u
            
            # Update the fem.Function with the new vector values
            u.x.petsc_vec.copy(u_vec)
            u.x.scatter_forward()

            # Compute increment norm
            increment_norm = delta_u.norm()
            self.increment_norms.append(increment_norm)
            
            # Reassemble residual with updated solution
            self._assemble_residual(residual_vec)
            residual_norm = residual_vec.norm()
            self.residual_norms.append(residual_norm)
            
            # Check convergence
            converged = self._check_convergence(residual_norm, residual_norm0, 
                                             increment_norm, iteration)
            
            print(f'Newton iteration {iteration}: '
                  f'||F|| = {residual_norm:.3e}, '
                  f'||du|| = {increment_norm:.3e}')
            
            if converged:
                print(f'Newton solver converged in {iteration} iterations')
                break
                
            # Check for divergence
            if residual_norm > 1e2 * residual_norm0:
                print(f'Newton solver diverged at iteration {iteration}')
                break
                
        # Cleanup
        jacobian_mat.destroy()
        residual_vec.destroy()
        delta_u.destroy()
        
        return iteration, converged
    
    def _assemble_residual(self, residual_vec):
        """

        Assemble the residual vector F(u).
        
        """
        # Reset residual vector
        residual_vec.zeroEntries()
        
        # Assemble the residual form
        fem.petsc.assemble_vector(residual_vec, self.problem.F)
        residual_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, 
                                 mode=PETSc.ScatterMode.REVERSE)

        
    def _assemble_jacobian(self, jacobian_mat):
        """
        
        Assemble the Jacobian matrix J(u).
        
        """
        # Create Jacobian matrix
        # jacobian_mat = fem.petsc.create_matrix(self.problem.J)
        jacobian_mat.zeroEntries()
        
        # Assemble Jacobian
        fem.petsc.assemble_matrix(jacobian_mat, self.problem.J, 
                                bcs=self.problem.bcs)
        jacobian_mat.assemble()
        
        return jacobian_mat
    
    def _apply_boundary_conditions(self, residual_vec, u_vec):
        """
        
        Apply boundary conditions to the linear system.
        
        """
        # Modify matrix and vector for boundary conditions
        # Compute b - J(u_D-u_(i-1))
        fem.petsc.apply_lifting(residual_vec, [self.problem.J], 
                              [self.problem.bcs], x0=[u_vec], alpha=-1.0)
        # Set du|_bc = u_{i-1}-u_D
        # Apply boundary conditions (set residual to 0 on constrained DOFs)
        fem.petsc.set_bc(residual_vec, self.problem.bcs, u_vec, -1.0)
        residual_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, 
                               mode=PETSc.ScatterMode.REVERSE)
        
        


    def _check_convergence(self, residual_norm: float, residual_norm0: float,
                          increment_norm: float, iteration: int) -> bool:
        """
        
        Check Newton convergence based on selected criterion.
        
        """
        if self.conv_criterion == "residual":
            # Absolute and relative residual criteria
            abs_converged = residual_norm < self.atol
            rel_converged = residual_norm < self.rtol * residual_norm0
            
            return abs_converged or rel_converged
            
        elif self.conv_criterion == "incremental":
            # Incremental convergence criterion
            return increment_norm < self.atol
            
        else:
            raise ValueError(f'Unknown convergence criterion: '
                             f'{self.conv_criterion}')
    
    def set_convergence_criteria(self, atol: float = None, rtol: float = None,
                               criterion: str = None):
        """
        Update convergence criteria.
        """
        if atol is not None:
            self.atol = atol
        if rtol is not None:
            self.rtol = rtol
        if criterion is not None:
            self.conv_criterion = criterion
            
    def set_relaxation_parameter(self, omega: float):
        """
        Set the relaxation parameter for damped Newton method.
        """
        if not 0 < omega <= 1.0:
            raise ValueError("Relaxation parameter must be in (0, 1]")
        self.relax_parameter = omega
        
    def get_convergence_history(self):
        """
        Return convergence history.
        """
        return {
            'residual_norms': self.residual_norms.copy(),
            'increment_norms': self.increment_norms.copy()
        }


class NonlinearPDE:
    """
    Example NonlinearPDE problem class structure.
    This should match the structure from the NewFrac training example.
    """
    def __init__(self, f_form, j_form, boundary_conditions, u_function):
        """
        Initialize nonlinear PDE problem.
        
        Args:
            F_form: UFL form for the residual F(u, v)
            J_form: UFL form for the Jacobian dF/du
            boundary_conditions: List of DirichletBC objects
            u_function: Function object for the solution
        """
        self.F = f_form
        self.J = j_form
        self.bcs = boundary_conditions
        self.u = u_function