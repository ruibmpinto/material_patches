import numpy as np
import dolfinx
from dolfinx import fem
from petsc4py import PETSc
import ufl
from typing import Tuple, Optional, List, Set
import torch
import torch.nn as nn



class RegionIdentifier:
    """
    Helper class to identify nodes in different regions of the mesh.

    TODO:
        - Read region delimiter from material patch.pkl file
    """
    def __init__(self, mesh, function_space, region_bounds: dict):
        """
        Initialize region identifier.
        
        Args:
            mesh: DOLFINx mesh
            function_space: Function space
            region_bounds: Dictionary with 'xmin', 'xmax', 'ymin', 'ymax', 
            'zmin', 'zmax' for the region
        """
        self.mesh = mesh
        self.V = function_space
        self.region_bounds = region_bounds
        
        # Get DOF coordinates
        self.dof_coords = self.V.tabulate_dof_coordinates()
        
        # Identify different node sets
        self._identify_regions()
        
    def _identify_regions(self):
        """
        Identify nodes in interior, boundary, and exterior regions.
        """
        xmin, xmax = self.region_bounds['xmin'], self.region_bounds['xmax']
        ymin, ymax = self.region_bounds['ymin'], self.region_bounds['ymax']
        zmin, zmax = self.region_bounds['zmin'], self.region_bounds['zmax']

        # Get node coordinates (for vector spaces, 
        # we need to handle DOF grouping)
        if self.V.num_sub_spaces > 0:
            # Vector function space - group DOFs by node
            dim = self.V.num_sub_spaces
            num_nodes = len(self.dof_coords) // dim
            # Take every dim-th coordinate
            node_coords = self.dof_coords[::dim]  
        else:
            # Scalar function space
            node_coords = self.dof_coords
            dim = 1
            
        # Find nodes in the region
        in_region_mask = (
            (node_coords[:, 0] >= xmin) & (node_coords[:, 0] <= xmax) &
            (node_coords[:, 1] >= ymin) & (node_coords[:, 1] <= ymax) &
            (node_coords[:, 2] >= zmin) & (node_coords[:, 2] <= zmax)
        )
        
        # Find boundary nodes (on the edge of the region)
        tol = 1e-12
        boundary_mask = (
            ((np.abs(node_coords[:, 0] - xmin) < tol) | 
             (np.abs(node_coords[:, 0] - xmax) < tol) |
             (np.abs(node_coords[:, 1] - ymin) < tol) | 
             (np.abs(node_coords[:, 1] - ymax) < tol) |
             (np.abs(node_coords[:, 2] - zmin) < tol) | 
             (np.abs(node_coords[:, 2] - zmax) < tol)) &
            in_region_mask
        )
        
        # Interior nodes are in region but not on boundary
        interior_mask = in_region_mask & ~boundary_mask
        
        # Convert node indices to DOF indices
        self.boundary_node_indices = np.where(boundary_mask)[0]
        self.interior_node_indices = np.where(interior_mask)[0]
        self.exterior_node_indices = np.where(~in_region_mask)[0]
        
        # Convert to DOF indices for vector spaces
        if dim > 1:
            self.boundary_dof_indices = np.concatenate([
                self.boundary_node_indices * dim + i for i in range(dim)
            ])
            self.interior_dof_indices = np.concatenate([
                self.interior_node_indices * dim + i for i in range(dim)
            ])
            self.exterior_dof_indices = np.concatenate([
                self.exterior_node_indices * dim + i for i in range(dim)
            ])
        else:
            self.boundary_dof_indices = self.boundary_node_indices
            self.interior_dof_indices = self.interior_node_indices
            self.exterior_dof_indices = self.exterior_node_indices
            
        print(f"Material patch identification complete:")
        print(f"  Boundary nodes: {len(self.boundary_node_indices)}")
        print(f"  Interior nodes: {len(self.interior_node_indices)}")
        print(f"  Exterior nodes: {len(self.exterior_node_indices)}")
        print(f"  Boundary DOFs: {len(self.boundary_dof_indices)}")


class SurrogateNewtonSolver:
    """
    Newton solver with GRNN-based internal force computation 
    for material patches.
    """
    
    def __init__(self, 
                 problem,
                 model_path: str,
                 region_bounds: dict,
                 max_iter: int = 50,
                 atol: float = 1e-9,
                 rtol: float = 1e-9,
                 conv_criterion: str = "residual",
                 relax_param: float = 1.0,
                 linear_solver: str = "default",
                 preconditioner: str = "default",
                 device: str = "cpu"):
        """
        Initialize hybrid solver class.
        
        Args:
            problem: NonlinearPDE problem instance
            gnn_model_path: Path to the trained surrogate model
            region_bounds: Dictionary defining the material patch bounds
            device: PyTorch device ('cpu' or 'cuda')
        """
        self.problem = problem
        self.max_iter = max_iter
        self.atol = atol
        self.rtol = rtol
        self.conv_criterion = conv_criterion
        self.relax_param = relax_param
        self.device = torch.device(device)
        
        # Load surrogate model
        self.gnn_model = torch.load(model_path, map_location=self.device)
        self.gnn_model.eval()
        
        # Initialize region identifier
        self.region_id = RegionIdentifier(
            problem.u.function_space.mesh, 
            problem.u.function_space, 
            region_bounds
        )
        
        # Initialize linear solver
        self._setup_linear_solver(linear_solver, preconditioner)
        
        # Convergence history
        self.residual_norms = []
        self.increment_norms = []
        
    def _setup_linear_solver(self, solver_type: str, pc_type: str):
        """
        Setup PETSc linear solver.
        """
        self.ksp = PETSc.KSP().create()
        
        if solver_type != "default":
            self.ksp.setType(solver_type)
        
        pc = self.ksp.getPC()
        if pc_type != "default":
            pc.setType(pc_type)
            
        self.ksp.setTolerances(rtol=self.rtol, atol=self.atol, max_it=1000)
        
    def solve(self, u) -> Tuple[int, bool]:
        """
        Solve the nonlinear problem: Newton's + GNN solvers.
        
        Args:
            u: dolfinx.fem.Function containing the initial guess and solution
            
        Returns:
            Tuple of (number_of_iterations, converged_flag)
        """
        # Get the PETSc vector from the Function
        u_vec = u.x.petsc_vec
        
        # Clear convergence history
        self.residual_norms.clear()
        self.increment_norms.clear()
        
        # Create working vectors
        residual_vec = u_vec.duplicate()
        correction_vec = u_vec.duplicate()
        
        # Initial residual computation
        self._assemble_residual_with_gnn(u, u_vec, residual_vec)
        residual_norm0 = residual_vec.norm()
        residual_norm = residual_norm0
        
        self.residual_norms.append(residual_norm)
        
        print(f"Hybrid GNN-Newton solver starting: ||F||_0 = {residual_norm0:.3e}")
        
        converged = False
        iteration = 0
        
        # Check if already converged
        if residual_norm < self.atol:
            print(f"Hybrid GNN-Newton solver converged in {iteration} iterations")
            return iteration, True
            
        # Newton iteration loop
        for iteration in range(1, self.max_iterations + 1):
            # Assemble Jacobian matrix - modified for material patch
            jacobian_mat = self._assemble_jacobian_with_gnn(u_vec)
            
            # Apply boundary conditions to Jacobian and residual
            self._apply_boundary_conditions(jacobian_mat, residual_vec, u_vec)
            
            # Solve linear system: J * delta_u = -F
            residual_vec.scale(-1.0)  # Change sign for correction
            
            self.ksp.setOperators(jacobian_mat)
            self.ksp.solve(residual_vec, correction_vec)
            
            # Check linear solver convergence
            if self.ksp.getConvergedReason() < 0:
                print(f"Warning: Linear solver failed at iteration {iteration}")
                
            # Apply relaxation and update solution
            correction_vec.scale(self.relaxation_parameter)
            u_vec += correction_vec
            
            # Update the Function with the new vector values
            u.x.petsc_vec.copy(u_vec)
            u.x.scatter_forward()
            
            # Compute increment norm
            increment_norm = correction_vec.norm()
            self.increment_norms.append(increment_norm)
            
            # Reassemble residual with updated solution
            self._assemble_residual_with_gnn(u, u_vec, residual_vec)
            residual_norm = residual_vec.norm()
            self.residual_norms.append(residual_norm)
            
            # Check convergence
            converged = self._check_convergence(residual_norm, residual_norm0, 
                                             increment_norm, iteration)
            
            print(f'Hybrid GNN-Newton iteration {iteration}: '
                  f'||F|| = {residual_norm:.3e}, '
                  f'||du|| = {increment_norm:.3e}')
            
            if converged:
                print(f'Hybrid GNN-Newton solver converged in ' 
                      f'{iteration} iterations')
                break
                
            # Check for divergence
            if residual_norm > 1e3 * residual_norm0:
                print(f'Hybrid GNN-Newton solver diverged at '
                      f'iteration {iteration}')
                break
                
        # Cleanup
        jacobian_mat.destroy()
        residual_vec.destroy()
        correction_vec.destroy()
        
        return iteration, converged
    
    def _assemble_residual_with_gnn(self, u_func, residual_vec):
        """
        Assemble residual with GNN-computed internal forces.
        """
        # Reset residual vector
        residual_vec.zeroEntries()
        
        # Assemble standard residual only for exterior nodes
        # (This is a simplified approach - you might need to modify the form)
        fem.petsc.assemble_vector(residual_vec, self.problem.F)
        residual_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, 
                               mode=PETSc.ScatterMode.REVERSE)
        
        # Zero out residual for interior nodes (they don't participate in assembly)
        with residual_vec.localForm() as loc:
            for dof_idx in self.region_id.interior_dof_indices:
                if dof_idx < loc.getSize():
                    loc.setValue(dof_idx, 0.0)
        
        # Compute GNN-based internal forces for boundary nodes
        gnn_forces = self._compute_gnn_forces(u_func)
        
        # Add GNN forces to residual for boundary nodes
        with residual_vec.localForm() as loc:
            for i, dof_idx in enumerate(self.region_id.boundary_dof_indices):
                if dof_idx < loc.getSize() and i < len(gnn_forces):
                    current_val = loc.getValue(dof_idx)
                    loc.setValue(dof_idx, current_val + gnn_forces[i])
        
        residual_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, 
                               mode=PETSc.ScatterMode.FORWARD)
        
        # Apply boundary conditions
        valid_bcs = [bc for bc in self.problem.bcs if bc is not None]
        if valid_bcs:
            fem.petsc.set_bc(residual_vec, valid_bcs)
    
    def _compute_gnn_forces(self, u_func):
        """
        Compute internal forces using the GNN model.
        """
        # Extract displacements for boundary nodes
        u_array = u_func.x.array
        
        # Get boundary node displacements
        boundary_displacements = u_array[self.region_id.boundary_dof_indices]
        
        # Prepare input for surrogate ML model
        # Reshape for GRNN: (batch_size, sequence_length, input_dim)
        input_tensor = torch.tensor(
            # CHECK SHAPE!
            boundary_displacements.reshape(1, -1, 1),
            dtype=torch.float32,
            device=self.device
        )
        
        # Compute forces using surrogate model
        with torch.no_grad():
            forces_tensor = self.gnn_model(input_tensor)
            forces = forces_tensor.cpu().numpy().flatten()
        
        # Ensure forces array matches the number of boundary DOFs
        if len(forces) != len(self.region_id.boundary_dof_indices):
            # CHECK SIZE!!!
            forces = np.resize(forces, 
                               len(self.region_id.boundary_dof_indices))
        
        return forces
    
    def _assemble_jacobian_with_gnn(self, jacobian_mat):
        """
        Assemble Jacobian matrix with GNN region modifications.
        """
        # Create Jacobian matrix
        jacobian_mat = fem.petsc.create_matrix(self.problem.J)
        jacobian_mat.zeroEntries()
        
        # Filter boundary conditions
        valid_bcs = [bc for bc in self.problem.bcs if bc is not None]
        
        # Assemble standard Jacobian
        fem.petsc.assemble_matrix(jacobian_mat, self.problem.J, bcs=valid_bcs)
        jacobian_mat.assemble()
        
        # Modify Jacobian for GNN region
        # Material patch interior nodes are NOT considered in the assembly
        for dof_idx in self.region_id.interior_dof_indices:
            jacobian_mat.zeroRows([dof_idx], diag=1.0)
        
        return jacobian_mat
    
    def _apply_boundary_conditions(self, jacobian_mat, residual_vec, u_vec):
        """
        Apply boundary conditions to the linear system.
        """
        valid_bcs = [bc for bc in self.problem.bcs if bc is not None]
        
        if valid_bcs:
            fem.petsc.apply_lifting(residual_vec, [self.problem.J], 
                                  [valid_bcs], [u_vec], -1.0)
            residual_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, 
                                   mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(residual_vec, valid_bcs)
    

    def _check_convergence(self, residual_norm: float, residual_norm0: float,
                          increment_norm: float, iteration: int) -> bool:
        """
        Check Newton convergence based on selected criterion.
        """

        if self.convergence_criterion == "residual":
            abs_converged = residual_norm < self.atol
            rel_converged = residual_norm < self.rtol * residual_norm0
            return abs_converged or rel_converged
        elif self.convergence_criterion == "incremental":
            return increment_norm < self.atol
        else:
            raise ValueError(f'Unknown convergence criterion: '
                             f'{self.convergence_criterion}')
    
    def get_convergence_history(self):
        """
        Return convergence history.
        """
        return {
            'residual_norms': self.residual_norms.copy(),
            'increment_norms': self.increment_norms.copy()
        }
