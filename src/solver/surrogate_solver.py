#                                                                      Modules
#%% ===========================================================================

import torch
import torch.nn as nn
import numpy as np
import dolfinx
from dolfinx import fem
from petsc4py import PETSc
import ufl
from typing import Tuple, Optional, Union
from pathlib import Path
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira Morais Pinto (rbarreira@ethz.ch, ' \
'rui.pinto@brown.edu)'
__credits__ = ['Rui Barreira Morais Pinto', ]
__status__ = 'development'
# =============================================================================
#

class SurrogateSolver:
    """
    GNN-based surrogate solver that replaces Newton iterations with 
    direct displacement-to-force mapping.
    """
    
    def __init__(self, 
                 gnn_model: nn.Module,
                 function_space,
                 device: str = 'cpu',
                 model_path: Optional[str] = None,
                 input_scaler = None,
                 output_scaler = None,
                 sequence_length: int = 1,
                 use_history: bool = False,
                 max_sequence_length: int = 10):
        """
        Initialize the GNN surrogate solver.
        
        Args:
            gnn_model: Trained PyTorch GNN model
            function_space: FEniCSx function space for the problem
            device: PyTorch device ('cpu' or 'cuda')
            model_path: Path to saved model weights (optional)
            input_scaler: Scaler for input normalization (e.g., 
                sklearn StandardScaler)
            output_scaler: Scaler for output denormalization
            sequence_length: Length of displacement history to use
            use_history: Whether to use displacement history for prediction
            max_sequence_length: Maximum sequence length to maintain
        """
        self.gnn_model = gnn_model
        self.function_space = function_space
        self.device = torch.device(device)
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.sequence_length = sequence_length
        self.use_history = use_history
        self.max_sequence_length = max_sequence_length
        
        # Move model to device and set to evaluation mode
        self.gnn_model.to(self.device)
        self.gnn_model.eval()
        
        # Load model weights if path provided
        if model_path:
            self.load_model(model_path)
            
        # Initialize displacement history buffer
        self.displacement_history = []
        
        # Get DOF information
        self.num_dofs = self.function_space.dofmap.index_map.size_local
        self.dof_coordinates = self._get_dof_coordinates()
        
        print(f"GNN Surrogate Solver initialized:")
        print(f"  - Device: {self.device}")
        print(f"  - DOFs: {self.num_dofs}")
        print(f"  - Sequence length: {self.sequence_length}")
        print(f"  - Use history: {self.use_history}")
    


    def load_model(self, model_path: str):
        """
        Load trained model weights.
        """

        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.gnn_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.gnn_model.load_state_dict(checkpoint)

        print(f"Loaded model from {model_path}")
    


    def _get_dof_coordinates(self):
        """
        Get coordinates of DOFs for spatial features if needed.
        """
        # Retrieve spatial information
        try:
            # For vector function spaces
            if self.function_space.num_sub_spaces > 0:
                # Get coordinates for vector space
                x = self.function_space.tabulate_dof_coordinates()
                # Take only local DOFs
                return x[:self.num_dofs]  
            else:
                # For scalar function spaces
                x = self.function_space.tabulate_dof_coordinates()
                return x[:self.num_dofs]
        except Exception as e:
            print(f"Warning: Could not get DOF coordinates: {e}")
            return None
    
    def solve(self, u_vec) -> Tuple[int, bool]:
        """
        'Solve' using surrogate model: directly predicts forces 
        from displacements.
        
        Args:
            u_vec: PETSc vector containing current displacement field
            
        Returns:
            Tuple of (1, True) to maintain compatibility with 
            Newton solver interface.
        """
        # Convert PETSc vector to numpy array
        displacement_array = self._petsc_to_numpy(u_vec)
        
        # Update displacement history if using temporal features
        if self.use_history:
            self._update_displacement_history(displacement_array)
            input_sequence = self._prepare_sequence_input()
        else:
            # (batch, seq, features)
            input_sequence = displacement_array.reshape(1, 1, -1)  
        
        # Predict forces using GNN model
        predicted_forces = self._predict_forces(input_sequence)
        
        # Update the solution vector with predicted forces
        # Update u_vec with the predicted values
        self._numpy_to_petsc(predicted_forces, u_vec)
        
        # Return (1 iteration, converged=True) for compatibility
        return 1, True
    
    def _petsc_to_numpy(self, petsc_vec) -> np.ndarray:
        """
        Convert PETSc vector to numpy array.
        """
        # Get local array from PETSc vector
        with petsc_vec.localForm() as loc_vec:
            return loc_vec.array.copy()
    
    def _numpy_to_petsc(self, numpy_array: np.ndarray, petsc_vec):
        """
        Update PETSc vector with numpy array values.
        """
        with petsc_vec.localForm() as loc_vec:
            loc_vec.array[:] = numpy_array
        
        petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, 
                             mode=PETSc.ScatterMode.FORWARD)
    
    def _update_displacement_history(self, displacement_array: np.ndarray):
        """
        Update the displacement history buffer.
        """
        self.displacement_history.append(displacement_array.copy())
        
        # Maintain maximum sequence length
        if len(self.displacement_history) > self.max_sequence_length:
            self.displacement_history.pop(0)
    
    def _prepare_sequence_input(self) -> np.ndarray:
        """
        Prepare sequence input for GRU model.
        """
        if len(self.displacement_history) == 0:
            raise ValueError("No displacement history available")
        
        # Take the most recent sequence_length steps
        seq_len = min(self.sequence_length, len(self.displacement_history))
        sequence = self.displacement_history[-seq_len:]
        
        # Pad sequence if needed
        while len(sequence) < self.sequence_length:
            sequence.insert(0, np.zeros_like(sequence[0]))
        
        # Convert to numpy array: (batch=1, sequence_length, features)
        sequence_array = np.array(sequence).reshape(1, self.sequence_length, -1)
        
        return sequence_array
    
    def _predict_forces(self, input_sequence: np.ndarray) -> np.ndarray:
        """
        Predict forces using the GRU model.
        """
        # Apply input scaling if available
        if self.input_scaler is not None:
            # Reshape for scaling: (batch * seq_len, features)
            original_shape = input_sequence.shape
            input_flat = input_sequence.reshape(-1, input_sequence.shape[-1])
            input_scaled = self.input_scaler.transform(input_flat)
            input_sequence = input_scaled.reshape(original_shape)
        
        # Convert to PyTorch tensor
        input_tensor = torch.FloatTensor(input_sequence).to(self.device)
        
        # Forward pass through GRU model
        with torch.no_grad():
            # GRU output: batch, seq_len, output_features) or 
            # (batch, output_features)
            output = self.gnn_model(input_tensor)
            
            # Handle different output shapes
            if len(output.shape) == 3:  # (batch, seq_len, features)
                # Take the last time step
                output = output[:, -1, :]  # (batch, features)
            
            # Convert back to numpy
            predicted_forces = output.cpu().numpy()
        
        # Apply output scaling if available
        if self.output_scaler is not None:
            predicted_forces = self.output_scaler.inverse_transform(
                predicted_forces)
        
        # Return flattened array for single sample
        return predicted_forces.flatten()
    
    def predict_forces_from_displacement(self, displacement_field) -> np.ndarray:
        """
        Direct method to predict forces from displacement field.
        Useful for standalone predictions.
        """
        if isinstance(displacement_field, fem.Function):
            displacement_array = displacement_field.vector.array.copy()
        elif hasattr(displacement_field, 'array'):
            displacement_array = displacement_field.array.copy()
        else:
            displacement_array = np.array(displacement_field)
        
        # Prepare input
        if self.use_history and len(self.displacement_history) > 0:
            self._update_displacement_history(displacement_array)
            input_sequence = self._prepare_sequence_input()
        else:
            input_sequence = displacement_array.reshape(1, 1, -1)
        
        return self._predict_forces(input_sequence)
    
    def reset_history(self):
        """
        Reset displacement history buffer.
        """

        self.displacement_history.clear()
        print("Displacement history reset")
    
    def set_sequence_length(self, new_length: int):
        """
        Update sequence length for temporal modeling.
        """
        self.sequence_length = new_length
        print(f"Sequence length updated to {new_length}")










class SurrogateProblem:
    """
    Wrapper class to ensure compatibility between problem interfaces.
    This replaces the traditional NonlinearPDE problem class.
    """
    
    def __init__(self, gnn_solver: SurrogateSolver, boundary_conditions=None):
        """
        Initialize surrogate ML-based problem wrapper.
        
        Args:
            gnn_solver: SurrogateSolver instance
            boundary_conditions: boundary conditions (for compatibility)
        """
        self.gnn_solver = gnn_solver
        self.bcs = boundary_conditions or []
        
        # Dummy forms for compatibility (not actually used)
        self.F = None
        self.J = None
    
    def solve(self, u_vec) -> Tuple[int, bool]:
        """
        Delegate to GNN solver.
        """

        return self.gnn_solver.solve(u_vec)


# Example usage functions
def create_gru_solver_example():
    """
    Example of how to set up and use the GRU surrogate solver.
    """
    
    # Example GRU model definition (replace with your actual model)
    class DisplacementToForceGRU(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
            self.dropout = nn.Dropout(0.1)
            
        def forward(self, x):
            gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_size)
            # Take last output
            last_output = gru_out[:, -1, :]  # (batch, hidden_size)
            output = self.fc(self.dropout(last_output))
            return output
    
    # This is a template - replace with your actual setup
    # 
    # # Create your mesh and function space
    mesh = dolfinx.mesh.create_unit_square(...)
    V = dolfinx.fem.VectorFunctionSpace(mesh, ("CG", 1))
    u = dolfinx.fem.Function(V)
    # 
    # # Initialize your trained GRU model
    # Number of DOFs
    input_size = V.dofmap.index_map.size_local  
    gnn_model = # READ from file 
    # DisplacementToForceGRU(
    #     input_size=input_size,
    #     hidden_size=128,
    #     num_layers=2,
    #     output_size=input_size)
    # # 
    # Create GRU surrogate solver
    gru_solver = SurrogateSolver(
        gnn_model=gnn_model,
        function_space=V,
        device='cpu',  # or 'cuda' if available
        model_path='path/to/your/trained_model.pth',
        use_history=True,
        sequence_length=5
    )
    
    # Create problem wrapper (for compatibility)
    problem = SurrogateProblem(gru_solver)
    
    # Use exactly like the original solver!
    num_iter, converged = problem.solve(u.vector)
    # OR directly:
    # num_iter, converged = gru_solver.solve(u.vector)
    
    pass


def load_and_use_pretrained_gru():
    """Example of loading a pre-trained GRU model and using it."""
    
    # Load your trained model
    # model_path = "path/to/your/displacement_to_force_gru.pth"
    # 
    # # Recreate model architecture (must match training)
    # gnn_model = DisplacementToForceGRU(
    #     input_size=your_input_size,
    #     hidden_size=your_hidden_size,
    #     num_layers=your_num_layers,
    #     output_size=your_output_size
    # )
    # 
    # # Create solver with pre-trained weights
    # solver = SurrogateSolver(
    #     gnn_model=gnn_model,
    #     function_space=your_function_space,
    #     model_path=model_path,
    #     device='cuda' if torch.cuda.is_available() else 'cpu'
    # )
    # 
    # # Use in your simulation loop
    # for load_step in load_steps:
    #     # Apply boundary conditions, loads, etc.
    #     # ...
    #     
    #     # Solve using GRU (replaces Newton solver)
    #     num_its, converged = solver.solve(u.vector)
    #     
    #     # Post-process results
    #     # ...
    
    pass
