
import numpy as np
from dolfinx import fem

def find_nodes(coords, gdim, V):
    """
    Finds the nodes in function space V_space whose coordinates match
    those in coords_array.

    coords_array should be a single (x,y,z) coordinate array.

    # FEniCSx 0.9.0 changed the API for locating nodes!

    Returns a list of global NODE indices.
    """
   
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

def find_boundary_nodes(domain, V):
    """Find nodes on the boundary of the domain"""

    # Create a function to mark boundary
    def boundary_marker(x):
        # For unit square/cube, boundary is where any coordinate is 0 or 1
        if domain.geometry.dim == 2:
            return np.logical_or(
                np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
                np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))
            )
        elif domain.geometry.dim == 3:
            return np.logical_or(
                np.logical_or(
                    np.logical_or(np.isclose(x[0], 0.0), 
                                  np.isclose(x[0], 1.0)),
                    np.logical_or(np.isclose(x[1], 0.0), 
                                  np.isclose(x[1], 1.0))
                ),
                np.logical_or(np.isclose(x[2], 0.0),
                              np.isclose(x[2], 1.0))
            )
    
    # Find boundary DOFs
    boundary_dofs = fem.locate_dofs_geometrical(V, boundary_marker)
    
    return boundary_dofs

def apply_displacement_bc(v_space, gdim, coords, displacement_values):
    """
    Apply displacement boundary condition at given coordinates.

    Refs:
    - https://jsdokken.com/dolfinx-tutorial/chapter2/linearelasticity_code.html
    """
    # Find nodes at the specified coordinates
    nodes = find_nodes(coords, gdim=gdim, V=v_space)
    
    if len(nodes) == 0:
        return None

    print(f'    nodes: {nodes}, displacement_values: {displacement_values}')

    return fem.dirichletbc(displacement_values, nodes, v_space)


def extract_nodal_forces(forces_internal, boundary_node_coords, gdim, V):
    """
    Extract forces at specific boundary nodes from the global
      force vector.
    """
    forces_dict = {}
    
    # Get reaction vector as numpy array
    force_array = forces_internal.getArray()

    # print(f'np.shape(reaction_array): {np.shape(reaction_array)}')

    # print(f'reaction_array: {reaction_array}')
    
    for node_label, coords in boundary_node_coords.items():
        # Find node_idx at these coordinates
        nodes = find_nodes(coords, gdim=gdim, V=V)
        
        # If we did not find anythingm raise error
        if len(nodes) == 0:
            raise RuntimeError
            
        # Extract forces at these DOFs
        node_reactions = []
        if nodes[0] <= len(force_array):
            if gdim == 3:
                # x component
                node_reactions.append(force_array[3*nodes[0]])
                # y component
                node_reactions.append(force_array[3*nodes[0]+1] )
                # z component
                node_reactions.append(force_array[3*nodes[0]+2] )
                # Save to dict - +1 to match LINCS node numbering, first idx=1
                forces_dict[nodes[0]] = np.array(
                    [node_reactions[0], node_reactions[1], node_reactions[2]])
            elif gdim == 2:
                # For 2D, we expect 2 components (x and y)
                # x component
                node_reactions.append(force_array[2*nodes[0]])
                # y component
                node_reactions.append(force_array[2*nodes[0]+1] )
                # Save to dict - +1 to match LINCS node numbering, first idx=1
                forces_dict[nodes[0]] = np.array(
                    [node_reactions[0], node_reactions[1]])
        else:
            raise RuntimeError

    return forces_dict

def extract_boundary_displacements(u, boundary_nodes, V, gdim):
    """
    Extract displacement values for boundary nodes
    
    Parameters:
    - u: displacement solution function
    - boundary_nodes: list of boundary node indices
    - V: function space
    - gdim: geometric dimension (2 or 3)
    
    Returns:
    - Dictionary mapping node index to displacement vector
    """
    boundary_displacements = {}
    
    # Get all DOF coordinates
    dof_coords = V.tabulate_dof_coordinates()
    
    # Extract displacement values at boundary nodes
    u_values = u.x.array
    
    for node_idx in boundary_nodes:
        if gdim == 2:
            # For 2D: extract x and y displacements
            disp = np.array([u_values[node_idx * gdim],
                              u_values[node_idx * gdim + 1]])
        elif gdim == 3:
            # For 3D: extract x, y, and z displacements
            disp = np.array([u_values[node_idx * gdim], 
                            u_values[node_idx * gdim + 1], 
                            u_values[node_idx * gdim + 2]])
        
        boundary_displacements[node_idx] = disp
    
    return boundary_displacements

