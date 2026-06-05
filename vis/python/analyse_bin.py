import numpy as np
import matplotlib.pyplot as plt

import bin_convert

# Binary files should be 2D. This works for SMR grids.

def make_2D_array(file_data, property):
    """ A function to make 2D arrays for each property. 
    The file_data is a dictionary containing the file data. See bin_convert.py for details.
    The property is the name of the property to be plotted.
    The function returns a 2D array for the property."""

    # Get the number of mesh blocks and their dimensions. This function only works for 2D bin files.
    nmb = file_data['n_mbs']    # Number of mesh blocks
    nx1 = file_data['nx1_mb']   # Number of cells in the x1 direction (horizontal) per meshblock
    nx2 = file_data['nx2_mb']   # Number of cells in the x2 direction (vertical). nx3 is 1 for 2D files.

    max_level = max(file_data['mb_logical'][i][3] for i in range(nmb))  # Get the maximum level of refinement in the mesh blocks

    property_arr = file_data['mb_data'][property]   # 4D array containing the property data for each mesh block
    # Initialize an empty array to hold the 2D data
    Arr = np.zeros((file_data['Nx2']*(2**max_level), file_data['Nx1']*(2**max_level)), dtype=property_arr[0].dtype)    # Nx2, Nx1 are total number of cells in root grid.

    for mb in range(nmb):
        mb_logical_indices = file_data['mb_logical'][mb]    # Logical indices of the mesh block in the 2D grid 
        refinement_level = mb_logical_indices[3]            # Refinement level of the mesh block
        factor = 2**(max_level - refinement_level)          # Factor to scale the mesh block size to the total grid size

        I = mb_logical_indices[0]*nx1*factor                       # Logical index in the x1 direction (horizontal)
        J = mb_logical_indices[1]*nx2*factor                       # Logical index in the x2 direction (vertical)


        block = property_arr[mb][0]
        refined_block = np.repeat(np.repeat(block, factor, axis=0), factor, axis=1)   # Refine the block by repeating its values

        Arr[J:J + nx2*factor, I:I + nx1*factor] = refined_block   # Place the refined block in the correct position in the 2D array
    return Arr

def make_3D_array(file_data, property):
    """ A function to make 3D arrays for each property. 
    The file_data is a dictionary containing the file data. See bin_convert.py for details.
    The property is the name of the property to be plotted.
    The function returns a 3D array for the property.
    The returned array has dimensions (Nx3, Nx2, Nx1)
    where Nx1, Nx2, Nx3 are the total number of cells in the root grid.
    - (0,0,0) of the array is at (zmin,ymin,xmin) of the root grid."""

    # Get the number of mesh blocks and their dimensions. This function works for 3D bin files.
    nmb = file_data['n_mbs']    # Number of mesh blocks
    nx1 = file_data['nx1_mb']   # Number of cells in the x1 direction (horizontal) per meshblock
    nx2 = file_data['nx2_mb']   # Number of cells in the x2 direction (vertical).
    nx3 = file_data['nx3_mb']   # Number of cells in the x3 direction (depth). nx3 is 1 for 2D files.

    max_level = max(file_data['mb_logical'][i][3] for i in range(nmb))  # Get the maximum level of refinement in the mesh blocks

    property_arr = file_data['mb_data'][property]   # 4D array containing the property data for each mesh block
    # Initialize an empty array to hold the 3D data
    Arr = np.zeros(( file_data['Nx3']*(2**max_level), file_data['Nx2']*(2**max_level), file_data['Nx1']*(2**max_level)), dtype=property_arr[0].dtype)    # Nx2, Nx1 are total number of cells in root grid.

    for mb in range(nmb):
        mb_logical_indices = file_data['mb_logical'][mb]    # Logical indices of the mesh block in the 3D grid 
        refinement_level = mb_logical_indices[3]            # Refinement level of the mesh block
        factor = 2**(max_level - refinement_level)          # Factor to scale the mesh block size to the total grid size

        I = mb_logical_indices[0]*nx1*factor                       # Logical index in the x1 direction (horizontal)
        J = mb_logical_indices[1]*nx2*factor                       # Logical index in the x2 direction (vertical)
        K = mb_logical_indices[2]*nx3*factor                       # Logical index in the x3 direction (depth)

        block = property_arr[mb]
        refined_block = np.repeat(np.repeat(np.repeat(block, factor, axis=0), factor, axis=1), factor, axis=2)   # Refine the block by repeating its values

        Arr[ K:K + nx3*factor, J:J + nx2*factor, I:I + nx1*factor] = refined_block   # Place the refined block in the correct position in the 3D array
    return Arr

def plot_figure(file_data, property):
    """ A function to plot the a property using matplotlib.
     Takes file_data dictionary and property and plots a figure."""

    Arr = make_2D_array(file_data, property)      # Make 2D array for the property

    plt.figure(figsize=(10, 6))
    plt.imshow(Arr, cmap='viridis', aspect='auto')
    plt.colorbar(label=property)
    plt.gca().invert_yaxis()                      # Invert the y-axis so that (0,0) is at the bottom left corner in the figure.
    plt.clim(vmin=np.min(Arr), vmax=np.max(Arr))  # Set the color limits to the min and max of the data
    #plt.title(f'2D Plot of {property} at Time = {file_data["time"]}, res = {file_data['Nx1']}x{file_data['Nx2']}')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.show()

def give_slice(Arr, index = 0, axis='z'):
    """ A function to give a slice of the 3D array for a given property.
    Takes file_data dictionary and property and returns a 2D slice of the 3D array.
    The returned array is sliced at the said index along a plane perpendicular to the specified axis.
    The return arrays are as follows:
    - If axis is 'x', array = (Y,Z)
    - If axis is 'y', array = (X,Z)
    - If axis is 'z', array = (Y,X)"""

    if axis == 'x':
        return Arr[:,:,index].T
    elif axis == 'y':
        return Arr[:,index, :].T
    else:
        return Arr[index, :, :]

def plot_3D_figure(file_data, property, index, axis=2):
    """ A function to plot the a property using matplotlib.
     Takes file_data dictionary and property and plots a figure."""

    Arr = make_3D_array(file_data, property)      # Make 3D array for the property

    plt.figure(figsize=(2, 8))

    if axis == 0:
        plt.imshow(Arr[index,:,:], cmap='viridis', aspect='auto', origin='lower')
    elif axis == 1:
        plt.imshow(Arr[:,index, :], cmap='viridis', aspect= 'auto', origin='lower')
    else:
        plt.imshow(Arr[:, :, index].T, cmap='viridis', aspect='auto', origin='lower')
    
    plt.colorbar(label=property)

    #plt.title(f'3D Plot of {property} at Time = {file_data["time"]}, res = {file_data['Nx1']}x{file_data['Nx2']}x{file_data['Nx3']}')
    plt.show()

if __name__ == "__main__" :

    file_path = input("Enter the path to the .bin file: ")
    prop = input("Enter the property to plot ('dens', 'velx', 'vely', 'eint', 's_00'): ")
    
    file_data = bin_convert.read_binary(file_path)
    data_dict = file_data['mb_data']

    head = file_data['header']
    n_mb = file_data['n_mbs']
    nx1 = file_data['nx1_mb']
    nx2 = file_data['nx2_mb']
    nx3 = file_data['nx3_mb']
    time = file_data['time']
    var_names = file_data['var_names']
    Nx1 = file_data['Nx1']
    Nx2 = file_data['Nx2']
    Nx3 = file_data['Nx3']
    nvars = file_data['nvars']
    mb_index = file_data['mb_index']
    mb_logical = file_data['mb_logical']
    mb_geometry = file_data['mb_geometry']
    nx1_out_mb = file_data['nx1_out_mb']
    nx2_out_mb = file_data['nx2_out_mb']
    nx3_out_mb = file_data['nx3_out_mb']
    


    
    print(f"Header: {head}")
    print(f"Number of mesh blocks: {n_mb}")
    print(f"Mesh block dimensions: {nx1} x {nx2} x {nx3}")
    print(f"Time: {time}")
    print(f"Variable names: {var_names}")
    print(f"Number of variables: {nvars}")
    print(f"Mesh block index: {mb_index}")
    print(f"Mesh block logical: {mb_logical}")
    print(f"Data dictionary keys: {data_dict.keys()}")
    #print(f'Total cells in the grid ={Nx1}*{Nx2}*{Nx3} = {Nx1 * Nx2 * Nx3}')
    #print(f'nx1_out_mb: {nx1_out_mb}')
    #print(f'nx2_out_mb: {nx2_out_mb}')
    #print(f'nx3_out_mb: {nx3_out_mb}')
    #print(f'Mesh block geometry: {mb_geometry}')

    plot_3D_figure(file_data, prop, 10, 0)
    plot_3D_figure(file_data, prop, 2*320//3, 1)
    plot_3D_figure(file_data, prop, 10, 2)




