import cv2
import numpy as np

def morphological_ops(rgb_image: np.ndarray, kernel_size:int) -> (np.ndarray,np.ndarray):
    '''
    Input
    rgb_image: 3 Channel RGB image in a numpy array
    kernel_size: kernel size for morphological operations
    '''

    dilated_image = None
    eroded_image = None
    image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=1)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    

    return (eroded_image,dilated_image)



def XY_to_idx(x_coord: float, y_coord: float, max_range: float, resolution: float) -> (int,int):
    '''
    Input
    x_coord: X-Coordinate of the point
    y_coord: Y-Coordinate of the point
    max_range: Maximum distance that a point can be
    resolution: (meters/px) value. Higher resolution indicates coarser maps.
    '''

    row_val = None
    col_val = None
    row_val = int((max_range + y_coord) / resolution)
    col_val = int((x_coord + max_range) / resolution)

    return (row_val,col_val)
    

 
 
