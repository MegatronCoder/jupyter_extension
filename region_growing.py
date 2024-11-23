import numpy as np
from queue import Queue
import cv2
import matplotlib.pyplot as plt



# MANUAL IMPLEMENTATION 
 

import cv2
import numpy as np
import matplotlib.pyplot as plt

def region_growing(img, seed, threshold=10):
   
    # Get image dimensions
    rows, cols = img.shape
    # Create an empty output image
    output = np.zeros_like(img)
    # Create a list of visited pixels
    visited = np.zeros_like(img, dtype=bool)
    
    # Create a list of neighbors to check, 4-connectivity (up, down, left, right)
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]
    
    # Initialize a stack for the seed point
    stack = [seed]
    # Get the intensity of the seed
    seed_intensity = img[seed]
    
    while stack:
        x, y = stack.pop()
        
        # If already visited, skip
        if visited[x, y]:
            continue
        
        # Mark the pixel as visited
        visited[x, y] = True
        
        # Add the pixel to the region if it's within the threshold
        if abs(int(img[x, y]) - seed_intensity) < threshold:
            output[x, y] = 255  # Mark the pixel as part of the region
            
            # Add neighbors to the stack
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny]:
                    stack.append((nx, ny))
    
    return output

def display_image(image, title="Image"):
   
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

image_path = r'D:\College\pccoe\5th sem\CV\Practicals\Final_practicals\something-main\something-main\dog\Dog Segmentation\Images\dog.8790.jpg'  # Change this to the path of your image

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
bilateral = cv2.bilateralFilter(img, 5, 15, 15)

# Define the seed point (x, y) 
seed_point = (200, 200)  

# Apply Region Growing
segmented_image = region_growing(bilateral, seed_point, threshold=20)

# Display the original and segmented images
display_image(bilateral, "Original Image")
display_image(segmented_image, "Region Growing Segmentation")
