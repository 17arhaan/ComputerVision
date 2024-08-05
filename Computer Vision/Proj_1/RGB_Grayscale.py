import cv2
from matplotlib import pyplot as plt
import os

# Define the path to the image
file_path = r"C:\Users\asus\OneDrive\Documents\Coding\DA-CV\Computer Vision\proj1\image.jpg"

# Check if the file exists
if os.path.exists(file_path):
    print("File exists.")
else:
    print("File does not exist. Please check the path.")
    exit()

# Read the image
img = cv2.imread(file_path)

# Check if the image was loaded successfully
if img is None:
    print("Error: Unable to load image.")
    exit()

# Convert the image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert the image to RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Limiting the pictures size
plt.figure(figsize=(10, 5))

# Plotting the RGB image
plt.subplot(1, 2, 1)
plt.title('RGB Image')
plt.imshow(img_rgb)
plt.axis('off')

# Plotting the grayscale image
plt.subplot(1, 2, 2)
plt.title('Grayscale Image')
plt.imshow(img_gray, cmap='gray')
plt.axis('off')  # Hide the axis

plt.show()

# Displaying both the images side by side
cv2.imwrite("image_gray.jpg", img_gray)
# Writing, saving the gray image
print("Grayscale image has been saved as 'image_gray.jpg'.")