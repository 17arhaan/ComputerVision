import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from scipy.ndimage import median_filter, maximum_filter, minimum_filter, gaussian_filter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2

# Functions
def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        global original_image, modified_image, display_original_image, display_modified_image
        original_image = Image.open(file_path).convert("RGB")
        modified_image = original_image.copy()
        display_original_image = ImageTk.PhotoImage(original_image.resize((400, 400)))
        original_image_label.config(image=display_original_image)
        original_image_label.image = display_original_image
        update_modified_image()

def update_modified_image():
    global modified_image, display_modified_image
    display_modified_image = ImageTk.PhotoImage(modified_image.resize((400, 400)))
    modified_image_label.config(image=display_modified_image)
    modified_image_label.image = display_modified_image
    plot_histogram(modified_image)  # Update histogram with the modified image

def smooth_image():
    global modified_image
    if original_image:
        try:
            sigma = float(sigma_entry.get())
        except ValueError:
            print("Invalid sigma value")
            return

        image_np = np.array(original_image)
        smoothed_image_np = gaussian_filter(image_np, sigma=sigma)
        modified_image = Image.fromarray(smoothed_image_np.astype(np.uint8))
        update_modified_image()
        plot_kernel(sigma)

def plot_kernel(sigma):
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    x = np.linspace(-sigma, sigma, kernel_size)
    y = np.linspace(-sigma, sigma, kernel_size)
    X, Y = np.meshgrid(x, y)
    gaussian_kernel = np.exp(-0.5 * (X ** 2 + Y ** 2) / sigma ** 2)
    gaussian_kernel /= gaussian_kernel.sum()

    kernel_window = tk.Toplevel(window)
    kernel_window.title(f"Gaussian Kernel (sigma={sigma})")

    figure, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(gaussian_kernel, cmap='viridis', interpolation='nearest')
    figure.colorbar(cax)
    ax.set_title(f"Gaussian Kernel (sigma={sigma})")
    plt.xlabel('X axis')
    plt.ylabel('Y axis')

    canvas = FigureCanvasTkAgg(figure, master=kernel_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def apply_median_filter():
    global modified_image
    if original_image:
        size = int(median_slider.get())
        image_np = np.array(original_image)
        filtered_image_np = median_filter(image_np, size=(size, size, 1))
        modified_image = Image.fromarray(filtered_image_np)
        update_modified_image()

def apply_max_filter():
    global modified_image
    if original_image:
        size = int(max_min_slider.get())
        image_np = np.array(original_image)
        filtered_image_np = maximum_filter(image_np, size=(size, size, 1))
        modified_image = Image.fromarray(filtered_image_np)
        update_modified_image()

def apply_min_filter():
    global modified_image
    if original_image:
        size = int(max_min_slider.get())
        image_np = np.array(original_image)
        filtered_image_np = minimum_filter(image_np, size=(size, size, 1))
        modified_image = Image.fromarray(filtered_image_np)
        update_modified_image()

def sharpen_image():
    global modified_image
    if original_image:
        image_np = np.array(original_image)
        sharpened_image_np = apply_sharpening_matrix(image_np)
        modified_image = Image.fromarray(sharpened_image_np)
        update_modified_image()

def apply_sharpening_matrix(image):
    sharpening_filter = np.array([[0, -0.25, 0], [-0.25, 2, -0.25], [0, -0.25, 0]])
    image = image.astype(np.float32)
    pad_size = 1
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
    sharpened_image = np.zeros_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                sharpened_image[y, x, c] = np.sum(padded_image[y:y + 3, x:x + 3, c] * sharpening_filter)
    return np.clip(sharpened_image, 0, 255).astype(np.uint8)

def reset_image():
    global modified_image
    if original_image:
        modified_image = original_image.copy()
        update_modified_image()

def unsharp_mask():
    global modified_image
    if original_image:
        try:
            sigma = float(sigma_entry.get())
            amount = float(amount_entry.get())
        except ValueError:
            print("Invalid sigma or amount value")
            return

        image_np = np.array(original_image)
        blurred_image_np = gaussian_filter(image_np, sigma=sigma)
        sharpened_image_np = image_np + (image_np - blurred_image_np) * amount
        sharpened_image_np = np.clip(sharpened_image_np, 0, 255).astype(np.uint8)
        modified_image = Image.fromarray(sharpened_image_np)
        update_modified_image()

def exit_fullscreen_mode(event=None):
    window.attributes('-fullscreen', False)
    window.geometry('1200x800')

def enter_fullscreen_mode(event=None):
    window.attributes('-fullscreen', True)

def toggle_fullscreen(event=None):
    if window.attributes('-fullscreen'):
        exit_fullscreen_mode()
    else:
        enter_fullscreen_mode()

def plot_histogram(image):
    figure, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    channels = ('Red', 'Green', 'Blue')
    for i, color in enumerate(channels):
        axes[i].hist(np.array(image)[:, :, i].ravel(), bins=256, range=(0, 256), color=color.lower())
        axes[i].set_title(f"{color} Channel")
    for widget in histogram_frame.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(figure, master=histogram_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def apply_kmeans_clustering():
    global modified_image
    if original_image:
        try:
            n_clusters = int(kmeans_entry.get())
        except ValueError:
            print("Invalid number of clusters")
            return

        image_np = np.array(original_image)
        pixel_values = image_np.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(pixel_values)
        centers = np.uint8(kmeans.cluster_centers_)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(image_np.shape)

        modified_image = Image.fromarray(segmented_image)
        update_modified_image()

def apply_canny_edge_detection():
    global modified_image
    if original_image:
        image_np = np.array(original_image.convert('L'))
        edges = cv2.Canny(image_np, threshold1=100, threshold2=200)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        modified_image = Image.fromarray(edges_rgb)
        update_modified_image()

def apply_color_detection():
    global modified_image
    if original_image:
        color_name = color_entry.get().lower()
        hsv_ranges = {
            'red': ([0, 50, 50], [10, 255, 255]),
            'green': ([35, 50, 50], [85, 255, 255]),
            'blue': ([100, 150, 150], [140, 255, 255]),
            'yellow': ([20, 100, 100], [30, 255, 255]),
        }
        if color_name not in hsv_ranges:
            print("Color not found in predefined list.")
            return
        lower_bound, upper_bound = hsv_ranges[color_name]
        lower_bound = np.array(lower_bound)
        upper_bound = np.array(upper_bound)

        image_np = np.array(original_image)
        hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        result_np = cv2.bitwise_and(image_np, image_np, mask=mask)

        modified_image = Image.fromarray(result_np)
        update_modified_image()

def apply_log_transform():
    global modified_image
    if original_image:
        c = 255 / np.log(1 + 255)
        epsilon = 1e-10
        image_np = np.array(original_image)
        log_image_np = c * np.log(1 + image_np + epsilon)
        log_image_np = np.clip(log_image_np, 0, 255).astype(np.uint8)
        modified_image = Image.fromarray(log_image_np)
        update_modified_image()

def capture_frame():
    global video_capture
    ret, frame = video_capture.read()
    if ret:
        cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(cv2_image)
        display_image = ImageTk.PhotoImage(image.resize((400, 400)))
        live_image_label.config(image=display_image)
        live_image_label.image = display_image

def start_video_capture():
    global video_capture
    video_capture = cv2.VideoCapture(0)
    video_capture_thread()

def video_capture_thread():
    if video_capture.isOpened():
        capture_frame()
        window.after(30, video_capture_thread)

def stop_video_capture():
    global video_capture
    if video_capture:
        video_capture.release()
        video_capture = None
        live_image_label.config(image='')

# Create main window
window = tk.Tk()
window.title("Image Processing Tool")
window.geometry('1200x800')
window.bind('<F11>', toggle_fullscreen)
window.bind('<Escape>', exit_fullscreen_mode)

# Original image display
original_image_label = tk.Label(window)
original_image_label.pack(side=tk.LEFT, padx=10, pady=10)

# Modified image display
modified_image_label = tk.Label(window)
modified_image_label.pack(side=tk.LEFT, padx=10, pady=10)

# Live image display
live_image_label = tk.Label(window)
live_image_label.pack(side=tk.LEFT, padx=10, pady=10)

# Controls frame
controls_frame = tk.Frame(window)
controls_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

# Buttons and sliders
open_button = tk.Button(controls_frame, text="Open Image", command=open_image)
open_button.pack(side=tk.LEFT, padx=5)

reset_button = tk.Button(controls_frame, text="Reset", command=reset_image)
reset_button.pack(side=tk.LEFT, padx=5)

smooth_button = tk.Button(controls_frame, text="Smooth Image", command=smooth_image)
smooth_button.pack(side=tk.LEFT, padx=5)

median_button = tk.Button(controls_frame, text="Median Filter", command=apply_median_filter)
median_button.pack(side=tk.LEFT, padx=5)

max_button = tk.Button(controls_frame, text="Max Filter", command=apply_max_filter)
max_button.pack(side=tk.LEFT, padx=5)

min_button = tk.Button(controls_frame, text="Min Filter", command=apply_min_filter)
min_button.pack(side=tk.LEFT, padx=5)

sharpen_button = tk.Button(controls_frame, text="Sharpen Image", command=sharpen_image)
sharpen_button.pack(side=tk.LEFT, padx=5)

unsharp_button = tk.Button(controls_frame, text="Unsharp Mask", command=unsharp_mask)
unsharp_button.pack(side=tk.LEFT, padx=5)

kmeans_button = tk.Button(controls_frame, text="KMeans Clustering", command=apply_kmeans_clustering)
kmeans_button.pack(side=tk.LEFT, padx=5)

canny_button = tk.Button(controls_frame, text="Canny Edge Detection", command=apply_canny_edge_detection)
canny_button.pack(side=tk.LEFT, padx=5)

color_button = tk.Button(controls_frame, text="Color Detection", command=apply_color_detection)
color_button.pack(side=tk.LEFT, padx=5)

log_button = tk.Button(controls_frame, text="Log Transform", command=apply_log_transform)
log_button.pack(side=tk.LEFT, padx=5)

start_video_button = tk.Button(controls_frame, text="Start Video Capture", command=start_video_capture)
start_video_button.pack(side=tk.LEFT, padx=5)

stop_video_button = tk.Button(controls_frame, text="Stop Video Capture", command=stop_video_capture)
stop_video_button.pack(side=tk.LEFT, padx=5)

# Controls for Gaussian filter
sigma_label = tk.Label(controls_frame, text="Sigma:")
sigma_label.pack(side=tk.LEFT)
sigma_entry = tk.Entry(controls_frame)
sigma_entry.pack(side=tk.LEFT)
sigma_entry.insert(0, '1.0')

# Controls for KMeans
kmeans_label = tk.Label(controls_frame, text="Number of Clusters:")
kmeans_label.pack(side=tk.LEFT)
kmeans_entry = tk.Entry(controls_frame)
kmeans_entry.pack(side=tk.LEFT)
kmeans_entry.insert(0, '2')

# Controls for Unsharp Mask
amount_label = tk.Label(controls_frame, text="Amount:")
amount_label.pack(side=tk.LEFT)
amount_entry = tk.Entry(controls_frame)
amount_entry.pack(side=tk.LEFT)
amount_entry.insert(0, '1.0')

# Controls for Median Filter
median_slider = tk.Scale(controls_frame, from_=1, to=10, orient=tk.HORIZONTAL, label="Median Filter Size")
median_slider.pack(side=tk.LEFT)

# Controls for Max/Min Filter
max_min_slider = tk.Scale(controls_frame, from_=1, to=10, orient=tk.HORIZONTAL, label="Filter Size")
max_min_slider.pack(side=tk.LEFT)

# Histogram frame
histogram_frame = tk.Frame(window)
histogram_frame.pack(side=tk.BOTTOM, fill=tk.X)

window.mainloop()
