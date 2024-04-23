import os
import pickle
import numpy as np
from PIL import Image, ImageDraw
import json


def overlay_images(background_image, overlay_image):
    """
    Overlay one image on top of another image.
    
    Parameters:
    - background_image_path: Path to the background image.
    - overlay_image_path: Path to the image to be overlaid.
   
    """
    background_image = background_image.convert('RGB')
    overlay_image = overlay_image.convert('RGB')
    # Resize overlay image to match the size of background image
    overlay_image = overlay_image.resize(background_image.size)
    
    # Create a new image to combine the two images
    combined_image = background_image.copy()
    
    # Paste overlay image onto the combined image
    combined_image.paste(overlay_image, (0, 0))
    return combined_image
    

def combine_images(image1, image2):
    # Get dimensions of the input images
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Calculate the width and height of the combined image
    combined_width = width1 + width2
    combined_height = max(height1, height2)

    # Create a new blank image with the combined dimensions
    combined_image = Image.new('RGB', (combined_width, combined_height))

    # Paste the first image onto the left side of the combined image
    combined_image.paste(image1, (0, 0))

    # Paste the second image onto the right side of the combined image
    combined_image.paste(image2, (width1, 0))

    return combined_image

def calculate_pixel_differences(image1, image2, threshold=50):
    """
    Calculate pixel differences between two images for individual RGB channels.
    Returns a list of dictionaries containing the coordinates of differing pixels and the differences.
    """
    if image1.size != image2.size or image1.mode != image2.mode:
        raise ValueError("Images must have the same size and mode.")

    width, height = image1.size
    differences = []

    for y in range(height):
        for x in range(width):
            pixel1 = image1.getpixel((x, y))
            pixel2 = image2.getpixel((x, y))
            
            # Calculate differences for individual color channels
            diff_r = abs(pixel1[0] - pixel2[0])
            diff_g = abs(pixel1[1] - pixel2[1])
            diff_b = abs(pixel1[2] - pixel2[2])
            
            # Check if any difference exceeds the threshold
            if diff_r > threshold or diff_g > threshold or diff_b > threshold:
                differences.append({"x": x, "y": y, "diff": (diff_r, diff_g, diff_b)})

    return differences


def highlight_regions(image, differences):
    """
    Highlight regions in the image where the pixel differences are significant.
    Returns a new image with highlighted regions.
    """
    # Create a copy of the input image
    highlighted_image = image.copy()
    
    # Draw on the copied image
    img_draw = ImageDraw.Draw(highlighted_image)
    for diff in differences:
        x, y = diff["x"], diff["y"]
        img_draw.rectangle([x, y, x + 1, y + 1], outline="red")
    
    return highlighted_image



def calculate_bpp(data_path, image_size):
    # Get the file size of the image in bytes
    file_size_bytes = os.path.getsize(data_path)

    # Convert file size to bits
    file_size_bits = file_size_bytes * 8

    width, height = image_size

    # Calculate total number of pixels
    total_pixels = width * height

    # Calculate bits per pixel
    bpp = file_size_bits / total_pixels

    return bpp

def calculate_psnr(original_image, reconstructed_image):
    # Convert images to numpy arrays
    original_array = np.array(original_image)
    reconstructed_array = np.array(reconstructed_image)

    # Calculate mean squared error
    mse = np.mean((original_array - reconstructed_array) ** 2)

    # Maximum possible pixel value
    max_pixel_value = 255  # Assuming 8-bit images

    # Calculate PSNR
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))

    return psnr

def mse_loss(image1, image2):
    """
    Calculate the Mean Squared Error loss between two images.

    Parameters:
    image1 (PIL.Image.Image): First image.
    image2 (PIL.Image.Image): Second image.

    Returns:
    float: Mean Squared Error between the two images.
    """
    # Convert images to numpy arrays
    arr1 = np.array(image1)
    arr2 = np.array(image2)
    
    # Check if the images have the same dimensions
    if arr1.shape != arr2.shape:
        raise ValueError("Images must have the same dimensions and number of channels")
    
    # Calculate MSE
    mse = np.mean((arr1 - arr2) ** 2)
    return mse