import numpy as np
import cv2
import os

def add_gaussian_noise(image_path, noise_std, output_path):
    # Read the image
    img = cv2.imread(image_path)
    
    if img is not None:
        # Add random noise to the image with specified standard deviation
        noise = np.random.normal(0, noise_std, img.shape)
        noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        # Save the noisy image
        cv2.imwrite(output_path, noisy_img)
        print(f"Noisy image saved: {output_path}")
    else:
        print(f"Failed to load image: {image_path}")

# Example usage
input_image = "dataset/00/01_palm/frame_00_01_0001.png"
output_image = "figures-output/gestureNoisy40.png"
noise_std = 40.0  # Specify the desired noise standard deviation

add_gaussian_noise(input_image, noise_std, output_image)