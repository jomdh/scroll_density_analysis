
import os
import argparse
from pathlib import Path
import tifffile
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def load_tiff_volume(input_folder):
    # Logic to load tiff files in a directory into a 3D numpy array
    
    # List all TIFF files in the directory
    tiff_files = sorted(input_folder.glob("*.tif"))
    
    # Read the first file to get the shape and dtype
    with tifffile.TiffFile(tiff_files[0]) as tif:
        sample = tif.asarray()
        if len(sample.shape) == 2:
            height, width = sample.shape
            channels = 1
        else:
            height, width, channels = sample.shape

    num_files = len(tiff_files)

    # Create an empty volume array
    volume = np.empty((height, width, num_files, channels), dtype=sample.dtype)

    # Load each TIFF file into the volume array
    for i, file in enumerate(tiff_files):
        with tifffile.TiffFile(file) as tif:
            image_data = tif.asarray()
            # If the image data is 2D, we add an extra dimension to make it 3D
            if len(image_data.shape) == 2:
                image_data = image_data[..., np.newaxis]
            volume[:, :, i, :] = image_data
    return volume

def calculate_average_density(volume, slice_range=None):
    # Logic to calculate average density over a range of slices
    # If no slice range is specified, use the entire volume
    if slice_range is None:
        slice_range = slice(0, volume.shape[2])

    # Select the slices in the specified range
    volume_slice = volume[:, :, slice_range]

    # Calculate the average density for each point over the Z axis
    average_density = np.mean(volume_slice, axis=2)

    return average_density

def differential_analysis(volume):
    # Logic to calculate the density difference between adjacent slices
    num_slices = volume.shape[2]

    # Create an empty volume array for the density differences
    # The number of slices is one less than the original volume
    volume_diff = np.empty((volume.shape[0], volume.shape[1], num_slices - 1), dtype=volume.dtype)

    # Calculate the change in density for each pair of slices
    for i in range(1, num_slices):
        # Get the current and previous slice
        current_slice = np.squeeze(volume[:, :, i])
        prev_slice = np.squeeze(volume[:, :, i - 1])

        # Calculate the change in density from the previous slice
        delta_density = current_slice - prev_slice

        # Add the absolute value of the change in density to the volume_diff
        volume_diff[:, :, i - 1] = np.abs(delta_density)
    return volume_diff

def plot_image(image, title=None, xlabel=None, ylabel=None, cmap='gray'):
    # Logic to plot an image
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

def plot_histogram(data1, data2, title1, title2, xlabel, ylabel):
    # Logic to plot histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot slide sums histogram
    ax1.bar(range(len(data1)), data1, color='blue', alpha=0.7)
    ax1.set_title(title1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.grid(True)
    
    # Plot z-scores
    ax2.bar(range(len(data2)), data2, color='blue', alpha=0.7)
    ax2.axhline(0, color='red', linestyle='--', label='Mean Value')
    ax2.set_title(title2)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Z-Score')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def calculate_slide_data(volume):
    # Logic to calculate slide data for histogram
    slide_sums = np.sum(volume, axis=(0, 1))
    z_scores = (slide_sums - np.mean(slide_sums)) / np.std(slide_sums)
    return slide_sums, z_scores

def apply_gradient(image, skew=1.0, gradient_type='linear'):
    """
    Apply a specified gradient to an image with different modes.

    Returns:
    numpy array: The image after applying the gradient.
    """
    # Normalize the image
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Apply the specified gradient
    if gradient_type == 'linear':
        skewed_image = normalized_image ** skew
    elif gradient_type == 'quadratic':
        skewed_image = normalized_image ** (2 * skew)
    elif gradient_type == 'logarithmic':
        skewed_image = np.log1p(normalized_image) ** skew
    elif gradient_type == 'parabolic':
        skewed_image = (1 - (1 - normalized_image) ** 2) ** skew
    else:
        raise ValueError("Invalid gradient type. Options are 'linear', 'quadratic', 'logarithmic', 'parabolic'.")

    # Map the skewed image values to the range 0-255
    return np.interp(skewed_image, (np.min(skewed_image), np.max(skewed_image)), (0, 255))


def process_volume(volume, gradient_type='linear', opacity_threshold=0.0, skew=1.0):
    """
    Process a 3D volume using a specified gradient.

    Parameters:
    volume (numpy array): The input 3D volume.
    gradient_type (str): The type of gradient to apply. Options are 'linear', 'quadratic', 'logarithmic', 'parabolic'.
    opacity_threshold (float): The opacity threshold to apply to the volume.
    skew (float): The skew factor to adjust the gradient.

    Returns:
    numpy array: The processed 3D volume.
    """
    # Logic to process the volume using the apply_gradient function
    num_slices = volume.shape[2]
    processed_slices = []

    for i in range(num_slices - 1):
        # Get the slice image
        slice_image = volume[:, :, i]

        # Apply the opacity threshold
        slice_image[slice_image <= opacity_threshold] = 0.0

        # Apply the gradient function with the skew value
        processed_slice = apply_gradient(slice_image, skew=skew, gradient_type=gradient_type)

        # Append the processed slice to the list
        processed_slices.append(processed_slice)

    # Stack the processed slices back together to form a new volume
    processed_volume = np.dstack(processed_slices)
    return processed_volume

def save_slices_as_tiff(volume, output_folder):
    # Logic to save the processed volume as TIFF files
    num_slices = volume.shape[2]
    for i in range(num_slices):
        # Get the slice
        slice_image = volume[:, :, i].astype(np.float32)

        # Normalize the image to the range [0, 255]
        normalized_slice = ((slice_image - np.min(slice_image)) / (np.max(slice_image) - np.min(slice_image)) * 255).astype(np.uint8)

        # Convert the slice to an RGBA image where the alpha channel matches the color
        rgba_image = np.zeros((slice_image.shape[0], slice_image.shape[1], 4), dtype=np.uint8)
        rgba_image[:, :, 0] = normalized_slice  # R
        rgba_image[:, :, 1] = normalized_slice  # G
        rgba_image[:, :, 2] = normalized_slice  # B
        rgba_image[:, :, 3] = normalized_slice  # A

        # Save the RGBA image as a TIFF file
        filename = output_folder / f"slice_{i:04d}.tif"
        tifffile.imwrite(filename, rgba_image)


    # export the volume as a NIFTI file

################################
# Main pipeline
def main():

    # Parser for command line arguments
    # Create the parser
    parser = argparse.ArgumentParser(description="Process a 3D volume")

    # Add arguments
    parser.add_argument('input', type=str, help='Input folder')  # Positional argument for input folder
    parser.add_argument('output', type=str, help='Output folder')  # Positional argument for output folder
    parser.add_argument('-t', '--threshold', type=float, default=0.4, help='Opacity threshold')
    parser.add_argument('-s', '--skew', type=float, default=1.0, help='Skew factor')
    parser.add_argument('-g', '--gradient', choices=['linear', 'quadratic', 'logarithmic', 'parabolic'], default='linear', help='Gradient type')
    parser.add_argument('-r', '--slice_range', nargs=2, type=int, help='Slice range')
 
    # Parse the arguments
    args = parser.parse_args()

    ### Input Validation ###
    # Check if input folder exists
    input_folder = Path(args.input)
    if not input_folder.exists():
        parser.error(f"The input folder {args.input} does not exist.")

    # Check if output folder exists
    output_folder = Path(args.output)
    if not output_folder.exists():
        # parser.error(f"The output folder {args.output} does not exist.")
        output_folder.mkdir(parents=True, exist_ok=True)

    # Check if the slice range is valid
    if args.slice_range and (args.slice_range[0] >= args.slice_range[1] or args.slice_range[0] < 0):
        parser.error("The slice range is invalid.")

    # Check if the skew and threshold values are valid
    if args.threshold < 0:
        parser.error("The threshold must be non-negative.")

    # Check if threshold is between 0 and 1
    if args.threshold > 1 or args.threshold < 0:
        parser.error("The threshold must be between 0 and 1.")

    # Check if gradient type is valid
    if args.gradient not in ['linear', 'quadratic', 'logarithmic', 'parabolic']:
        parser.error("The gradient type is invalid.")

    ### Main Pipeline ###
    # Convert the input folder to a Path object
    input_folder = Path(args.input)

    # Convert the output folder to a Path object
    output_folder = Path(args.output)
    # Create the output folder if it does not exist
    output_folder.mkdir(exist_ok=True, parents=True)


    # Load the volume from the input folder
    print("Loading volume...")
    volume = load_tiff_volume(input_folder)
    print("Volume loaded.")

    # Calculate the average density of the volume
    print("Calculating preview map...")
    preview_volume= calculate_average_density(volume)
    # Plot the preview volume as a 2D image
    plot_image(preview_volume, 'Preview: Average Density of the Volume')
    print('close the plot to continue')


    # perform the differential analysis
    print("Performing differential analysis...")
    volume = differential_analysis(volume)
    print("Differential analysis complete.")
    # NOTE, from here on, the volume not the original one,
    # this is a map of CHANGE in density, not the density itself   


    # Values for skew, threshold and gradient method
    skew_value = args.skew
    opacity = args.threshold
    gradient_type = args.gradient

    # Process the volume 
    print("Processing volume...")
    volume = process_volume(volume, gradient_type, opacity_threshold=opacity ,skew=skew_value)
    print("Volume processed.")


    # Calculate histogram and z-scores for each slide in the volume
    slide_sums, z_scores = calculate_slide_data(volume)

    # Plot slide histogram and z-scores side by side
    plot_histogram(slide_sums, z_scores, 'Density per Slide', 'Z-Score of Density in Each Slide', 'Slide Number', 'Sum of Pixel Values')


    # If no range is specified, set it to the full range; otherwise, use the specified range
    if args.slice_range:
        slice_range = args.slice_range
    else:
        slice_range = [0, volume.shape[2] - 1]

    slice_range = range(slice_range[0], slice_range[1])

    print(f"Slice range: {slice_range}")


    # Calculate and display the average density over the specified range
    print( "Calculating average density of the differential volume...")
    avg_density = calculate_average_density(volume, slice_range)
    plot_image(avg_density, 'Average Density', 'Average Density')
    print('close the plot to continue')

    # Save the average density as a TIFF file
    # If  image is in float format
    if avg_density.dtype.kind == 'f':
        avg_density = (255 * (avg_density - np.min(avg_density)) / np.ptp(avg_density)).astype(np.uint8)

    # Create an Image object from the NumPy array
    img = Image.fromarray(avg_density)

    # Save the image as a TIFF file
    img.save("%s_%s_%s_%s_%s_%s.tiff" % (output_folder, slice_range.start, slice_range.stop, args.gradient, skew_value, opacity ))


    # # Save the slices from the processed volume as TIFF files
    # print("Saving slices...")
    # save_slices_as_tiff(volume, output_folder)
    # print("Slices saved.")

    # import pickle
    # # save the volume as a pickle file
    # with open(output_folder / 'volume.pkl', 'wb') as f:
    #     pickle.dump(volume, f)
    
    # # save the average density as a pickle file
    # with open(output_folder / 'avg_density.pkl', 'wb') as f:
    #     pickle.dump(avg_density, f)



if __name__ == '__main__':
    main()
