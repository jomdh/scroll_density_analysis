import numpy as np
import tifffile
from pathlib import Path
from skimage import io, transform
from PIL import Image
import os

# set the relative path to your folder of TIFF files
input_folder = Path("data\layer_82749")
output_folder = Path("data\layer_82749_density")
output_folder.mkdir(parents=True, exist_ok=True)


# Small constant to add before taking log to avoid zero intensity pixels
log_shift = 0.1

# Gradient functions
def linear_gradient(num_files):
    return np.linspace(1, 0, num_files)

def exponential_gradient(num_files, exp_scale=10):
    return np.exp(-np.linspace(0, exp_scale, num_files))

def parabolic_gradient(num_files):
    return (np.linspace(-1, 1, num_files))**2

# Default gradient is linear
default_gradient = linear_gradient



def density_plot(input_folder, chunk_size=(1000, 1000), gradient=default_gradient, output_filename='density_z.png'):
    # Check if the input_folder exists and is a directory
    input_folder = Path(input_folder)
    if not input_folder.exists() or not input_folder.is_dir():
        raise ValueError(f"{input_folder} does not exist or is not a directory.")

    # Get a sorted list of the TIFF files in the input_folder
    tiff_files = sorted(input_folder.glob("*.tif"))

    if not tiff_files:
        raise ValueError(f"No TIFF files found in {input_folder}.")

    with tifffile.TiffFile(tiff_files[0]) as tif:
        height, width = tif.asarray().shape
        dtype = tif.asarray().dtype

    num_files = len(tiff_files)

    # Calculate the number of chunks
    num_chunks_y = height // chunk_size[0]
    num_chunks_x = width // chunk_size[1]

    # Process each chunk
    for chunk_y in range(num_chunks_y):
        for chunk_x in range(num_chunks_x):
            print(f"Processing chunk {chunk_y} {chunk_x}...")
            # Calculate the chunk boundaries
            y_start = chunk_y * chunk_size[0]
            y_end = y_start + chunk_size[0]
            x_start = chunk_x * chunk_size[1]
            x_end = x_start + chunk_size[1]

            volume = np.empty((chunk_size[0], chunk_size[1], num_files), dtype=dtype)

            for i, file in enumerate(tiff_files):
                with tifffile.TiffFile(file) as tif:
                    # Load the chunk from the TIFF file
                    volume[:, :, i] = tif.asarray()[y_start:y_end, x_start:x_end]

            # Apply the gradient to the volume
            z_gradient = gradient(num_files)

            volume = volume * z_gradient[None, None, :]

            # Compute the sum of the adjusted density values along the Z axis
            density_z_adj = np.sum(volume, axis=2)

            # Apply a logarithmic scale to the adjusted density values
            density_z_log_adj = np.log10(np.where(density_z_adj > 0, density_z_adj, np.nan) + log_shift)

            if not np.isnan(density_z_log_adj).all():
                # Normalize density_z_log_adj for image representation
                density_z_log_adj_normalized = (density_z_log_adj - np.nanmin(density_z_log_adj)) / (np.nanmax(density_z_log_adj) - np.nanmin(density_z_log_adj))
                density_z_log_adj_normalized *= 255.0
            else:
                print(f"All values are NaN for chunk {chunk_y} {chunk_x}")
                # Create a black image for chunks where all values are NaN
                density_z_log_adj_normalized = np.zeros((chunk_size[0], chunk_size[1]))

            # Create a PIL image from the numpy array
            img = Image.fromarray(density_z_log_adj_normalized.astype(np.uint8), 'L')

            # Save the image as a PNG file in the parent folder of the input folder
            chunk_filename = f"{output_filename.rsplit('.', 1)[0]}_chunk_{chunk_y}_{chunk_x}.png"
            Path(output_folder / gradient.__name__).mkdir(parents=True, exist_ok=True)
            img.save( output_folder / gradient.__name__ / chunk_filename)


def stitch_images(input_folder, output_filename='density_z.png'):
    input_folder = Path(input_folder)
    if not input_folder.exists() or not input_folder.is_dir():
        raise ValueError(f"{input_folder} does not exist or is not a directory.")

    # List all png files in the input_folder
    image_files = sorted(input_folder.glob("*.png"))

    if not image_files:
        raise ValueError(f"No image files found in {input_folder}.")

    # Calculate the number of chunks
    num_chunks_y = len(set([int(file.stem.split("_chunk_")[1].split("_")[0]) for file in image_files]))
    num_chunks_x = len(set([int(file.stem.split("_chunk_")[1].split("_")[1]) for file in image_files]))

    # Load all images
    images = [Image.open(file) for file in image_files]

    # Check if all images have the same size
    if len(set(img.size for img in images)) > 1:
        raise ValueError("Not all images have the same size.")

    # Create an empty image of the right size
    width, height = images[0].size
    stitched_image = Image.new('L', (width * num_chunks_x, height * num_chunks_y))

    # Paste the images
    for i, img in enumerate(images):
        x = i % num_chunks_x
        y = i // num_chunks_x
        stitched_image.paste(img, (width * x, height * y))

    # Save the stitched image
    stitched_image.save(input_folder.parent/ output_filename)


if __name__ == "__main__":
    density_plot(input_folder, chunk_size=(300, 300), gradient=linear_gradient, output_filename='density_z_linear.png')
    # Path(output_folder / 'linear').mkdir(parents=True, exist_ok=True)
    stitch_images(output_folder / 'linear_gradient', output_filename='density_z_linear.png')
    # density_plot(input_folder, chunk_size=(1000, 1000), gradient=parabolic_gradient, output_filename='density_z_parabolic.png')
    # stitch_images(output_folder / 'parabolic_gradient', output_filename='density_z_parabolic.png')
    # # density_plot(input_folder, chunk_size=(1000, 1000), gradient=exponential_gradient, output_filename='density_z_exp.png')
    # # stitch_images(output_folder / 'exponential_gradient', output_filename='density_z_exp.png')