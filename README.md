

This repository hosts a Python script to create density maps from a series of 2D TIFF images representing slices of a 3D volume, such as the X-ray scans of unopened Herculaneum scrolls in the Vesuvius Challenge. The density maps visualize the internal structure of the volume, potentially helping to identify features like text within the scrolls.

Required Python libraries:

numpy tifffile pathlib skimage pillow

Usage

The main script is density_plot.py. It applies a gradient function to the pixel intensities across the Z-axis of the volume, then sums them up to create a 2D image representing the 3D density of the volume.

By default, the script processes TIFF images in the input directory and outputs density maps in the output directory. 

You can modify the input_folder and output_folder variables in the script to change these directories, or just import as a lybrary and custom your function.


The script applies three gradient functions, dimming the core of the papyri, and enhancing the faces/edges:

    linear_gradient: Weights slices from top (1) to bottom (0).
    exponential_gradient: Weights slices from top (high) to bottom (low) exponentially.
    parabolic_gradient: Weights slices highest at the top and bottom, lowest in the middle.

Contributing

Contributions are welcome. Please open an issue to discuss your ideas or submit a pull request.
License

This project is licensed under the MIT License. But if you get a price, be so kind of paying a beer :)

Acknowledgements

This project relies in data part of the Vesuvius Challenge to read unopened Herculaneum scrolls.