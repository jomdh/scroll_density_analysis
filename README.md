# Papyrus Differential Analysis


Welcome to the Papyrus Differential Analysis, a non-AI technique for the analysis of scrolls.

This is proposal leverages deterministic algorithms to stack 2D TIFF images into a 3D volume, emphasize the ink, and facilitate further in-depth investigation. The authenticity of the results is preserved by minimizing the risk of 'hallucinations', since no ML technique is involved.

# Content of the Repo:

 - Notebook guiding through the reasoning of the analysis.
 - Notebook with a POC for a graphical tool.
 - A command line tool (see below).

I'm working on a stand alone version, but it's in very early stage atm.




## Requirements

- Python 3.7 or later
- NumPy
- tifffile
- Matplotlib
- Pillow
- IPython

## Usage

```bash
python diffanalisys_v.0.3beta.py <input_folder> <output_folder> [-t <opacity_threshold>] [-s <skew_factor>] [-g <gradient_type>] [-r <slice_range>]
```

- `input_folder`: Path to the folder containing the input TIFF files. This argument is required.
- `output_folder`: Path to the folder where the output TIFF files will be saved. This argument is required.
- `opacity_threshold`: The opacity threshold to apply to the volume. The default value is 0.4.
- `skew_factor`: The skew factor to adjust the gradient. The default value is 1.0.
- `gradient_type`: The type of gradient to apply. Options are 'linear', 'quadratic', 'logarithmic', 'parabolic'. The default value is 'linear'.
- `slice_range`: The range of slices to process. Should be specified as a pair of integers. If not provided, the entire volume is processed.

## Example

```bash
python diffanalisys_v.0.3beta.py ./input ./output -t 0.5 -s 1.2 -g logarithmic -r 10 20
```

This command will process the TIFF files in the `./input` directory and save the processed slices in the `./output` directory. It will apply a logarithmic gradient with a skew factor of 1.2 and an opacity threshold of 0.5. It will only process slices 10 through 20.


## License

This project is licensed under the terms of the GPL3 license.
