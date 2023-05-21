# Papyrus Unroll Analysis

## Introduction

Welcome to the Papyrus Unroll Analysis, a non-AI technique for the analysis and deciphering of ancient papyrus scrolls.

This is proposal leverages deterministic algorithms to stack 2D TIFF images into a 3D volume, emphasize features such as ink or text, and facilitate further in-depth investigation. The authenticity of the results is preserved by minimizing the risk of 'hallucinations'.



## Requirements

- Python 3.7 or later
- NumPy
- tifffile
- Matplotlib
- Pillow
- IPython

## Usage

```bash
python volume_processor.py <input_folder> <output_folder> [-t <opacity_threshold>] [-s <skew_factor>] [-g <gradient_type>] [-r <slice_range>]
```

- `input_folder`: Path to the folder containing the input TIFF files. This argument is required.
- `output_folder`: Path to the folder where the output TIFF files will be saved. This argument is required.
- `opacity_threshold`: The opacity threshold to apply to the volume. The default value is 0.4.
- `skew_factor`: The skew factor to adjust the gradient. The default value is 1.0.
- `gradient_type`: The type of gradient to apply. Options are 'linear', 'quadratic', 'logarithmic', 'parabolic'. The default value is 'linear'.
- `slice_range`: The range of slices to process. Should be specified as a pair of integers. If not provided, the entire volume is processed.

## Example

```bash
python volume_processor.py ./input ./output -t 0.5 -s 1.2 -g logarithmic -r 10 20
```

This command will process the TIFF files in the `./input` directory and save the processed slices in the `./output` directory. It will apply a logarithmic gradient with a skew factor of 1.2 and an opacity threshold of 0.5. It will only process slices 10 through 20.

## Important Note

This program generates several plots for the purpose of visualizing the processed data. In order to proceed with the program, these plots must be manually closed.

## License

This project is licensed under the terms of the GPL3 license.
