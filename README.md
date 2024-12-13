# Super-Resolution (SR) Microscopy Benchmarking Framework
This repository provides a Python-based implementation of a benchmarking framework for evaluating super-resolution (SR) microscopy techniques. The framework assesses both resolution enhancement and biological content preservation using a suite of image quality metrics and a pixel-wise reconstruction confidence map.

## Features
- **System-Agnostic**: Works with various SR techniques, including Enhanced Super-Resolution Radial Fluctuations (eSRRF), Structured Illumination Microscopy (SIM), and Airyscan.

- **Comprehensive Metrics**: Supports normalized mean squared error (nMSE), peak signal-to-noise ratio (PSNR), structural similarity index measure (SSIM), Pearson cross-correlation (PCC), and mutual information (MI).

- **Resolution Mapping**: Generates pixel-wise resolution confidence maps using Fourier Ring Correlation (FRC).

- **Synthetic and Experimental Data**: Compatible with both synthetic datasets and experimental SR microscopy images.

## Installation

1. Clone the repository:
  ```bash
  git clone https://github.com/yourusername/sr-microscopy-benchmark.git
  cd sr-microscopy-benchmark
  ```

2. Create a virtual environment:
    ```bash
    mamba create -n sr-metric
    mamba activate sr-metric
    # install the required dependencies
    pip install -r requirements.txt
    ```
## Usage
### Input Data Preparation
Prepare raw and reconstructed images from your SR microscopy system. Ensure images are formatted as standard .tif files or any other format supported by your image processing library.

### Running the Benchmark

Run the benchmarking script with your dataset:
```bash
python benchmark.py --raw path/to/raw_image --reconstructed path/to/reconstructed_image
```
Options include:
- `--metrics`: Specify metrics to calculate (default: all metrics).
- `--output`: Specify the output directory for results and plots.

### Results Interpretation

The framework generates:

- **Global Metrics**: Numerical results for nMSE, PSNR, SSIM, PCC, and MI.

- **Resolution Maps**: Visualization of pixel-wise reconstruction confidence.

- **Reports**: Summarized results in a `.csv` file for easy integration with other tools.

### Example Workflow

1. Prepare synthetic or experimental datasets with raw and reconstructed images.
2. Use the provided [benchmark.py](./benchmark.py) script to evaluate your SR method.
3. Analyze the output metrics and resolution maps to refine your SR technique or validate its biological relevance.

### Repository Structure
```
sr-microscopy-benchmark/
├── benchmark.py          # Main benchmarking script
├── requirements.txt      # Dependencies
├── data/                 # Example datasets
├── results/              # Output results and visualizations
├── src/                  # Source code for metrics and mapping
└── README.md             # Documentation
```

###Contact

For questions or support, please open an issue or contact [Jacky Ka Long Ko](mailto:ka.ko@kennedy.ox.ac.uk).
