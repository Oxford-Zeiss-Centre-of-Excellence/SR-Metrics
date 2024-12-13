import argparse
import os
import numpy as np
from skimage import io
from metrics import calculate_metrics
from resolution_map import generate_resolution_map


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking framework for super-resolution microscopy.")
    parser.add_argument("--raw", type=str, required=True, help="Path to the raw image.")
    parser.add_argument("--reconstructed", type=str, required=True, help="Path to the reconstructed image.")
    parser.add_argument(
        "--metrics", type=str, nargs="*", default=["nMSE", "PSNR", "SSIM", "PCC", "MI"],
        help="List of metrics to calculate. Default: all metrics."
    )
    parser.add_argument("--output", type=str, default="results", help="Directory to save the results and visualizations.")
    return parser.parse_args()


def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at {path}")
    return io.imread(path)


def save_results(metrics_results, resolution_map, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save metrics results
    metrics_path = os.path.join(output_dir, "metrics_results.txt")
    with open(metrics_path, "w") as f:
        for metric, value in metrics_results.items():
            f.write(f"{metric}: {value}\n")

    # Save resolution map
    resolution_map_path = os.path.join(output_dir, "resolution_map.tif")
    io.imsave(resolution_map_path, (resolution_map * 255).astype(np.uint8))

    print(f"Results saved in {output_dir}")


def main():
    args = parse_arguments()

    # Load images
    print("Loading images...")
    raw_image = load_image(args.raw)
    reconstructed_image = load_image(args.reconstructed)

    # Calculate metrics
    print("Calculating metrics...")
    metrics_results = calculate_metrics(raw_image, reconstructed_image, metrics=args.metrics)

    # Generate resolution map
    print("Generating resolution map...")
    resolution_map = generate_resolution_map(raw_image, reconstructed_image)

    # Save results
    print("Saving results...")
    save_results(metrics_results, resolution_map, args.output)


if __name__ == "__main__":
    main()
