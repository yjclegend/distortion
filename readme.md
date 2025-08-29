# Distortion Correction: Reproduction Guide

This repository provides tools and experiments for camera distortion correction using various regression-based models. The codebase includes synthetic experiments, model training, and image correction scripts.

## Setup

1. **Clone the repository**

```bash
git clone <repo-url>
cd distortion
```

2. **Install dependencies**

Install all required Python libraries using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Synthetic Experiments

The script `experiments/model_robustness.py` contains several synthetic experiments to evaluate model robustness. To run a specific experiment, uncomment the corresponding function call in the `__main__` section at the bottom of the file. For example:

```python
if __name__ == "__main__":
    tb = DC_TestBench()
    tb.case_k()  # Uncomment to run the 'k' parameter experiment
```

Then run:

```bash
python experiments/model_robustness.py
```

## Model Estimation (RFM)

To estimate a distortion correction model (RFM), use a sample image from `data/`. The model parameters will be saved to `saved_model/` by default. To estimate the reverse model, set `reverse=True` in the training function.

Example usage is provided in `distortion_correction/metric/regression_dc.py` under the `train_decoupled` function.

## Image Correction

- **Forward Correction:**
  - Use `experiments/correct_forward.py` to correct distortion in the forward direction.
  - The script loads a pre-trained model and applies correction to a sample image. The output is saved as `image_output.png`.

- **Backward Correction (Inverse):**
  - Use `experiments/correct_with_interp.py` to apply inverse correction.
  - The script loads a reverse model and saves the corrected image as `image_output_inverse.png`.

## Directory Structure

- `data/` — Sample images for training and testing
- `saved_model/` — Saved model parameters
- `experiments/` — Scripts for experiments and image correction
- `distortion_correction/` — Core model and metric implementations
- `requirements.txt` — Python dependencies

## Notes

- Ensure all dependencies are installed before running any scripts.
- For custom experiments, modify or extend the provided scripts as needed.
- Results and models are saved in the respective output directories by default.

---

For any issues or questions, please refer to the code comments or open an issue in the repository.
