# Gluformer Project Context

## Project Overview
**Gluformer** is a Transformer-based model designed for personalized glucose forecasting with uncertainty quantification. It is the official implementation of the paper "Gluformer: Transformer-Based Personalized Glucose Forecasting with Uncertainty Quantification" (arXiv:2209.04526).

The model uses an Encoder-Decoder architecture to predict future glucose levels based on historical Continuous Glucose Monitoring (CGM) data. It includes a specific mechanism for estimating prediction variance (uncertainty).

## Environment Setup
The project uses `conda` for dependency management.

1.  **Create Environment:**
    ```bash
    conda env create -f environment.yml
    ```
2.  **Activate Environment:**
    ```bash
    conda activate gluformer
    ```

**Key Dependencies:**
*   Python 3.8
*   PyTorch 1.10.1
*   NumPy, Pandas, Matplotlib, SciPy
*   Click (for CLI)

## File Structure
*   `gluformer/`: Core model implementation (Encoder, Decoder, Attention, etc.).
    *   `model.py`: Main `Gluformer` class definition.
*   `gludata/`: Data handling.
    *   `data_loader.py`: PyTorch Dataset implementation (`CGMData`).
    *   `data/`: Contains scripts for data splitting (`split.py`) and visualization (`view.ipynb`).
*   `utils/`: Helper functions for training and evaluation.
*   `trials/`: Output directory for experiment logs and model checkpoints.
*   `model_train.py`: Main script for training the model.
*   `model_eval.py`: Main script for evaluating the model.
*   `experiment.ipynb`: Jupyter notebook for experimenting with synthetic data.

## Usage

### Training
To train the model, use `model_train.py`. The script uses `click` for argument parsing.

**Example Command:**
```bash
python model_train.py \
    --trial_id "trial_1" \
    --model_path "model_best_1.pth" \
    --num_samples 5 \
    --epochs 100 \
    --stop_epochs 10 \
    --r_drop 0.2
```

**Common Arguments:**
*   `--trial_id`: Identifier for the experiment (creates a folder in `trials/`).
*   `--gpu_index`: GPU ID to use (default: 0).
*   `--loss_name`: Loss function ("mixture" or "mse").
*   `--batch_size`: Batch size (default: 32).
*   `--len_pred`: Prediction horizon length.

### Evaluation
To evaluate a trained model, use `model_eval.py`.

**Example Command:**
```bash
python model_eval.py \
    --trial_id "trial_1" \
    --num_samples 100 \
    --r_drop 0.2
```

## Development Notes
*   **Model Logic:** The core transformer logic is in `gluformer/model.py`. It combines `DataEmbedding`, `Encoder`, `Decoder`, and `Variance` modules.
*   **Data Format:** The system expects data to be in specific pickle files (`train_data.pkl`, etc.) within `gludata/data/` or `UM_data.pkl`.
*   **Uncertainty:** The model predicts a distribution (mean and variance) when using the "mixture" loss.
