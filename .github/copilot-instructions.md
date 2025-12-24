# Gluformer: Transformer-Based Glucose Forecasting

## Project Overview
Gluformer is a research implementation of a transformer model for personalized continuous glucose monitoring (CGM) forecasting with uncertainty quantification. The model predicts future glucose levels using historical CGM readings and temporal features.

**Paper**: [Gluformer: Transformer-Based Personalized Glucose Forecasting](https://arxiv.org/abs/2209.04526)

## Architecture & Key Components

### Data Flow Pipeline
1. **Data Loading** ([gludata/data_loader.py](gludata/data_loader.py)): `CGMData` class loads pickled time series segments
   - Each segment: `(subject_id, glucose_readings, temporal_marks)`
   - Handles variable-length patient segments with overlapping windows
   - Parameters: `seq_len` (lookback), `label_len` (decoder input), `pred_len` (forecast horizon)

2. **Sample Augmentation** ([utils/collate.py](utils/collate.py), [utils/train.py](utils/train.py)): 
   - `modify_collate()` replicates each batch sample `num_samples` times for mixture density training
   - Critical for uncertainty quantification via multi-sample posterior estimation

3. **Model Architecture** ([gluformer/model.py](gluformer/model.py)):
   - **Encoder**: Multi-layer transformer with optional distillation via `ConvLayer` (MaxPooling between layers)
   - **Decoder**: Transformer decoder with cross-attention to encoder output
   - **Variance Module** ([gluformer/variance.py](gluformer/variance.py)): Separate network predicts log-variance per subject (outputs in [-10, 10] range via Tanh)
   - **Embeddings** ([gluformer/embed.py](gluformer/embed.py)): Combines subject ID embedding with positional encoding

### Loss Function & Training
- **Mixture Loss** ([utils/train.py](utils/train.py)): `ExpLikeliLoss` computes log-sum-exp over replicated samples for mixture density modeling
  - Reshapes predictions: `[batch*num_samples, len_pred, 1]` → `[batch, len_pred, num_samples]`
  - Balances prediction accuracy with variance estimation
- **Alternative**: Standard MSE loss with `num_samples=1` for deterministic training
- **Data Scaling**: Raw glucose values (38-402 mg/dL) scaled to `[-scale, scale]` range (typically scale=5)

## Development Workflows

### Environment Setup
```bash
conda env create -f environment.yml
conda activate gluformer
```
**Note**: Python environment is in `gluformer/bin/` (not a package, but a conda env directory)

### Running Experiments
All commands run from repository root:

**Training**:
```bash
python model_train.py --trial_id "trial_1" --model_path "model_best_1.pth" \
  --num_samples 5 --epochs 100 --stop_epochs 10 --r_drop 0.2
```

**Evaluation**:
```bash
python model_eval.py --trial_id "trial_1" --num_samples 100 --r_drop 0.2
```

**Key Parameters**:
- `--num_samples`: Number of Monte Carlo samples (5 for training, 100 for eval)
- `--loss_name`: "mixture" (default) or "mse"
- `--len_seq`, `--len_label`, `--len_pred`: Sequence lengths (default: 180, 60, 12 timesteps)
- `--distil`: Enable distillation layers in encoder (default: True)

See [trials/trials.txt](trials/trials.txt) for full example commands.

### Data Preparation
- Pre-split data expected in `gludata/data/`: `train_data.pkl`, `val_data.pkl`, `test_data.pkl`
- Use [gludata/data/split.py](gludata/data/split.py) to split raw `UM_data.pkl` (CC BY-NC-SA license)
- Each pickle contains list of `(subject_id, np.array[glucose], np.array[temporal_features])`

## Project-Specific Conventions

### File Organization
- **Core model**: `gluformer/*.py` (attention, encoder, decoder, embed, variance, model)
- **Data utilities**: `gludata/*.py`
- **Training infrastructure**: `utils/*.py` (train loops, collate functions)
- **Experiment tracking**: `trials/{trial_id}/` stores models and logs
- **Notebook exploration**: [experiment.ipynb](experiment.ipynb) demonstrates synthetic data training

### Coding Patterns
- **Path handling**: All scripts use `os.getcwd()` as root and expect execution from repository root
- **Device management**: Scripts accept `--gpu_index` parameter; default to CPU if CUDA unavailable
- **Model saving**: Best model saved to `./trials/{trial_id}/model_best.pth` via `EarlyStop` callback
- **Batch processing**: `process_batch()` in [utils/train.py](utils/train.py) handles device placement and decoder input construction

### Decoder Input Construction
Critical pattern in `process_batch()`:
```python
dec_inp = torch.zeros([batch_y.shape[0], len_pred, batch_y.shape[-1]])
dec_inp = torch.cat([batch_y[:, :len_label, :], dec_inp], dim=1)
```
Concatenates `label_len` timesteps of ground truth with zero-padded prediction horizon.

## Important Notes
- **Subject ID Embedding**: Model uses learnable embeddings per subject for personalization
- **Temporal Encoding**: Both encoder and decoder receive temporal marks (e.g., time of day, day of week)
- **Uncertainty Quantification**: Variance network operates on encoder output before decoder input
- **Early Stopping**: Patience-based on validation loss, no delta threshold by default
