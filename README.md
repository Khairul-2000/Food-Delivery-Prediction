# PyTorch Delivery Time Predictor

A small PyTorch project that trains a simple feed-forward neural network to predict delivery time (minutes) from delivery distance (miles). The trained weights and normalization statistics are saved to `model_bundle.pth`, and a separate script loads the bundle to make a prediction and print a simple decision.

## What’s in this repo

- `model.py` — trains the model on a small dataset (currently hard-coded in the script), live-plots training, and saves `model_bundle.pth`.
- `test_model.py` — loads `model_bundle.pth`, predicts delivery time for a single distance, de-normalizes the output, and prints a decision (under/over 45 minutes + bike vs car).
- `helper_utils.py` — plotting helpers used during training.
- `C1_M1_Lab_3_tensors.ipynb` — notebook work related to tensors.
- `data.csv` — present in the repo, but not currently used by the training script.
- `check.py` — small scratch script for quick tensor shape checks.

## Setup

### Option A: Use the existing `venv/` (if it’s already created)

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Option B: Create a new virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Notes:
- `requirements.txt` includes CUDA-related packages (the `nvidia-*` entries). If you’re on CPU-only or your system doesn’t support CUDA 12, you may want to install a CPU-only build of PyTorch instead of relying on those entries.

## Train the model

Run:

```bash
python model.py
```

This will:
- standardize (z-score) distances and times
- train a small network (`Linear(1→3) → ReLU → Linear(3→1)`) for 3000 epochs
- plot training progress periodically (requires a working matplotlib display backend)
- save the trained model + normalization stats to `model_bundle.pth`

## Make a prediction (load saved model)

Run:

```bash
python test_model.py
```

By default, `test_model.py` predicts for:

```python
distance_to_predict = 5.1
```

Edit that value to try different distances.

The script prints:
- predicted delivery time in minutes
- whether it can be promised under 45 minutes
- which vehicle to use (bike if `<= 3` miles, else car)

## Jupyter notebook

If you want to run the notebook:

```bash
jupyter notebook
```

Then open `C1_M1_Lab_3_tensors.ipynb`.

## Troubleshooting

- If plots don’t show up when running `model.py`, install/enable an appropriate matplotlib backend for your environment (or run inside Jupyter).
- If `torch.load("model_bundle.pth")` fails, retrain with `python model.py` to regenerate the bundle.

## License

No license file is included. Add one if you plan to share or reuse this project broadly.
