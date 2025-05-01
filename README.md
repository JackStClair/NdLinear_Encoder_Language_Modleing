# Transformer Language Model with NdLinear

This project implements a Transformer-based language model using NdLinear layers for efficient tensor transformations. The model is trained on the Penn Treebank dataset and includes support for pre-trained GloVe embeddings.

## Requirements

- Python 3.12
- CUDA-capable GPU (recommended)
- pip package manager

## Installation

1. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install PyTorch with CUDA support (if using GPU):
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

3. Install remaining requirements:
```bash
pip install -r requirements.txt
```

## Usage

The model can be run with the following command:

```bash
python run.py output_file.csv
```

Where `output_file.csv` is the path where the perplexity results will be saved.

### Model Parameters

The model can be configured by modifying the following parameters in `run_model.py`:

- `DIMENSION`: Model dimension (default: 512)
- `BATCH_SIZE`: Training batch size (default: 32)
- `LEARNING_RATE`: Learning rate for optimization (default: 0.0001)
- `NUM_EPOCHS`: Number of training epochs (default: 30)
- `NUM_HEADS`: Number of attention heads (default: 16)
- `NUM_LAYERS`: Number of transformer layers (default: 4)
- `DROPOUT`: Dropout rate (default: 0.3)
- `EMBEDDING_DIM`: Word embedding dimension (default: 100)

### GPU Usage

The model automatically uses CUDA if available. To specify a particular GPU, you can set the `CUDA_VISIBLE_DEVICES` environment variable:

```bash
CUDA_VISIBLE_DEVICES=4 python run.py output_file.csv
```

## Output

The script outputs:
- Training progress including loss and perplexity metrics
- Final perplexity scores for the test set
- Results saved to the specified CSV file

## Features

- Transformer architecture with NdLinear layers
- Pre-trained GloVe embeddings integration
- Learned positional encoding
- Automatic GPU utilization
- Progress tracking with tqdm
- Comprehensive metrics logging
