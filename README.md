# Text Classification Tutorial

This repository demonstrates three distinct methods for building text classification models, each with their own strengths and use cases.

*Originally created for a PyData Sofia few-shot text classification demo.*

## Overview

This project implements three different text classification approaches:

1. **TF-IDF + Logistic Regression** - Traditional feature-based approach
2. **SetFit** - Few-shot learning with sentence transformers
3. **Model2Vec** - Static embeddings for efficient classification

All models are trained on the same dataset and pushed to Hugging Face Hub for easy sharing and deployment.

## Dataset

The project uses a text classification dataset with the following labels:
- GENERATING COMMUNICATIVE TEXT
- INFORMATION SEARCH
- SOFTWARE DEVELOPMENT
- GENERATING CREATIVE TEXT
- HOMEWORK PROBLEM

Data is stored in Parquet format in the `data/` directory:
- `train.parquet` - Training data
- `eval.parquet` - Validation data
- `test.parquet` - Test data

## Quick Start

### Prerequisites

- Python 3.10+
- UV package manager (recommended) or pip

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd pydata-text-classification

# Install dependencies
uv sync
```

### Environment Setup

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Edit `.env` and add your credentials:
```bash
HF_TOKEN="your_huggingface_token"
HF_PROFILE_NAME="your_hf_username"
PROJECT_NAME="text-class-tutorial"
```

### Running All Training

Execute all three training methods in sequence:

```bash
python run_all_training.py
```

Or run individual training scripts:

```bash
# TF-IDF approach
python train_tfidf.py

# Model2Vec approach
python train_model2vec.py minishlab/potion-base-32M

# SetFit approach
python train_setfit.py BAAI/bge-small-en-v1.5
```

## Methods Comparison

### 1. TF-IDF + Logistic Regression (`train_tfidf.py`)

**Approach**: Traditional bag-of-words with TF-IDF weighting
- **Vectorizer**: TF-IDF with uni/bi-grams, English stop words removal
- **Classifier**: Logistic Regression
- **Deployment**: Pushed to HF Hub using `skops` library
- **Pros**: Fast training, interpretable, lightweight
- **Cons**: No semantic understanding, struggles with synonyms

### 2. SetFit (`train_setfit.py`)

**Approach**: Few-shot learning with sentence transformers
- **Base Model**: BAAI/bge-small-en-v1.5
- **Method**: Contrastive learning + classification head
- **Training**: 90 steps with evaluation every 5 steps
- **Deployment**: Native HF Hub integration
- **Pros**: Great few-shot performance, semantic understanding
- **Cons**: Requires more compute, larger model size

### 3. Model2Vec (`train_model2vec.py`)

**Approach**: Static embeddings with efficient classification
- **Base Model**: minishlab/potion-base-32M
- **Method**: Pre-computed static embeddings + classifier
- **Training**: Fast fitting on static representations
- **Deployment**: Pipeline format to HF Hub
- **Pros**: Very fast inference, good performance
- **Cons**: Static embeddings, no context adaptation

## Model Performance

All models report the following metrics:
- Accuracy
- Precision, Recall, F1-score (macro and micro averages)
- Training time
- Inference time

Metrics are saved locally in the `models/` directory and included in model cards on Hugging Face Hub.

## Project Structure

```
├── data/                     # Dataset files
├── models/                   # Local model outputs
├── __marimo__/               # Marimo notebook cache
├── train_tfidf.py           # TF-IDF training script
├── train_setfit.py          # SetFit training script
├── train_model2vec.py       # Model2Vec training script
├── run_all_training.py      # Sequential training runner
├── settings.py              # Project configuration
├── utils.py                 # Utility functions
├── pyproject.toml           # Dependencies
├── Makefile                 # Linting and formatting
└── README.md                # This file
```

## Development

### Code Quality

Run linting and formatting:

```bash
# Run both check and format
make

# Or individually
make lint    # Run ruff check
make format  # Run ruff format
```

### Model Deployment

All trained models are automatically pushed to Hugging Face Hub with the naming convention:
- `{HF_PROFILE_NAME}/{PROJECT_NAME}-tfidf`
- `{HF_PROFILE_NAME}/{PROJECT_NAME}-setfit`
- `{HF_PROFILE_NAME}/{PROJECT_NAME}-model2vec`

## Dependencies

Key libraries used:
- `datasets` - Data loading and processing
- `scikit-learn` - TF-IDF and traditional ML
- `setfit` - Few-shot learning framework
- `model2vec` - Static embedding models
- `skops` - Scikit-learn model deployment
- `huggingface_hub` - Model sharing and deployment
- `typer` - CLI interface

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting: `make`
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Links

- [Hugging Face Models](https://huggingface.co/krumeto)
- [SetFit Documentation](https://huggingface.co/docs/setfit)
- [Model2Vec Documentation](https://github.com/MinishLab/model2vec)
- [Skops Documentation](https://skops.readthedocs.io/)

---
