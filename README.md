
# AI Text Summarization with Transformers

This project is an implementation of a text summarization system using transformer-based models, specifically BART and T5. The project includes data preprocessing, model training, evaluation, and a simple user interface for summarizing text.

## Project Overview

Text summarization is a key task in Natural Language Processing (NLP), allowing large bodies of text to be condensed into more manageable summaries while retaining essential information. This project leverages modern transformer architectures to build a state-of-the-art summarization model.

## Features
- Transformer-based model (T5) for text summarization.
- Model training with progress tracking.
- Evaluation using ROUGE and BLEU scores.
- Simple user interface for interacting with the model (Flask backend).

## Project Structure

```bash
AI-Text-Summarization/
│
├── data/                   # Dataset storage
│   ├── raw/                # Raw data before preprocessing
│   ├── processed/          # Preprocessed data
│
├── src/                    # Source code
│   ├── data_preprocessing/  # Scripts for data cleaning and tokenization
│   ├── models/              # Transformer model scripts
│   ├── training/            # Model training code
│   ├── evaluation/          # Model evaluation scripts
│   ├── ui/                  # UI scripts (Flask backend, React/HTML frontend)
│
├── logs/                   # Training and evaluation logs
│
├── checkpoints/            # Model checkpoints
│
├── requirements.txt        # List of dependencies
│
├── README.md               # Project documentation
│
└── dockerfile              # Docker setup for deployment (optional)
```

## Installation

### Clone the repository
```bash
git clone https://github.com/Hung-341/AI-Text-Summarization-using-Transformers.git
cd AI-Text-Summarization-using-Transformers
```

### Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Collection
Download the dataset using Hugging Face's Datasets API:
```python
from datasets import load_dataset
dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
```

### 2. Preprocess Data
Run the preprocessing script to clean and tokenize the data:
```bash
python src/data_preprocessing/preprocess.py
```

### 3. Train the Model
To train the transformer model (BART or T5), run:
```bash
python src/training/train.py
```

### 4. Evaluate the Model
Evaluate the model's performance on the test set using ROUGE and BLEU scores:
```bash
python src/evaluation/evaluate.py
```

### 5. Run the UI
To start the Flask server and interact with the model through a simple UI:
```bash
python src/ui/app.py
```

### 6. Optional: Docker Deployment
If you want to deploy the project using Docker, build and run the Docker container:
```bash
docker build -t ai-summarization .
docker run -p 5000:5000 ai-summarization
```

## Dependencies

- Python 3.x
- Hugging Face Transformers
- Hugging Face Datasets
- PyTorch or TensorFlow
- Flask (for backend)
- ROUGE, BLEU (for evaluation)
- Docker (optional, for deployment)

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- Hugging Face for the pre-trained models and datasets.
- PyTorch/TensorFlow for the deep learning frameworks.

## Contact

For any questions or issues, feel free to open an issue or contact me via email at [hunglg.341@gmail.com].
