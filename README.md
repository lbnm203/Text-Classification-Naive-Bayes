# Text-Classification-Use-Naive-Bayes

This project is a simple implementation of a spam message classification model using the Naive Bayes algorithm. The model is trained on a dataset of spam and non-spam messages and can predict whether a given message is spam or not.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Contributing](#contributing)

## Project Overview

The project aims to build a text classification model to detect spam messages. The model uses natural language processing (NLP) techniques to preprocess text data and the Naive Bayes classifier for the prediction task.

## Features

- Preprocessing text data (lowercasing, punctuation removal, tokenization, stopword removal, stemming).
- Creation of a word dictionary based on the training data.
- Feature extraction for each message based on the word dictionary.
- Model training using Naive Bayes.
- Evaluation of the model on validation and test datasets.
- Predicting whether a new message is spam or not.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/lbnm203/Text-Classification-Naive-Bayes.git
    ```

2. Navigate to the project directory:

    ```bash
    cd Text-Classification-Naive-Bayes
    ```

4. Download NLTK data:

    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    ```

## Usage

1. **Train the model and evaluate its performance:**

    Run the `main.py` script to train the model and evaluate it on validation and test datasets:

    ```bash
    python main.py
    ```

2. **Make a prediction:**

    The `main.py` script also includes a sample prediction where you can input a custom message to see if it’s classified as spam or not.

## File Descriptions

- `main.py`: The main script to train the model, evaluate it, and make predictions.
- `preprocess.py`: Contains all the functions for text preprocessing and feature extraction.
- `predict.py`: Contains the function for making predictions using the trained model.
- `data/2cls_spam_text_cls.csv`: The dataset used for training and evaluating the model.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.
