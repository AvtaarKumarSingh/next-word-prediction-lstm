# Next Word Prediction using LSTM

An AI-powered next-word prediction and sentence completion application built using Python, TensorFlow, LSTM, and Streamlit. The model learns word sequences from an IT Helpdesk and Customer Support text dataset to predict the next word and generate sentence continuations.

## Features

- Next-word prediction based on input text.
- Sentence completion by predicting multiple words.
- LSTM neural network for learning sequential word patterns.
- Text tokenization and sequence padding.
- Interactive web interface built with Streamlit.
- Adjustable prediction length.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Streamlit
- Natural Language Processing (NLP)
- Long Short-Term Memory (LSTM)

## Project Structure

```text
next-word-prediction-lstm/
├── app.py              # Streamlit web application
├── codefile.ipynb      # Model development and training notebook
├── data.csv            # Text dataset
├── lstm.keras          # Saved LSTM model
├── tokenizer.pkl       # Saved tokenizer
├── max_len.pkl         # Maximum sequence length
└── README.md           # Project documentation
```

## Dataset

This project uses a custom IT Helpdesk and Customer Support text dataset containing approximately 1,200 synthetic English sentences.

The dataset contains IT-related scenarios and technical support sentences for learning word sequences and predicting sentence continuations.

- Format: CSV
- File: `data.csv`
- Text column: `data`
- Approximate size: 1,200 sentences
- Purpose: Training a next-word prediction model

The dataset was created for this project. No external public dataset is claimed as the source of this text data.

## How It Works

1. Data Preparation: Loads sentences from the CSV dataset.
2. Tokenization: Converts words into numerical tokens.
3. Sequence Generation: Creates input sequences and their next-word targets.
4. Padding: Makes input sequences the same length.
5. Model Training: Trains an LSTM neural network to learn word patterns.
6. Model Saving: Saves the trained model and tokenizer.
7. Prediction: Predicts the next word based on the input text.
8. Sentence Generation: Repeats prediction to generate multiple words.
9. Web Interface: Displays the completed sentence through Streamlit.

## Application Preview

Here is a preview of the NextWord AI application built using Streamlit.

<p align="center">
  <img src="images/app-screenshot.png" alt="NextWord AI Streamlit Application" width="850">
</p>

<p align="center">
  <em>Interactive next-word prediction and sentence completion interface.</em>
</p>

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/AvtaarKumarSingh/next-word-prediction-lstm.git
```

### 2. Navigate to the Project Directory

```bash
cd next-word-prediction-lstm
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
python -m pip install streamlit tensorflow numpy pandas
```

Use a Python version compatible with your installed TensorFlow version.

### 5. Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

http://localhost:8501

## Usage

1. Launch the Streamlit application.
2. Enter a sentence or a few starting words.
3. Select the number of words to predict.
4. Click the Predict Next Words button.
5. View the generated sentence.

Example input:

```text
The password reset may
```

The model predicts a continuation based on patterns learned during training. Actual output depends on the model and its vocabulary.

## Model Files

- `lstm.keras`: Saved LSTM neural network.
- `tokenizer.pkl`: Tokenizer used to convert words into numerical sequences.
- `max_len.pkl`: Saved maximum sequence length, if used by the application.

Keep the model, tokenizer, and sequence-length configuration consistent. If you retrain the model, save and use the matching files.

## Limitations

- The model is trained on a relatively small synthetic dataset.
- Predictions are limited by the training vocabulary and learned patterns.
- Generated sentences may be repetitive or grammatically incorrect.
- The project is intended for educational and demonstration purposes.
- A larger and more diverse dataset may improve prediction quality.

## Future Improvements

- Expand the dataset with more diverse technical support sentences.
- Improve text preprocessing and vocabulary handling.
- Evaluate prediction accuracy on a separate test dataset.
- Experiment with different LSTM architectures and hyperparameters.
- Add top-k predictions and confidence scores.
- Explore Transformer-based language models.
- Deploy the application to a cloud hosting platform.

## Learning Objectives

This project demonstrates practical concepts in:

- Natural Language Processing (NLP)
- Text preprocessing and tokenization
- Sequential data preparation
- Deep learning with LSTM
- Model training and serialization
- Building machine learning applications with Streamlit
