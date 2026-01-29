Comparative Analysis of RNN and Transformer Models for Hate Speech Detection

This repository contains the implementation code for the term project **"Comparative Analysis of Recurrent Neural Networks and Transformer-Based Models for Hate Speech Detection in Social Media"**.

The project evaluates and compares two deep learning architectures:
1.  **Bi-LSTM (Bidirectional LSTM):** A lightweight recurrent model using embeddings.
2.  **BERT (Bidirectional Encoder Representations from Transformers):** A fine-tuned transformer model (`bert-base-uncased`).

## Structure

- `main.py`: The main script that loads data, trains both models, and outputs performance metrics.
- `labeled_data.csv`: Dataset file (automatically downloaded by the script if not present).

## Installation & Usage

### 1. Clone the repository
```bash
### git clone [https://github.com/usakgun/Hate_Speech_Detection_LSTM_vs_BERT.git](https://github.com/usakgun/Hate_Speech_Detection_LSTM_vs_BERT.git)
cd Hate_Speech_Detection_LSTM_vs_BERT

2. Install Dependencies
Install the required libraries with this command:
pip install pandas numpy torch scikit-learn transformers

3. Run the Project
You can run the training and evaluation script with a single command:
python main.py
Note: The BERT training process might take 45-60 minutes on a CPU environment.

Results
The models were evaluated on the Davidson et al. dataset. Below is a summary of the performance trade-offs observed during the study:
Model | Macro F1-Score | Accuracy | Inference Latency (per sample)
Bi-LSTM | 0.58 | 0.94 | ~0.29 ms
BERT | [0.XX] | [0.XX]| [XX.XX] ms

Observation: While BERT generally provides better semantic understanding, Bi-LSTM offers significantly lower latency, making it suitable for real-time edge applications.

Dataset
This project uses the Hate Speech and Offensive Language Dataset provided by Davidson et al.

Source: t-davidson/hate-speech-and-offensive-language

👤 Author
Umut Sabri Akgün Department of Computer Engineering Bahçeşehir University
