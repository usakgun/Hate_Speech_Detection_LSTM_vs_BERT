=============================================================================
Comparative Analysis of RNN and Transformer Models for Hate Speech Detection
=============================================================================

This repository contains the implementation code for the term project:
"Comparative Analysis of Recurrent Neural Networks and Transformer-Based Models 
for Hate Speech Detection in Social Media".

The project evaluates and compares two deep learning architectures:
1. Bi-LSTM (Bidirectional LSTM): A lightweight recurrent model initialized 
   with learned embeddings.
2. BERT (Bidirectional Encoder Representations from Transformers): A fine-tuned 
   transformer model (bert-base-uncased).

-----------------------------------------------------------------------------
Project Structure
-----------------------------------------------------------------------------
- main.py           : The main script that loads data, trains both models, 
                      performs error analysis, and outputs metrics.
- labeled_data.csv  : Dataset file (automatically downloaded by the script 
                      if not present).

-----------------------------------------------------------------------------
Installation & Usage
-----------------------------------------------------------------------------

1. Clone the repository:
   git clone https://github.com/usakgun/Hate_Speech_Detection_LSTM_vs_BERT.git
   cd Hate_Speech_Detection_LSTM_vs_BERT

2. Install Dependencies:
   (No separate requirements file needed)
   pip install pandas numpy torch scikit-learn transformers

3. Run the Project:
   python main.py

   *Note: The BERT training process might take 45-60 minutes on a CPU.*

-----------------------------------------------------------------------------
Results
-----------------------------------------------------------------------------
The models were evaluated on the Davidson et al. dataset. 
Below are the experimental results:

| Model    | Macro F1-Score | Accuracy | Inference Latency (per sample) |
|----------|----------------|----------|--------------------------------|
| Bi-LSTM  | 0.57           | 0.94     | 0.35 ms                        |
| BERT     | 0.53           | 0.94     | 35.77 ms                       |

KEY OBSERVATIONS:
-----------------
* Speed vs. Accuracy: 
  The Bi-LSTM model is approximately 100 times faster than BERT during 
  inference (0.35ms vs 35.77ms), confirming its suitability for real-time 
  edge applications.

* Precision vs. Recall: 
  While Bi-LSTM achieved a better F1-Score balance, BERT demonstrated 
  significantly higher Precision (0.68) for the hate class compared to 
  LSTM (0.53). This indicates that when BERT flags a tweet as hate speech, 
  it is highly likely to be correct.

-----------------------------------------------------------------------------
Dataset
-----------------------------------------------------------------------------
This project uses the "Hate Speech and Offensive Language Dataset" provided 
by Davidson et al.

- Source: https://github.com/t-davidson/hate-speech-and-offensive-language
- Classes: Mapped to binary labels (Hate vs. Non-Hate) for this study.

-----------------------------------------------------------------------------
Author
-----------------------------------------------------------------------------
Umut Sabri Akgün
Department of Computer Engineering
Bahçeşehir University
