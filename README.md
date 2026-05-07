# Emotion-Aware Chatbot

An intelligent emotion-aware conversational AI system that detects user emotions and generates empathetic, context-aware responses in real time using fine-tuned transformer models and LLM integration.

---

## Features

* Emotion detection using fine-tuned RoBERTa
* Real-time empathetic response generation using Gemini API
* Hybrid emotion detection (keyword-based + deep learning)
* Smart memory for recent emotion tracking
* Repeated distress pattern detection
* Dual-layer safety filtering system
* Interactive Streamlit-based chatbot interface
* Evaluation using F1 score, confusion matrix, and loss curves

---

## System Architecture

```text
User Input
   ↓
Safety Filter
   ↓
Emotion Detection (RoBERTa)
   ↓
Smart Memory
   ↓
Gemini API
   ↓
Chatbot Response
```

---

## Dataset

* Dataset: GoEmotions (Google Research)
* Original Dataset Size: ~58,000 samples
* Training Used: 2000 samples
* Validation Used: 500 samples
* Emotion Mapping: 28 emotions simplified into 8 core emotions

---

## Technologies Used

### AI / Machine Learning

* RoBERTa
* Hugging Face Transformers
* PyTorch
* Gemini 2.5 Flash API

### Application Development

* Streamlit
* Python

### Data Processing & Evaluation

* NumPy
* Pandas
* Scikit-learn

---

## Evaluation Metrics

The model was evaluated using:

* Accuracy
* Macro F1 Score
* Weighted F1 Score
* Confusion Matrix
* Training vs Validation Loss Curve

---

## Project Structure

```text
emotion-aware-chatbot/
│
├── app.py
├── chatbot.ipynb
├── emotion_model_training.ipynb
├── phase2_complete.ipynb
├── emotion_model_v2/
├── confusion_matrix.png
├── f1_per_class.png
├── loss_curve.png
├── requirements.txt
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/DhruvTyagi6969/emotion-aware-chatbot.git
cd emotion-aware-chatbot
```

### Create Virtual Environment

```bash
python -m venv chatbot_env
source chatbot_env/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
streamlit run app.py
```

---

## Important Note

The trained model weights (`model.safetensors`) are excluded from the repository due to GitHub file size limitations.

---

## Future Improvements

* Train on full GoEmotions dataset
* Improve rare emotion classification
* Add voice input support
* Implement long-term memory
* Add multilingual support
* Deploy on cloud infrastructure

---

## Author

Dhruv Tyagi

GitHub:
[DhruvTyagi6969](https://github.com/DhruvTyagi6969?utm_source=chatgpt.com)
