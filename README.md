# 🛡️ Hindi Hate Speech Detection

An NLP research project that detects hate speech in Hindi text by translating it to English and classifying it using classical ML models (TF-IDF + Logistic Regression / SVM / Naive Bayes), with a DistilBERT deep learning model built as a comparative experiment. The best-performing pipeline is deployed as a Flask web app.

> Built as a research project (professor-approved) to explore and compare traditional ML vs. transformer-based approaches for a low-resource-language hate speech task.

🔗 **Live Demo**: [hate-speech-detection-research-purpose.onrender.com](https://hate-speech-detection-research-purpose.onrender.com/)

> Note: hosted on Render's free tier — the app may take 30–60 seconds to spin up on first load if it's been idle.

---

## 📌 Overview

- **Task**: Binary hate speech classification (Hate / Not Hate)
- **Language**: Hindi (input) → translated to English for modeling
- **Dataset**: [HASOC](https://hasocfire.github.io/hasoc/) Hindi hate speech dataset
- **Approach**: Translate Hindi → English, clean text, vectorize with TF-IDF, train and compare multiple classifiers
- **Deployment**: Flask web app (English pipeline) with a simple UI + REST API

---

## 🚀 Live Pipeline (What's Deployed)

The deployed model is an end-to-end scikit-learn `Pipeline` (text cleaning → TF-IDF → Logistic Regression), saved as `Hate_speech_pipeline2.pkl`, served through a Flask app (`main.py`) with an HTML front end (`index.html`).

**Note:** The current deployment only accepts **English text**. The Hindi→English translation step was used to build the training dataset; it is not wired into the live Flask app's request path.

---

## 🧠 Models Experimented With

Multiple models were trained and compared on TF-IDF features (unigrams to trigrams, max 15,000 features) before selecting the final one:

| Model | Accuracy | Notes |
|---|---|---|
| Naive Bayes | Experimented | Used as an early baseline |
| Logistic Regression (balanced) | **80.13%** | Best standalone accuracy |
| SVM (LinearSVC, balanced, calibrated) | 80.02% | Comparable to LR |
| End-to-end Pipeline — SVM | 79.59% | Full pipeline incl. cleaning step |
| **End-to-end Pipeline — Logistic Regression** | **79.81%** | ✅ **Deployed model** |
| DistilBERT (fine-tuned) | Experimental | Built as a comparison to the ML baseline (see below) |

**Classification report — Logistic Regression (balanced):**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Not Hate (0) | 0.77 | 0.82 | 0.79 |
| Hate Speech (1) | 0.83 | 0.79 | 0.81 |
| **Accuracy** | | | **0.80** |

> Logistic Regression was selected as the final model for deployment due to its strong, consistent accuracy (~80%) and lower inference cost compared to SVM/DistilBERT — an important factor since the dataset is fairly balanced and the project targets a lightweight, easily deployable API.

---

## 🤖 DistilBERT Experiment (Deep Learning Comparison)

As a further research extension, a separate notebook (`BERT_HateSpeech.ipynb`) fine-tunes `distilbert-base-uncased` on the same dataset and train/test split (`random_state=42`, `test_size=0.2`) for a fair comparison against the TF-IDF + LR/SVM baseline.

Key differences from the ML pipeline:

- **Minimal cleaning** was used for BERT (only URL/mention removal, whitespace normalization) — heavy cleaning like stopword removal and lemmatization was intentionally *not* applied, since it can destroy contextual meaning that transformer models rely on (e.g. removing "not" can flip a sentence's meaning).
- **Heavy cleaning** (stopword removal, lemmatization, punctuation stripping) was used for the classical ML models, since TF-IDF treats each token independently and benefits from noise reduction.
- Trained for 3 epochs using `AdamW` (`lr=2e-5`) on CPU.
- This model is **not deployed** — it exists as a research comparison and is not part of the live Flask app.

---

## 🧹 Preprocessing Pipeline (ML models)

1. Lowercasing
2. Remove URLs, `@mentions`, and `#` symbols
3. Remove punctuation
4. Strip non-English characters (residual Hindi tokens after translation)
5. Stopword removal (`sklearn.ENGLISH_STOP_WORDS`, with negations `no`, `not`, `never`, `nor` explicitly preserved to retain sentiment-flipping meaning)
6. Lemmatization (`NLTK WordNetLemmatizer`)

---

## 🗂️ Dataset

- **Source**: HASOC Hindi hate speech dataset (`hindi_dataset.tsv`)
- **Labels**: `task_1` (`HOF` / `NOT`) mapped to binary `Label_binary` (`1` = hate, `0` = not hate)
- **Translation**: Hindi text translated to English using `deep_translator` (`GoogleTranslator`)
- **Final size**: 4,655 rows after removing nulls/duplicates
- **Class balance**: 2,465 hate (1) / 2,190 not-hate (0) — reasonably balanced

---

## 🛠️ Tech Stack

- **Language**: Python
- **Data handling**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **NLP**: NLTK (lemmatization), scikit-learn (TF-IDF, stopwords)
- **ML Models**: scikit-learn (Logistic Regression, SVM/LinearSVC, Naive Bayes)
- **Deep Learning**: PyTorch, HuggingFace Transformers (DistilBERT)
- **Translation**: `deep_translator` (Google Translate API wrapper)
- **Web Framework**: Flask
- **Deployment**: Render

---

## 📁 Project Structure

```
hate-speech-detection/
├── main.py                              # Flask app (routes: /, /predict)
├── templates/
│   └── index.html                       # Web UI (AI-generated front end)
├── Hate_speech_pipeline2.pkl            # Deployed model (clean → TF-IDF → LR)
├── notebooks/
│   ├── hindi_to_english_translation.ipynb   # Hindi → English translation of HASOC dataset
│   ├── Hate_speech_hindi.ipynb              # ML experiments: NB, LR, LR (balanced), SVM
│   └── BERT_HateSpeech.ipynb                # DistilBERT fine-tuning experiment
├── translated_hindi_dataset.csv         # Translated dataset used for training
└── README.md
```

---

## ⚙️ Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/PraB-05/Hate-Speech-Detection-Research-Purpose.git
   cd Hate-Speech-Detection-Research-Purpose
   ```

2. **Install dependencies**
   ```bash
   pip install flask pandas numpy scikit-learn nltk
   ```

3. **Place the trained model** (`Hate_speech_pipeline2.pkl`) in the project root, and make sure `main.py` points to it with a **relative path**, e.g.:
   ```python
   with open('Hate_speech_pipeline2.pkl', 'rb') as f:
       model = pickle.load(f)
   ```
   *(Update this from the local absolute path used during development before deploying.)*

4. **Run the app locally**
   ```bash
   python main.py
   ```
   The app will be available at `http://localhost:5000`

---

## 🌐 API Usage

**Base URL:** `https://hate-speech-detection-research-purpose.onrender.com`

**Endpoint:** `POST /predict`

**Request:**
```json
{
  "text": "your sentence here"
}
```

**Response:**
```json
{
  "Prediction": "Hate speech"
}
```

or

```json
{
  "Prediction": "No Hate speech"
}
```

---

## ⚠️ Limitations

- The deployed app accepts **English input only**; Hindi text must be pre-translated (this step is not yet wired into the live app).
- Translation quality (via Google Translate) directly affects prediction accuracy on the original Hindi text.
- Accuracy (~80%) reflects a research-stage model trained on a moderately sized dataset (~4.6K samples) — not production-hardened.
- DistilBERT results are experimental and were not benchmarked to completion for deployment comparison.

## 🔭 Future Work

- Integrate the Hindi→English translation step directly into the `/predict` endpoint so raw Hindi input can be submitted.
- Complete and benchmark the DistilBERT model against the ML baseline for a full comparison.
- Explore Hindi-native transformer models (e.g. IndicBERT, MuRIL) to avoid translation-induced information loss.
- Expand the dataset for better generalization.

---

## 🙏 Acknowledgements

- [HASOC](https://hasocfire.github.io/hasoc/) for the Hindi hate speech dataset
- Project completed for academic/research purposes under faculty guidance
- Front-end UI (`index.html`) was AI-generated

---

## 👤 Author

**Prasann Bhorkar**
GitHub: [PraB-05](https://github.com/PraB-05)
Project repo: [Hate-Speech-Detection-Research-Purpose](https://github.com/PraB-05/Hate-Speech-Detection-Research-Purpose)

