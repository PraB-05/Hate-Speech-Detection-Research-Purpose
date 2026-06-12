# Hate-Speech-Detection-Research-Purpose
In this we have did the research on how a model will behave based on abusive / hate speech data if we train it on various ml or deel learning algorithm based on its metrices and learn various techniques along with cleaning and feature eng. and this analysis is all about text-based data so we have dig deeper in NLP.

Overview

This project aims to detect hate speech from social media text using Natural Language Processing (NLP) and Machine Learning techniques. The original dataset consisted of Hindi text samples, which were translated into English before applying preprocessing, feature extraction, and classification techniques.

The project explores traditional machine learning approaches such as Logistic Regression and Naive Bayes and is being extended with Support Vector Machines (SVM) and Transformer-based models like DistilBERT.

Workflow
1. Dataset Preparation
Collected a labeled Hindi hate speech dataset.
Translated Hindi text into English to leverage English NLP preprocessing techniques and machine learning pipelines.
Inspected and cleaned the translated dataset.
2. Text Preprocessing

Applied multiple NLP preprocessing techniques:

Lowercasing
Punctuation removal
Stop-word removal
Tokenization
Stemming
3. Feature Engineering

Used TF-IDF (Term Frequency–Inverse Document Frequency) to convert textual data into numerical feature vectors suitable for machine learning models.

4. Model Training

Implemented and evaluated:

Logistic Regression
Multinomial Naive Bayes

Currently experimenting with:

Support Vector Machine (SVM)
DistilBERT
5. Model Evaluation

Performance was evaluated using:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix
Results
Model	Accuracy
Logistic Regression	81%
Multinomial Naive Bayes	Evaluated

The Logistic Regression model achieved approximately 81% accuracy on the translated dataset.

Future Work
Evaluate Support Vector Machine (SVM) performance.
Fine-tune DistilBERT for contextual text classification.
Compare traditional machine learning models against transformer-based architectures.
Analyze the impact of translation on classification performance.
Deploy the model as an interactive web application.
