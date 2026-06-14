# Email Spam Classification with Naive Bayes, FastAPI, and n8n

An end-to-end email spam detection pipeline built with a Multinomial Naive Bayes classifier trained on 190,000+ emails, served via a FastAPI endpoint, and integrated with Gmail through an n8n automation workflow.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [API Deployment](#api-deployment)
- [n8n Workflow Integration](#n8n-workflow-integration)
- [Limitations and Future Work](#limitations-and-future-work)

---

## Project Overview

This project implements a complete spam detection system, covering raw data exploration and preprocessing through to a live Gmail integration that automatically flags and removes incoming spam emails.

The core classifier uses a TF-IDF vectorizer paired with a Multinomial Naive Bayes model, achieving 95.23% accuracy and a ROC-AUC of 0.9935 on a held-out test set of 38,319 emails. The trained model is exported and served through a FastAPI REST API, which is called in real time by an n8n workflow triggered on every incoming Gmail message.

---

## Repository Structure

```
├── spam-ham-eda-github.ipynb          # Exploratory data analysis
├── spam-ham-cleaning-github.ipynb     # Data preprocessing pipeline
├── spam-ham-model-evaluation-github.ipynb  # Model training and evaluation
├── spam-ham-api-github.ipynb          # FastAPI server + ngrok tunnel (Colab)
├── data/
│   ├── raw/                           # Original dataset
│   └── processed/
│       └── cleaned.csv                # Output of preprocessing pipeline
├── models/
│   └── naive_bayes_model.pkl          # Trained model (joblib)
└── README.md
```

---

## Dataset

**Source:** [190k Spam/Ham Email Dataset on Kaggle](https://www.kaggle.com/datasets/meruvulikith/190k-spam-ham-email-dataset-for-classification/data)

The dataset contains 193,852 labelled emails drawn primarily from commercial and pharmaceutical spam alongside corporate ham emails from the Enron email corpus.

| Class | Count | Proportion |
|-------|-------|------------|
| Spam  | ~102,700 | ~53% |
| Ham   | ~91,150  | ~47% |

The dataset is near-balanced, so no resampling techniques were required.

---

## Exploratory Data Analysis

Notebook: `spam-ham-eda-github.ipynb`

Key findings from EDA:

**Class distribution.** Approximately 53% spam and 47% ham. Class imbalance is not a significant concern.

**Email length.** Ham emails are on average longer than spam emails (334 words vs 210 words). Ham emails often contain full workplace conversation threads, while spam tends to be concise and promotional. Several extreme ham outliers exceeded 1.5 million words and were identified as likely corrupted records.

**Word frequency and word clouds.** The token `escapenumber` (a masked numerical placeholder) appears frequently in both classes. More distinctive patterns emerge beneath this token: spam emails are dominated by marketing and pharmaceutical terminology, while ham emails reflect corporate communication patterns.

**Bigram analysis.** Spam bigrams are typically associated with URLs, pricing, and product promotions. Ham bigrams reflect forwarded message threads and workplace correspondence.

**Handcrafted feature analysis.** Spam emails are more likely to contain URLs, exclamation marks, and dollar signs. URL presence and exclamation mark frequency were identified as the strongest indicators of spam.

> **Diagram placeholder**: class balance pie chart

> **Diagram placeholder**: email length distribution by class (box plot)

> **Diagram placeholder**: top 20 unigrams per class (bar chart)

---

## Data Preprocessing

Notebook: `spam-ham-cleaning-github.ipynb`

A comprehensive preprocessing pipeline was applied to transform the raw dataset into a machine-learning-ready format.

### Dataset Evolution

| Stage | Rows Remaining | Rows Removed |
|-------|---------------|--------------|
| Raw Dataset | 193,852 |: |
| Null Removal | 193,850 | 2 |
| Outlier Removal | 191,911 | 1,939 |
| Empty Text Removal | 191,594 | 317 |
| **Final Dataset** | **191,594** | **2,258 total** |

### Pipeline Steps

**Null removal.** 2 rows with missing values were dropped.

**Outlier removal.** Class-specific 99th percentile thresholds were used to remove 1,939 anomalous rows based on extreme text length, as identified during EDA. Legitimate long-form emails were preserved.

**Text normalisation.** Unwanted characters were removed, obfuscated spam terms were corrected, and noise was filtered. The `escapenumber` token was intentionally retained as numerical patterns are informative for spam detection.

**Stopword removal.** Standard stopwords were removed while preserving contextually important terms including negations, pronouns, and common call-to-action words.

**Lemmatization.** Words were reduced to their root forms to consolidate vocabulary.

**Empty text removal.** 317 rows whose content became empty after cleaning were dropped.

The final cleaned dataset was saved to `data/processed/cleaned.csv`.

---

## Model Training

Notebook: `spam-ham-model-evaluation-github.ipynb`

A TF-IDF vectorizer was paired with a Multinomial Naive Bayes classifier. Multiple configurations were evaluated across vocabulary sizes, n-gram ranges, and smoothing parameters.

### Best Configuration

| Parameter | Value |
|-----------|-------|
| Vectorizer | TF-IDF |
| Max features | 50,000 |
| N-gram range | (1, 1) — unigrams only |
| Classifier | Multinomial Naive Bayes |
| Alpha (smoothing) | 0.5 |

Notably, all top-performing configurations used unigrams rather than bigrams. Adding bigrams substantially increased vocabulary size without meaningfully improving classification performance, suggesting that individual words already provide sufficient discriminative signal.

The trained model was saved to `models/naive_bayes_model.pkl` using joblib.

---

## Model Evaluation

Test set size: **38,319 emails**

### Summary Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 95.23% |
| F1 Score | 0.9486 |
| ROC-AUC | 0.9935 |
| PR-AUC | 0.9922 |

### Per-Class Performance

| Class | Precision | Recall | Notes |
|-------|-----------|--------|-------|
| Ham | High | Very high | Very few legitimate emails flagged as spam |
| Spam | High | 0.9268 | 1,332 spam emails misclassified as ham |

### Error Analysis

**False negatives (spam missed):** 1,332 spam emails were classified as ham. Many of these resemble formal business communication and lack the obvious markers typically associated with spam.

**False positives (ham flagged):** 495 legitimate emails were incorrectly flagged as spam, typically due to the presence of URLs, numerical patterns, or keywords commonly associated with spam.

> **Diagram placeholder**: confusion matrix

> **Diagram placeholder**: ROC curve

> **Diagram placeholder**: prediction probability distribution (spam vs ham)

### Key Observations

The ROC-AUC of 0.9935 indicates excellent class separation, meaning the model is highly effective at ranking spam above legitimate emails across all decision thresholds. The PR-AUC of 0.9922 further confirms strong performance in the high-precision region, which is important for real-world spam filtering where false positives are disruptive.

The primary limitation is spam recall at 0.9268. Well-written spam that mimics professional communication remains the hardest class of errors to address with a bag-of-words approach.

---

## API Deployment

Notebook: `spam-ham-api-github.ipynb`

The trained model is served as a REST API using FastAPI, deployed on Google Colab and exposed publicly via an ngrok tunnel.

### Endpoint

```
POST /predict
```

**Request body:**
```json
{
  "text": "Email subject and body text here"
}
```

**Response:**
```json
{
  "prediction": "SPAM",
  "confidence": 0.9408,
  "flagged": true
}
```

### Running the API (Colab)

1. Run Cell 1 to install dependencies
2. Run Cell 2 to mount Google Drive and load the model
3. Run Cell 3 to define the FastAPI app
4. Run Cell 4 to start the server and open the ngrok tunnel
5. Copy the `/predict` URL printed in Cell 4 output

Additional endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Returns model status |
| `/docs` | GET | Interactive API documentation (Swagger UI) |
| `/predict` | POST | Spam classification |

---

## n8n Workflow Integration

The FastAPI endpoint is connected to Gmail through an n8n automation workflow. The workflow polls Gmail every minute for new emails, sends each email to the model, and automatically labels detected spam.

### Workflow Architecture

```
Gmail Trigger → HTTP Request (POST /predict) → IF prediction = SPAM → Add SPAM Label → Delete Message
                                              → IF prediction = HAM  → (no action)
```

### Workflow Nodes

**Gmail Trigger.** Polls the inbox every minute and outputs email metadata including subject, snippet, sender, and message ID.

**HTTP Request.** Sends a POST request to the ngrok `/predict` endpoint with the email subject and snippet combined as the input text. Returns `prediction`, `confidence`, and `flagged`.

**IF node.** Routes the email based on the `prediction` field. Emails classified as `SPAM` proceed to the true branch; ham emails exit through the false branch untouched.

**Add Label.** Applies the Gmail `SPAM` label to flagged messages.

**Delete Message.** Permanently removes the flagged email from the inbox.

### Why a custom model over Gmail's built-in filter?

Gmail's native spam filter is rule-based, relying on keyword matching, sender reputation, and known blacklists. This pipeline is ML-based and offers several advantages:

- Learns statistical patterns across the full email rather than matching fixed rules
- Can detect novel spam that does not match known signatures
- Confidence scores are available for threshold tuning
- The model is fully owned and retrainable on domain-specific data
- Suitable as a second-pass filter on top of Gmail's existing detection

---

## Limitations and Future Work

**Model limitations.** Multinomial Naive Bayes treats words as independent features and ignores word order and context. This makes it susceptible to false positives when legitimate emails contain spam-related keywords, and false negatives when spam emails use formal business language.

**Dataset scope.** The dataset is drawn primarily from commercial spam and Enron corporate emails. Performance may degrade on email domains that differ significantly from this distribution.

**Infrastructure.** The current deployment relies on Google Colab and ngrok, which are not persistent. A production deployment would require hosting the FastAPI model on a persistent service such as Railway, Render, or a cloud VM so the endpoint remains live independently of a Colab session.

### Potential Improvements

- Threshold tuning to reduce false positives at the cost of slightly lower spam recall
- Additional handcrafted features (URL count, exclamation mark frequency, dollar sign presence) alongside TF-IDF
- Experimentation with alternative models including Logistic Regression, Support Vector Machines, and transformer-based architectures such as BERT
- Retraining on a broader and more diverse email dataset to improve generalisation
- Persistent API deployment for production use

---

## Results Summary

A TF-IDF and Multinomial Naive Bayes pipeline trained on 191,594 emails achieves **95.23% accuracy**, **F1 of 0.9486**, and **ROC-AUC of 0.9935** on a test set of 38,319 emails, demonstrating that a relatively simple NLP baseline can perform excellently on spam classification while highlighting clear opportunities for further refinement.
