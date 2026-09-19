# Spam Email Classifier (NLP + Machine Learning)

A web app that classifies an email as **SPAM** or **HAM** using TF-IDF features
and a Multinomial Naive Bayes classifier, served through Flask.

## Pipeline

```
User email
  -> text cleaning (lowercase, remove URLs, punctuation, special characters)
  -> tokenization (NLTK)
  -> stopword removal (NLTK)
  -> TF-IDF vectorizer (text becomes numbers)
  -> Multinomial Naive Bayes
  -> SPAM / HAM prediction
  -> confidence from model.predict_proba()
  -> separate keyword rules that explain the text in plain language
  -> Flask + HTML/CSS/JS interface
```

## Folder structure

```
Spam-Email-Classifier/
├── dataset/
│   └── spam.csv
├── model/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

## Setup (Windows)

```
cd Spam-Email-Classifier
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Then open http://127.0.0.1:5000

## Dataset

`dataset/spam.csv` needs one text column and one label column.
`train_model.py` recognises common names automatically:

* text: `text`, `message`, `email`, `body`, `content`, `sms`, `v2`
* label: `label`, `category`, `class`, `target`, `spam`, `type`, `v1`

Labels may be `spam`/`ham` or `1`/`0`. Spam becomes 1, ham becomes 0.
The script prints which columns it selected, so it never guesses silently.

Inspect any dataset before training:

```
python -c "import pandas as pd; d=pd.read_csv('dataset/spam.csv', encoding='latin-1'); print(d.columns.tolist()); print(d.head())"
```

The dataset in `dataset/spam.csv` has 5,728 emails (4,360 ham, 1,368 spam)
with the columns `text` and `spam`, where 1 = spam and 0 = ham. Training keeps
5,695 rows after duplicates are removed.

Metrics from the training run on this dataset (test split of 1,139 emails):

```
Accuracy : 0.9965
Precision: 0.9891
Recall   : 0.9964
F1-score : 0.9927

                 Predicted HAM   Predicted SPAM
Actual HAM               862                3
Actual SPAM                1              273
```

## Notes

* Metrics printed by `train_model.py` come from the actual test split.
  Nothing is hard-coded.
* The confidence percentage is `model.predict_proba()` for the predicted class.
  It is the model's probability, not a guarantee of correctness.
* The "Why?" bullets come from a separate keyword layer. They are a
  human-readable explanation, not the internal features of the Naive Bayes
  decision.
* NLTK stopwords and the tokenizer download automatically on first run. If
  there is no internet, the code falls back to a built-in stopword list and a
  regex tokenizer so the project still works.
