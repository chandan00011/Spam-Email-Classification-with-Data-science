import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# 1. Load Dataset
file_path = "C:/Users/Tanuj_Mann/Downloads/spam (1).csv"
df = pd.read_csv(file_path, encoding='latin1')

# 2. Check columns and head
print("Columns:", df.columns)
print(df.head())

# 3. Rename for clarity if needed, and drop empty cols if any
df = df.rename(columns={'v1':'label', 'v2':'text'})
df = df[['label', 'text']]

# 4. Map labels to 0/1
df['label'] = df['label'].map({'spam':1, 'ham':0})

# 5. Feature Extraction
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(df['text'])
y = df['label']

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# 8. Evaluate
print("\nAccuracy:", accuracy_score(y_test, model.predict(X_test)))
print("\nClassification Report:\n", classification_report(y_test, model.predict(X_test)))

# Confusion Matrix
cm = confusion_matrix(y_test, model.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Top 20 keywords overall
feature_names = vectorizer.get_feature_names_out()
total_counts = X.sum(axis=0).A1
freq_df = pd.DataFrame({'term': feature_names, 'count': total_counts})
top_keywords = freq_df.sort_values(by='count', ascending=False).head(20)
plt.figure(figsize=(10,6))
sns.barplot(x='count', y='term', data=top_keywords, palette='viridis')
plt.title('Top 20 Keywords by Frequency')
plt.xlabel('Frequency')
plt.ylabel('Keyword')
plt.show()

# ✅ New Visualization 1: Spam vs Ham count
plt.figure(figsize=(6,4))
sns.countplot(x='label', data=df, palette='coolwarm')
plt.title('Number of Spam vs Ham Messages')
plt.xlabel('Label (0=Ham, 1=Spam)')
plt.ylabel('Count')
plt.show()

# ✅ New Visualization 2: Message Length Distribution
df['message_length'] = df['text'].apply(len)
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='message_length', hue='label', bins=50, kde=True, palette='coolwarm')
plt.title('Distribution of Message Lengths by Label')
plt.xlabel('Message Length')
plt.ylabel('Frequency')
plt.show()

# ✅ New Visualization 3: Word Cloud for Spam
spam_words = ' '.join(df[df['label']==1]['text'])
spam_wc = WordCloud(width=600, height=400, background_color='black').generate(spam_words)
plt.figure(figsize=(8,6))
plt.imshow(spam_wc, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Spam Messages')
plt.show()

# Word Cloud for Ham
ham_words = ' '.join(df[df['label']==0]['text'])
ham_wc = WordCloud(width=600, height=400, background_color='black').generate(ham_words)
plt.figure(figsize=(8,6))
plt.imshow(ham_wc, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Ham Messages')
plt.show()

# ✅ New Visualization 4: Top 20 Spam-only Keywords
spam_df = df[df['label']==1]
spam_tfidf = vectorizer.fit_transform(spam_df['text'])
spam_feature_names = vectorizer.get_feature_names_out()
spam_total_counts = spam_tfidf.sum(axis=0).A1
spam_freq_df = pd.DataFrame({'term': spam_feature_names, 'count': spam_total_counts})
top_spam_keywords = spam_freq_df.sort_values(by='count', ascending=False).head(20)
plt.figure(figsize=(10,6))
sns.barplot(x='count', y='term', data=top_spam_keywords, palette='magma')
plt.title('Top 20 Keywords in Spam Messages')
plt.xlabel('Frequency')
plt.ylabel('Keyword')
plt.show()

# updated dataset
file_path = "C:/Users/Tanuj_Mann/Downloads/spam (1).csv"
df = pd.read_csv(file_path, encoding='latin1')
df = df.rename(columns={'v1':'label', 'v2':'text'})
df['label'] = df['label'].map({'spam':1, 'ham':0})
df['message_length'] = df['text'].apply(len)
df.to_csv('spam_cleaned.csv', index=False)


spam_df = df[df['label'] == 1]

# Use CountVectorizer to get word counts
vectorizer = CountVectorizer(stop_words='english', max_features=20)
X = vectorizer.fit_transform(spam_df['text'])

# Sum word counts across all spam messages
total_counts = X.sum(axis=0).A1
keywords = vectorizer.get_feature_names_out()

# Create DataFrame with keywords and counts
keyword_df = pd.DataFrame({'keyword': keywords, 'count': total_counts})

# Sort by count descending
keyword_df = keyword_df.sort_values(by='count', ascending=False)

keyword_df.to_csv('spam_top_keywords.csv', index=False)

keyword_df.to_csv('spam_top_keywords.csv', index=False)
