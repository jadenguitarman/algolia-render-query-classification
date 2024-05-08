from pandas import read_csv
from joblib import dump

import re, string
from nltk import pos_tag, download
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
download('punkt')
download('averaged_perceptron_tagger')
download('wordnet')
download('stopwords')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer

df_train = read_csv('dataset.tsv', sep='\t')

def get_wordnet_pos(tag):
	if tag.startswith('J'):
		return wordnet.ADJ
	elif tag.startswith('V'):
		return wordnet.VERB
	elif tag.startswith('N'):
		return wordnet.NOUN
	elif tag.startswith('R'):
		return wordnet.ADV
	else:
		return wordnet.NOUN

wl = WordNetLemmatizer()
def preprocess(text):
	text = text.lower()
	text = text.strip()
	text = re.compile('<.*?>').sub('', text)
	text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
	text = re.sub('\s+', ' ', text)
	text = re.sub(r'\[[0-9]*\]',' ', text)
	text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
	text = re.sub(r'\d',' ', text)
	text = re.sub(r'\s+',' ', text)
	text = ' '.join([i for i in text.split() if i not in stopwords.words('english')])

	word_pos_tags = pos_tag(
		word_tokenize(
			text
		)
	)
	# Map the position tag and lemmatize the word/token
	return " ".join([wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)])

df_train['clean_text'] = df_train['query'].apply(preprocess)
df_train.head()

X_train, X_val, y_train, y_val = train_test_split(df_train["clean_text"],
												  df_train["target"],
												  test_size=0.2,
												  shuffle=True)

tfidf_vectorizer = TfidfVectorizer(use_idf=True)
X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_vectors_tfidf = tfidf_vectorizer.transform(X_val)

dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

lr_tfidf = LogisticRegression(solver='liblinear', C=10, penalty='l2')
lr_tfidf.fit(X_train_vectors_tfidf, y_train)

y_predict = lr_tfidf.predict(X_val_vectors_tfidf)
y_prob = lr_tfidf.predict_proba(X_val_vectors_tfidf)[:,1]
print(classification_report(y_val, y_predict))
print('Accuracy Score:', accuracy_score(y_val, y_predict))
fpr, tpr, thresholds = roc_curve(y_val, y_prob)
print('AUC:', auc(fpr, tpr))

dump(lr_tfidf, 'classifier.joblib')
