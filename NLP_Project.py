import numpy as np
import pandas as pd
import nltk.downloader
import nltk
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import re
import os
import codecs # for encoding and decoding
from sklearn import feature_extraction
#import mpld3  #The mpld3 project brings together Matplotlib, the popular Python-based graphing library, and D3js, the popular JavaScript library for creating interactive data visualizations for the web. The result is a simple API for exporting your matplotlib graphics to HTML code which can be used within the browser, within standard web pages, blogs, or tools such as the IPython notebook.
from sklearn.metrics.pairwise import cosine_similarity  
import os 
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS
from nltk.tokenize import ToktokTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
import string
from nltk.stem.snowball import SnowballStemmer


nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('punkt')

import warnings 
warnings.filterwarnings(action = 'ignore')

###############################################################################

DataQ = pd.read_csv("C:/Users/saram/Downloads/6th Semester/NLP/Project/archive/Questions.csv",encoding='latin-1')
DataT = pd.read_csv("C:/Users/saram/Downloads/6th Semester/NLP/Project/archive/Tags.csv",encoding='latin-1')

print(DataQ.head())

DataT['Tag'] = DataT['Tag'].astype(str)
grouped_tags = DataT.groupby("Id")['Tag'].apply(lambda DataT: ' '.join(DataT))
#grouped_tags

grouped_tags_final = pd.DataFrame({'Id':grouped_tags.index, 'Tags':grouped_tags.values})
print(grouped_tags_final)

DataQ.drop(columns=['OwnerUserId', 'CreationDate', 'ClosedDate'], inplace=True)
new_question_df = DataQ.merge(grouped_tags_final, on='Id')
print(new_question_df)

print(f"Minimum Score: {new_question_df['Score'].min()}")
print(f"Maximum Score: {new_question_df['Score'].max()}")
#deleting queries with score less than 5
final_question_df =new_question_df.sample(200)

new_question_df = new_question_df[new_question_df['Score'] > 5]

new_question_df['Tags'] = new_question_df['Tags'].apply(lambda x: x.split())
print(new_question_df)
unique_tags = list(set([item for sublist in new_question_df['Tags'].values for item in sublist]))

print(len(unique_tags))

final_question_df =new_question_df.drop('Score',axis=1)
final_question_df =final_question_df.drop('Id',axis=1)
print(final_question_df)
###############################################################################

flat_list = [item for sublist in final_question_df['Tags'].values for item in sublist]
keywords = nltk.FreqDist(flat_list)
keywords = nltk.FreqDist(keywords)
print(sum(keywords.values()))
flat_list = [item for sublist in final_question_df['Tags'].values for item in sublist]
keywords = nltk.FreqDist(flat_list)
frequencies_words = keywords.most_common(100)
tags_features = [word[0] for word in frequencies_words]
print(tags_features)

###############################################################################

def most_common(tags):
    tags_filtered = []
    for i in range(len(tags)):
        if tags[i] in tags_features:
            tags_filtered.append(tags[i])
    return tags_filtered


final_question_df['Tags'] = final_question_df['Tags'].apply(lambda x: most_common(x))
final_question_df['Tags'] = final_question_df['Tags'].apply(lambda x: x if len(x) > 0 else None)
final_question_df.dropna(subset=['Tags'], inplace=True)
print(final_question_df)

###############################################################################

#remove <p> in body column
final_question_df['Body'] = final_question_df['Body'].apply(lambda x: BeautifulSoup(x).get_text())
print(final_question_df['Body']) 

stopwords = nltk.corpus.stopwords.words('english')

###############################################################################
def casefolding(text):
    text = [ word for word in text if word.casefold() not in stopwords]
    #text = text.lower()
    #text = text.strip(' ')
    return text


def no_punct(text):
    #word_list = []
    #word_list = text.split()
    no_punct_filtered_speech_words = [''.join(char for char in word if char not in string.punctuation) for word in text]
    no_punct_filtered_speech_words = [word for word in no_punct_filtered_speech_words if word] # To remove empty strings
    
    from collections import Counter
    words_counter = Counter(no_punct_filtered_speech_words)
    return ' '.join(no_punct_filtered_speech_words)


token = ToktokTokenizer()
#lemma = WordNetLemmatizer()

def lemmatizeWords(text):
    lemmatizer = WordNetLemmatizer()
    words = token.tokenize(text)
    listLemma = []
    for w in words:
        x = lemmatizer.lemmatize(w, pos="v")
        listLemma.append(x)
    return ' '.join(map(str, listLemma))

def stopWordsRemove(text):
    stopwords = nltk.corpus.stopwords.words('english')
    words = token.tokenize(text)
    filtered = [w for w in words if not w in stopwords]
    return ' '.join(map(str, filtered))

stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

##################################################################

# Remove stopwords, punctuation and lemmatize for text in body
final_question_df['Body'] = final_question_df['Body'].apply(lambda x: casefolding(x)) 
final_question_df['Body'] = final_question_df['Body'].apply(lambda x: no_punct(x)) 
final_question_df['Body'] = final_question_df['Body'].apply(lambda x: lemmatizeWords(x))
final_question_df['Body'] = final_question_df['Body'].apply(lambda x: stopWordsRemove(x))

# Remove stopwords, punctuation and lemmatize for title.
final_question_df['Title'] = final_question_df['Title'].apply(lambda x: str(x))
final_question_df['Title'] = final_question_df['Title'].apply(lambda x: casefolding(x)) 
final_question_df['Title'] = final_question_df['Title'].apply(lambda x: no_punct(x)) 
final_question_df['Title'] = final_question_df['Title'].apply(lambda x: lemmatizeWords(x)) 
final_question_df['Title'] = final_question_df['Title'].apply(lambda x: stopWordsRemove(x))

print(final_question_df)

###############################################################################
from sklearn.model_selection import train_test_split

xx = final_question_df.drop('Tags', axis=1)
X_train, X_test, y_train, y_test = train_test_split(xx,final_question_df['Tags'], test_size = 0.2, random_state = 0)

print(xx.head)
###############################################################################

from sklearn.preprocessing import LabelEncoder

# Perform label encoding
label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)

###############################################################################
from sklearn.feature_extraction.text import TfidfVectorizer

'''x_trainl = X_train.tolist()
x_testl = X_test.tolist()
y_trainl = y_train.tolist()
y_testl = y_test.tolist()'''
#yyy = final_question_df['Title'].tolist()

# tfidf vectorizer 
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

#fit the vectorizer to data
tfidf_vectorizer.fit(final_question_df['Body'])
xtrain_tfidfB =  tfidf_vectorizer.transform(X_train)
xtest_tfidfB =  tfidf_vectorizer.transform(X_test)

tfidf_vectorizer.fit(final_question_df['Title'])
xtrain_tfidfT =  tfidf_vectorizer.transform(X_train)
xtest_tfidfT =  tfidf_vectorizer.transform(X_test)


print(xtrain_tfidfB.data)
print(xtrain_tfidfT.data)

"""
tfidf_matrix = tfidf_vectorizer.fit_transform(xxx) 
tfidf_matrix2 = tfidf_vectorizer.fit_transform(yyy) 

terms = tfidf_vectorizer.get_feature_names()
print(tfidf_matrix.data)
print(tfidf_matrix2.data)
"""

#X_tfidf, y_bin, test_size = 0.2, random_state = 0)



###############################################################################
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Initialize the SVM model
svm_model = SVC(kernel='linear')

# Train the model on the training data
svm_model.fit(X_train, y_train)

# Predict the tags for the test data
y_pred = svm_model.predict(X_test)

# Print the model's accuracy
print("SVM Accuracy:", accuracy_score(y_test, y_pred))