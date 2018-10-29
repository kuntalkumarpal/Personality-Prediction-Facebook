import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn import linear_model
import numpy as np
from scipy.sparse import hstack
from sklearn import cross_validation
from datetime import time
from datetime import date
import datetime
import random
import collections
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

import math
data = pd.read_csv('reg_input_0.1.csv')
print (data.columns)
data_topics_liwc_op = data.filter([ 'topics', 'WC', 'Clout','Tone', 'WPS', 'Sixltr','i', u'we', 'you', 'they', 'article', 'negate','compare', 'number','anx','sad','family', 'friend', 'female','male','insight', 'certain','see', 'hear', 'feel','body', 'health','ingest', 'achieve', 'power', 'risk', 'focuspast','focusfuture', 'work','leisure', 'home', 'money', 'relig', 'death', 'informal','swear', 'assent', 'nonflu', 'filler', 'AllPunc','Period', 'Comma', 'Colon', 'SemiC', 'Exclam', 'Dash','Quote', 'Apostro', 'Parenth', 'OtherP','ope'], axis =1)
print (data_topics_liwc_op)

X = data_topics_liwc_op.filter([ 'topics', 'WC', 'Clout','Tone', 'WPS', 'Sixltr','i', u'we', 'you', 'they', 'article', 'negate','compare', 'number','anx','sad','family', 'friend', 'female','male','insight', 'certain','see', 'hear', 'feel','body', 'health','ingest', 'achieve', 'power', 'risk', 'focuspast','focusfuture', 'work','leisure', 'home', 'money', 'relig', 'death', 'informal','swear', 'assent', 'nonflu', 'filler', 'AllPunc','Period', 'Comma', 'Colon', 'SemiC', 'Exclam', 'Dash','Quote', 'Apostro', 'Parenth', 'OtherP'], axis =1)
Y = data_topics_liwc_op.filter(['ope'],axis=1)
print("X")
print(X)
print("Y")
print(Y)

topic_train = X.topics
print(topic_train)
vect = CountVectorizer()
vect.fit(topic_train)
topic_train_dtm = vect.transform(topic_train)
print("topic train dym shape")
print(topic_train_dtm.shape)
topic_train_dtm_numpy = np.asarray(topic_train_dtm)
print("Actual countvectoriser")
print (topic_train_dtm_numpy.shape)

tfidf = TfidfTransformer(norm="l2")
tfidf.fit(topic_train_dtm)
topic_train_tfidf_dtm = tfidf.transform(topic_train_dtm)
print("tfidf actual size")
topic_train_tfidf_dtm_numpy = csr_matrix(topic_train_tfidf_dtm)
print("numpy tfidf")
print (topic_train_tfidf_dtm_numpy.shape)
lwic = X.filter(['WC', 'Clout','Tone', 'WPS', 'Sixltr','i', u'we', 'you', 'they', 'article', 'negate','compare', 'number','anx','sad','family', 'friend', 'female','male','insight', 'certain','see', 'hear', 'feel','body', 'health','ingest', 'achieve', 'power', 'risk', 'focuspast','focusfuture', 'work','leisure', 'home', 'money', 'relig', 'death', 'informal','swear', 'assent', 'nonflu', 'filler', 'AllPunc','Period', 'Comma', 'Colon', 'SemiC', 'Exclam', 'Dash','Quote', 'Apostro', 'Parenth', 'OtherP'])

lwic_array = lwic.as_matrix()
lwic_array_numpy = np.array(lwic_array)
print("lwic")
print(lwic_array_numpy.shape)
X_matrix = topic_train_tfidf_dtm
print "X_matrix : ",X_matrix.shape
Y_array=Y.as_matrix()
Y_matrix=np.array(Y_array)



print("SVR Linear")
clf = svm.SVR(kernel='linear')

scores = cross_val_score(clf, X_matrix, Y_matrix, cv=5	, n_jobs=4, scoring='mean_squared_error')
print("scores")
print(scores)
mse_scores = -scores
print("mse")
print(mse_scores)
rmse_scores = np.sqrt(mse_scores)
print("rmse")
print(rmse_scores)
print("mean rmse")
print(rmse_scores.mean())


print("SVR polynomial")
k_range = list(range(1, 5))
k_scores = []
for k in k_range:
	clf = svm.SVR(kernel='poly', degree = k)
	print "k=",k
	scores = cross_val_score(clf, X_matrix, Y_matrix, cv=5, n_jobs=4, scoring='mean_squared_error')
	print("scores")
	print(scores)
	mse_scores = -scores
	print("mse")
	print(mse_scores)
	rmse_scores = np.sqrt(mse_scores)
	print("rmse")
	print(rmse_scores)
	print("mean rmse")
	print(rmse_scores.mean())
	k_scores.append(rmse_scores.mean())
print(k_scores)

