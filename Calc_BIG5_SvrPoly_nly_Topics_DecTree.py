import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn import svm
#import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import hstack
from sklearn import cross_validation
from datetime import time
from sklearn.cross_validation import cross_val_score
from sklearn import tree

''' Prepare data '''
data = pd.read_csv('reg_input_0.9.csv')
print "Data Read : ",data.columns.shape
features = ['topics','WC', 'Clout','Tone', 'WPS', 'Sixltr','i', u'we', 'you', 'they', 'article', 'negate','compare', 'number','anx','sad','family', 'friend', 'female','male','insight', 'certain','see', 'hear', 'feel','body', 'health','ingest', 'achieve', 'power', 'risk', 'focuspast','focusfuture', 'work','leisure', 'home', 'money', 'relig', 'death', 'informal','swear', 'assent', 'nonflu', 'filler', 'AllPunc','Period', 'Comma', 'Colon', 'SemiC', 'Exclam', 'Dash','Quote', 'Apostro', 'Parenth', 'OtherP']
#print features[0]
X = data[features[0]]
Y = data['ope']
print "X",len(X) #(48701, 55)
print "Y",len(Y) #(48701, 1)
topic_train = X


''' Vectorizing X '''
vect = CountVectorizer()
vect.fit(topic_train)
topic_train_dtm = vect.transform(topic_train)
print "topic train dym shape", topic_train_dtm.shape

''' TFIDF Transform '''
tfidf = TfidfTransformer(norm="l2")
tfidf.fit(topic_train_dtm)
topic_train_tfidf_dtm = tfidf.transform(topic_train_dtm)
print "topic_train_tfidf_dtm size", topic_train_tfidf_dtm.shape #(48701, 96459)


''' Getting the LIWC features '''
#lwic = X.filter(['WC', 'Clout','Tone', 'WPS', 'Sixltr','i', u'we', 'you', 'they', 'article', 'negate','compare', 'number','anx','sad','family', 'friend', 'female','male','insight', 'certain','see', 'hear', 'feel','body', 'health','ingest', 'achieve', 'power', 'risk', 'focuspast','focusfuture', 'work','leisure', 'home', 'money', 'relig', 'death', 'informal','swear', 'assent', 'nonflu', 'filler', 'AllPunc','Period', 'Comma', 'Colon', 'SemiC', 'Exclam', 'Dash','Quote', 'Apostro', 'Parenth', 'OtherP'])
#print features[1:]
lwic = data[features[1:]]
print "lwic.shape : ",lwic.shape
lwic_array = lwic.as_matrix()
lwic_array_numpy = np.array(lwic_array) #(48701, 54)
print "lwic_array_numpy.shape : ",lwic_array_numpy.shape


''' Only TOPICS '''
X_matrix = topic_train_tfidf_dtm
print "X_matrix shape : ",X_matrix.shape #(48701, 96513)

#X_matrix =  lwic_array_numpy
#Y_array=Y.as_matrix()
Y_matrix=np.array(Y).reshape(len(Y),1)
print "Y_matrix shape : ",Y_matrix.shape	#(48701, 1)

'''print Y_matrix[0]
print Y_matrix[1]
print X_matrix[0][1]
print X_matrix[1][1]'''

'''
# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X_matrix, Y_matrix)
regr_2.fit(X_matrix, Y_matrix)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
'''

depth = []
for i in range(2,7):
	clf = tree.DecisionTreeRegressor(criterion='mse',max_depth=i)
	# Perform 5-fold cross validation
	scores = cross_val_score(estimator=clf, X=X_matrix, y=Y_matrix, cv=5, n_jobs=4,scoring='mean_squared_error')
	mse = -scores
	rmse = np.sqrt(mse)
	depth.append((i,rmse.mean()))
	print i,rmse.mean()
print(depth)

