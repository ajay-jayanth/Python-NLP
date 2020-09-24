''' 
    Author:      Ajay Jayanth
    Date:        9/22/20
    Description: Naive Bayes Algroithm predicts the Content Category feature of text 
                 using a tf-idf vectorized set of text data
'''
# C:\Users\msctb\AppData\Local\Programs\Python\Python38-32\Scripts

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


MainDf = pd.read_csv('Label-Sample200.csv')
#print(MainDf)
'''
Check for NaNs [FIXED]
is_NaN = MainDf.isnull()
row_has_NaN = is_NaN.any(axis = 1)
rows_with_NaN = MainDf[row_has_NaN]
print(rows_with_NaN)
'''

#Split data to training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(MainDf['text'], MainDf['Content_category'], test_size = 0.3, random_state = 10)
#Vectorize training feature
train_Vect = CountVectorizer()
V_X_train = train_Vect.fit_transform(X_train)
'''
The Vectorizer takes the dataset if text values (i.e. training X values) and turns it into a matrix of token counts of the
whole vocabulary or a limited vocabulary that can be set
'''
#Vectorize the testing feature
test_Vect = CountVectorizer(vocabulary=train_Vect.vocabulary_)
V_X_test = test_Vect.fit_transform(X_test)

'''
The model will be MultinomialNB, which is a scikit model that predicts from values tf-idf vectors using
a Naive-Bayes Classifier
'''
model = MultinomialNB()
model = model.fit(V_X_train, Y_train)

#Predict the Y-Values from the model
Y_Predicted = model.predict(V_X_test)

#Output the classification report and confusion matrix
print("Training Size: %d" % X_train.shape[0])
print("Test Size: %d" % X_test.shape[0])
print(metrics.classification_report(Y_test, Y_Predicted, zero_division = 0)) #Because of small sample size of questions, the precision and f-score could be dividing by 0
print("Confusion matrix: ")
print(metrics.confusion_matrix(Y_test, Y_Predicted))