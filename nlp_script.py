''' Author:      Ajay Jayanth
    Date:        9/22/20
    Description: Naive Bayes Algroithm predicts the Content Category feature of text 
                 using a tf-idf vectorized set of text data

'''
# C:\Users\msctb\AppData\Local\Programs\Python\Python38-32\Scripts

'''
rather than representing a text T in its feature space as {Word_i: count(Word_i, T) for Word_i in Vocabulary}, 
you can represent it in a topic space as {Topic_i: Weight(Topic_i, T) for Topic_i in Topics}
'''
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


MainDf = pd.read_csv('Label-Sample200.csv')
print(MainDf)
'''
Check for NaNs [FIXED]
is_NaN = MainDf.isnull()
row_has_NaN = is_NaN.any(axis = 1)
rows_with_NaN = MainDf[row_has_NaN]
print(rows_with_NaN)
'''

X_train, X_test, Y_train, Y_test = train_test_split(MainDf[['text', 'Audience_category', 'Content_source']], MainDf['Content_category'], test_size = 0.3, random_state = 10)
print(Y_train)
print(Y_test)