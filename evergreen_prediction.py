# Framework Structure

# FEATURE CREATION

# If Training:
# 1. create_features - Create features using training data
# 2. _fit_transform - call to apply preprocessors to dataset
# 3. _proc_fit_transform
# 4. fit data to preprocessors (ex: imputer, scaler, etc.)
# 5. transform data using preprocessor

# If Predicting:
# 1. create_features - Create features using training data
# 2. _transform - call to apply all applicable preprocessors to dataset
# 3. transform data using preprocessor

# TRAIN MODEL
# - Apply a selected model against the feature set (X) and responses (y)

# CROSS VALIDATE
# - compute cross_val_score to evaluate model

##########################################

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import itertools
import urllib
import urllib2
import math


class Featurizer():
  def __init__(self):
    self._imputer = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    self._binarizer = preprocessing.Binarizer()
    self._scaler = preprocessing.StandardScaler()
    self._preprocs = [self._imputer,
                      #self._scaler, 
                      #self._binarizer 
                      ]

  def _fit_transform(self, dataset):
    for p in self._preprocs:
      dataset = self._proc_fit_transform(p, dataset)
    return dataset

  def _transform(self, dataset):
    for p in self._preprocs:
      dataset = p.transform(dataset)
    return dataset

  def _proc_fit_transform(self, p, dataset):
    p.fit(dataset)
    dataset = p.transform(dataset)
    return dataset


  def create_features(self, dataset, train_len, training=False):
    numerical_data = dataset[['avglinksize',
                              'compression_ratio',
                              'frameTagRatio',
                              'image_ratio',
                              'lengthyLinkDomain',
                              'is_news',
                              'linkwordscore',
                              'numwords_in_url',
                              'alchemy_category_score']]
    
    alchemy_category_features = pd.get_dummies(dataset['alchemy_category'])
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, min_df=1)

    keywords = []
    for line in dataset['alchemy_keywords']:
      words_list = []
      try:
        for word in line.split():
          try:
            encoded_word = word.encode('-utf-8')
          except UnicodeDecodeError:
            encoded_word = ""
          words_list.append(encoded_word)
      except AttributeError:
        continue  
      keywords_str = " ".join(str(x) for x in words_list)
      keywords.append(keywords_str)

    keyword_series = pd.Series(keywords)
    keyword_features = pd.DataFrame(vectorizer.fit_transform(keyword_series).toarray())
    data = pd.concat([numerical_data, alchemy_category_features, keyword_features], axis=1)

    if training:
      data = self._fit_transform(data[:train_len])
    else:
      data = self._transform(data[train_len:])
    return data


# LogisticRegression() - 0.84941550354397377
def train_model(X, y):
  # model = GradientBoostingClassifier(n_estimators=100, max_depth=10)
  # model = RidgeClassifierCV(alphas=[ 0.1, 1., 10. ])
  model = LogisticRegression()
  # model = DecisionTreeClassifier() 
  # model = RandomForestClassifier(n_estimators=100) 
  model.fit(X, y)
  #print model.coef_
  return model


def main():
  train_data = pd.read_csv('train.tsv', sep = '\t')
  test_data = pd.read_csv('test.tsv', sep = '\t')
  full_data = pd.concat([train_data, test_data])
  full_data.replace(to_replace='?', value=np.nan, inplace=True)
  train_len = len(train_data)
  featurizer = Featurizer()
  
  print "train_data: %d, test_data: %d, full_data: %d" % (train_len, len(test_data), len(full_data))
  print "Transforming dataset into features..."
  X = featurizer.create_features(full_data, train_len, training=True)
  print "Training Feature Matrix X:"
  # print X
  y = train_data.label
  print "Training Responses y:"
  # print y

  print "Training model..."
  model = train_model(X,y)

  print "Cross validating..."
  print np.mean(cross_val_score(model, X, y, scoring='roc_auc'))


if __name__ == '__main__':
  main()




