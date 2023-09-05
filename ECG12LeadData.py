# evaluate multinomial logistic regression model
import pandas as pd
import sys
import numpy as np
import re
import csv
import os
import glob
import zipfile
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, f1_score, auc

def get_data(path):
  diagnostic_features = pd.read_excel(path, engine = 'openpyxl')
  diagnostic_features_ar = np.asarray(diagnostic_features)
  X = diagnostic_features_ar[:, 3:15]
  labels = diagnostic_features_ar[:, 1]
  return X, labels

def encode_data(input):
  label_encoder = LabelEncoder()
  transormed_input = label_encoder.fit_transform(input)
  return transormed_input

def encode_labels(labels):
  label_encoder = LabelEncoder()
  labels = encode_data(labels)

  for i in range(labels.shape[0]):
    if labels[i] in (0,1):
        labels[i] = 0
    elif labels[i] in (2,3,4,6,9,10):
        labels[i] = 1
    elif labels[i] in (5,8):
        labels[i] = 2
    else:
        labels[i] = 3

  return labels

def encode_age(age):
  for i in range(age.shape[0]):
    if age[i] <= 55:
        age[i] = 0
    elif age[i] > 55 and age[i] <=70:
        age[i] = 1
    else:
       age[i] = 2

  return age

def predict_condition_from_ecg(path_prefix, model_name):

	#get_data
	file_path = prefix + 'Diagnostics.xlsx'
	X, labels = get_data(file_path)

	#encode_labels()
	label_encoder = LabelEncoder()
	X[:, 1] = encode_data(X[:, 1])
	X[:,0] = encode_age(X[:,0])
	labels = encode_labels(labels)
	y = labels

	norm1 = MinMaxScaler()
	norm2 = StandardScaler()

	X[:,2:] = norm2.fit_transform(X[:,2:])
	if model_name == "logistic_regression":
		model = LogisticRegression(multi_class='multinomial', solver='saga', max_iter = 1000)
	elif model_name == "random_forest_classifier":
		model = RandomForestClassifier(max_depth=100, random_state=0)
	elif model_name == "svm":
		model = svm.SVC()
	elif model_name == "xgboost":
		model = XGBClassifier()

	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the model and collect the scores
	n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	# report the model performance
	print('Mean accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

	trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3, random_state=0)
	clf = model.fit(trainX, trainy)

	predict = model.predict(testX)

	conf_mat = confusion_matrix(testy, predict)

	print(conf_mat)

	print(metrics.classification_report(testy, predict, digits=3))

if __name__ == "__main__":
	prefix = '/content/drive/MyDrive/12 Lead ECG Database/'
	if len(sys.argv) > 1:
		predict_condition_from_ecg(prefix, sys.argv[1])
	else:
		print("Please enter the model name")
