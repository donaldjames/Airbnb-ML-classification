from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import streamlit as st

def precision_recall(X, Y, model, parameter):
	# Use label_binarize to be multi-label like settings
	Y = label_binarize(Y, classes=[0, 1, 2])
	n_classes = Y.shape[1]

	# Split into training and test
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33, random_state=42)

	# We use OneVsRestClassifier for multi-label prediction
	# Run classifier
	classifier = OneVsRestClassifier(model)
	if parameter == 0:
		# DT
		y_score = classifier.fit(X_train, Y_train).predict_proba(X_test)
	else:
		# SVM
		y_score = classifier.fit(X_train, Y_train).decision_function(X_test)

	# For each class
	precision = dict()
	recall = dict()
	average_precision = dict()
	for i in range(n_classes):
	    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
	                                                        y_score[:, i])
	    average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

	# A "micro-average": quantifying score on all classes jointly
	precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
	    y_score.ravel())
	average_precision["micro"] = average_precision_score(Y_test, y_score,
	                                                     average="micro")
	print('Average precision score, micro-averaged over all classes: {0:0.2f}'
	      .format(average_precision["micro"]))

	plt.figure()
	plt.step(recall['micro'], precision['micro'], where='post')

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title(
	    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
	    .format(average_precision["micro"]))
	st.pyplot()
	return