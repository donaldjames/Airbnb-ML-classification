import streamlit as st
import pandas as pd
import models
import seaborn as sns

dataset_name = "Airbnb Dataset"
labels = [0, 1, 2]
labels1 = [0, 1, 2, 3, 4, 5]

# Function to pre-process data for predicting the best hotels based on the review_count
def airbnb_load_preprocess():
	airbnb_data = pd.read_csv('AB_NYC_2019.csv')
	raw_data = airbnb_data
	airbnb_data.dropna(subset=["last_review"], inplace=True)
	years = pd.DatetimeIndex(airbnb_data['last_review']).year
	# sns.catplot(x="room_type", y="price", data=airbnb_data)
	# st.pyplot()
	# sns.catplot(x="neighbourhood", y="price", data=airbnb_data)
	# st.pyplot()
	# sns.catplot(x="room_type", y="number_of_reviews", data=airbnb_data)
	# st.pyplot()
	for i in range(len(airbnb_data)):
		if years[i] < 2015:
			airbnb_data.drop(airbnb_data.index[i], inplace=True)
	airbnb_data = airbnb_data.reset_index(drop=True)
	X = airbnb_data[['id', 'host_id', 'neighbourhood_group', 'neighbourhood', 'room_type', 'price']]
	# convert categorigcal variables to numeric via one-hot encoding
	X = pd.get_dummies(X)
	# divide number_of_reviews into bins to get different classifications
	airbnb_data['reviews_bins'] = pd.cut(x=airbnb_data['number_of_reviews'], bins=[0, 30, 50, 650], labels=[0, 1, 2])
	Y = airbnb_data.reviews_bins
	return X, Y

X, Y = airbnb_load_preprocess()
model = ["------", "Naive Bayes", "Decision Tree", "Perceptron", "MLP", "XGBoost"]
option = st.sidebar.selectbox('Machine Learning Model', model)
with st.spinner('Training the model'):
	if option == "Naive Bayes":
		model = models.naive_bayes(X, Y, labels, dataset_name)
	if option == "Decision Tree":
		model = models.decision_tree(X, Y, labels, dataset_name, random_state=5, criterion='entropy', ccp_alpha=0.005, min_samples_split=2)
	if option == "Perceptron":
		model = models.single_layer_perceptron(X, Y, labels, dataset_name, eta0=0.1, random_state=0, max_iter=100)
	if option == "MLP":
		model = models.mlp(X, Y, labels, dataset_name, random_state=0, learning_rate=0.05, activation='logistic', hidden_layer_sizes=(6,), max_iter=500)
	if option == "XGBoost":
		model = models.xgboost_model(X, Y, labels, dataset_name)
	else:
		pass

menu = st.sidebar.checkbox("About Info")
if menu:
	st.write("Supervised ML for Airbnb dataset. Using Streamlit for visualisation and applying Naive Bayes, Decision Tree, Single and Multi-layer Perceptron, XGBoost")
	st.write("This is the second part of group coursework (COMM055, University of Surrey). The members of group are Amit Bechelet, Donald James, Hisham Parol, Namra Sultan")
	st.write("Github link: https://github.com/donaldjames/Airbnb-ML-classification")