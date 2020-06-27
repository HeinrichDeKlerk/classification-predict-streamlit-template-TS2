"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import numpy as np

# Text processing
import string
import re

#import spacy
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

# Data processing
from sklearn.utils import resample
from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split

# Visualization 
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
plt.style.use('ggplot')

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		raw = pd.read_csv("resources/train.csv")
		# Detect and remove duplicate rows
		raw = raw.drop_duplicates(subset=['message'])

        # Remove blanks
		def remove_blanks(df):
			blanks = []
			for index, tweet in enumerate(df['message']):
				if type(tweet) == str:
					if tweet in ['', ' ']:
						blanks.append(index)
			return df.drop(blanks)
		raw = remove_blanks(raw)
		 
		# Remove special characters
		def clean_text(text):
			text = str(text).lower()
			text = re.sub('\[.*?\]', '', text) 
			text = re.sub('https?://\S+|www\.\S+', 'URL', text)
			text = re.sub('<.*?>+', '', text)
			text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
			text = re.sub('\n', '', text)
			text = re.sub('\w*\d\w*', '', text)
			return text
		raw['clean_tweet'] = raw['message'].apply(lambda x: clean_text(x))

		# Remove stop-words
		stop_words = stopwords.words('english')  # Assign stop_words list
		def remove_stopword(text):
			return [word for word in text.split() if word not in stop_words]
		raw['clean_tweet'] = raw['clean_tweet'].apply(lambda x: remove_stopword(x))

		# Join text
		def join_text(text):
			text = ' '.join(text)
			return text
		raw['clean_tweet'] = raw['clean_tweet'].apply(lambda x: join_text(x))

		# Assign feature and response variables
		X = raw['clean_tweet']
		y = raw['sentiment']

		# Addressing imbalance
		heights = [len(y[y == label]) for label in [0, 1, 2, -1]]
		bars = pd.DataFrame(zip(heights, [0,1,2,-1]), columns=['heights','labels'])
		bars = bars.sort_values(by='heights',ascending=True)

		# Let's pick a class size of roughly half the size of the largest size
		class_size = 3500
		bar_label_df = bars.set_index('labels')
		resampled_classes = []

		for label in [0, 1, 2, -1]:
		# Get number of observations from this class
			label_size = bar_label_df.loc[label]['heights']

    		# If label_size < class size the upsample, else downsample
			if label_size < class_size:
				# Upsample
				label_data = raw[['clean_tweet', 'sentiment']][raw['sentiment'] == label]
				label_resampled = resample(label_data,
							    		    # sample with replacement
                                			# (we need to duplicate observations)
                                   			replace=True,
                                			# number of desired samples
                                			n_samples=class_size,
                                			random_state=27)
			else:
		    	# Downsample
				label_data = raw[['clean_tweet', 'sentiment']][raw['sentiment'] == label]
				label_resampled = resample(label_data,
                                        	# sample without replacement
                                			# (no need for duplicate observations)
                                			replace=False,
                                			# number of desired samples
                                			n_samples=class_size,
                                			random_state=27)

			resampled_classes.append(label_resampled)

		# Assign feature and response variables from resampled data
		resampled_data = np.concatenate(resampled_classes, axis=0)

		X_resampled = resampled_data[:, :-1]
		y_resampled = resampled_data[:, -1]

		# Plot resampled data against original data
		if st.checkbox("View plot of resampled data"):
			heights = [len(y_resampled[y_resampled == label]) for label in [0, 1, 2, -1]]
			bars_resampled = pd.DataFrame(zip(heights, [0, 1, 2, -1]),
											  columns=['heights', 'labels'])
			bars_resampled = bars_resampled.sort_values(by='heights', ascending=True)

			fig = go.Figure(data=[
							go.Bar(name='Original', x=[-1, 0, 2, 1], y=bars['heights']),
							go.Bar(name='Resampled', x=[-1, 0, 2, 1], y=bars_resampled['heights'])
			])
			fig.update_layout(xaxis_title="Sentiment", yaxis_title="Sample size")
			st.pyplot()

		df_resampled = pd.DataFrame(X_resampled.reshape(-1,1))
		df_resampled.columns = ['tweet']
		df_resampled['sentiment'] = y_resampled
		df_resampled['sentiment'] = df_resampled['sentiment'].astype('int')

		# Splitting data
		X_train, X_test, y_train, y_test = train_test_split(
										   df_resampled['tweet'].values,
										   df_resampled['sentiment'].values,
 										   test_size=0.1, random_state=42)

		# Create a spaCy tokenizer
		#spacy.load('en')
		#lemmatizer = spacy.lang.en.English()

		#def tokenize(text):
		#	tokens = lemmatizer(text)
		#return [token.lemma_ for token in tokens]

		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
