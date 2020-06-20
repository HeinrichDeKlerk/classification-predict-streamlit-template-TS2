"""

    Simple Streamlit webserver application for serving developed classification
	models.

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
from markdown import markdown
from bs4 import BeautifulSoup
from PIL import Image
# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

@st.cache(persist=True)   # Improve speed and cache data

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
"""Tweet Classifier App with Streamlit """

# Creates a main title and subheader on your page -
# these are static across all pages
 
st.title("Tweet Classifier")
st.subheader("Climate change tweet classification")

# Creating sidebar 
# you can create multiple pages this way
st.sidebar.title("Pages")
selection = st.sidebar.radio(label="",options = ["Information","Exploratory Data Analysis","Insights","Prediction"])

# Building out the "Information" page
if selection == "Information":
    st.info("With the change in time, consumers have become more conscious about acquiring products/services from brands that uphold certain values and ideals. They also consider the service provider's stances towards issues such as climate change. In order to appeal to these consumers, organisations should understand their sentiments. They need to understand how their products will be received whilst trying to decrease their environmental impact or carbon footprint. This can be achieved using Machine Learning.")

    # You can read a markdown file from supporting resources folder
    if st.button("How does the app work"):
        app_info = markdown(open("resources/info.md").read())   
        st.markdown(app_info,unsafe_allow_html=True)
        
    st.subheader("Description of Sentiment Classes")
    descrip_image = Image.open("resources/imgs/climate_data_sentiment_description.png")
    st.image(descrip_image, use_column_width=True)

    st.subheader("Raw Twitter data and label")
    if st.checkbox('Show raw data'): # data is hidden if box is unchecked
        st.write(raw[['sentiment', 'message']]) # will write the df to the page
	
# Building out the predication page
if selection == "Prediction":
    st.info("Prediction with ML Models")
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
    
# Building EDA page
#eda = st.sidebar.select()
if selection == "Exploratory Data Analysis":
    st.info("Summarize the main characters of the data and get perspective on what the data can tell us.In this regard get more understanding about what it represents and how to apply it.")
    if st.checkbox("Preview DataFrame"):
        if st.button("Tail"):
            st.write(raw.tail())
        else:
            st.write(raw.head())

    # Viewing each sentiment
    sentiment = raw['sentiment'].unique()
    selected_sentiment = st.multiselect("View analysis by sentiment",sentiment)

    # mask to filter dataframe
    mask_sentiment = raw['sentiment'].isin(selected_sentiment)
    data = raw[mask_sentiment]

# Building the insights page
if selection == "Insights":
    st.info("Report on the insights gained from the analysis")


st.sidebar.title("About")
st.sidebar.info(
    """
    This app is maintained by EDSA students. It serves as a project
    for a classification sprint.

    **Authors:**\n
    Kennedy Mbono\n
    Nyandala Ramaru\n
    Marcus Moeng\n
    Heinrich De Klerk\n
    Nombulelo Msibi\n

"""
)


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
