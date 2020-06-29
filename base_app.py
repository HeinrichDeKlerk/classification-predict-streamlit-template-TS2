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
import joblib
import os
import pickle
from markdown import markdown

# Data dependencies
import pandas as pd
import numpy as np

# Text processing
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer, word_tokenize
import string
import re

# Data processing
from sklearn.utils import resample
from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split

# Visual dependencies
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from bs4 import BeautifulSoup
from PIL import Image
import plotly.graph_objects as go

# Visual dependencies
matplotlib.use("Agg")
plt.style.use('ggplot')

# Define spacy dependencies
spacy.load('en')
lemmatizer = spacy.lang.en.English()

# Load your raw data
read_and_cache_csv = st.cache(pd.read_csv, allow_output_mutation=True)
raw = read_and_cache_csv("resources/kaggle_train.csv")


# Define custom functions
def tokenize(text):
    tokens = lemmatizer(text)
    return [token.lemma_ for token in tokens]


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key


@st.cache
def clean_text(text):
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', 'URL', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text


@st.cache(persist=True)
def prep_eda_df(df):

        # preprocess eda data
        # Tweet length by word count, character count, and punctuation count
        eda_data = raw.copy()
        # Extract URL's
        pattern_url = r'(http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+)'
        eda_data['Url'] = eda_data['message'].str.extract(pattern_url)
        # Replace URL with string 'web-url'
        eda_data['message'] = eda_data['message'].replace(pattern_url, 'web-url', regex=True)

        # Clean text with clean_text() function
        eda_data['clean_tweet'] = eda_data['message'].apply(lambda x: clean_text(x))

        # Tokenize tweets with nltk
        # tokeniser = word_tokenize()
        eda_data['tokens'] = eda_data['message'].apply(word_tokenize)
        eda_data['tweet_length'] = eda_data['tokens'].str.len()

        # Tweet Character count column
        eda_data['character_count'] = eda_data['message'].apply(lambda c: len(c))
        # repeat for punctuation
        eda_data['punctuation_count'] = eda_data['message'].apply(lambda x: len([i for i in str(x) if i in string.punctuation]))
        eda_df = eda_data.copy()
        return eda_df

# Prepare eda data
eda_data = prep_eda_df(raw)


def sent_kde_plots(df, values, target):
        fig, ax = plt.subplots()
        col = list(df[target].unique())

        for c in col:
            sns.kdeplot(df[values][df[target] == c], shade=True, label=sent_dict.get(c))

        plt.xlabel(values)
        plt.ylabel('Density')
        plt.title('Distribution of Tweet {}'.format(values))
        return


def wordcloud_gen(df, target, values):
        sent = list(df[target].unique())
        dft = train_data.groupby(target)[values].apply(' '.join)
        for s in sent:
            text = dft[s]
            wordcloud = WordCloud(background_color='white', max_words=100,
                                  max_font_size=50).generate(text)
            plt.figure()
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title('Tweets under {} Class'.format(s))
            plt.axis('off')
        return


# label the sentiments
def sentiment_label(df_):
        if df_['sentiment'] == 2:
                return "News"
        elif df_['sentiment'] == 1:
                return "Pro"
        elif df_['sentiment'] == 0:
                return "Neutral"
        elif df_['sentiment'] == -1:
                return "Anti"


# Load pickled models and Vectorizers
file = open("resources/mod_and_vect.pkl", "rb")
TF_1 = pickle.load(file)
TF_2 = pickle.load(file)
CV_2 = pickle.load(file)
NL_SVM_TF1 = pickle.load(file)
LR_TF2 = pickle.load(file)
LSVM = pickle.load(file)
LRCV = pickle.load(file)
file.close()

# Define sentiment dictionary
sent_dict = {-1: 'Anti', 0: 'Neutral', 1: 'Pro', 2: 'News'}

# Apply string labels to sentiments
raw["label"] = raw.apply(sentiment_label, axis=1)


# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages

    st.title("Tweet Classifier")
    st.subheader("Climate change tweet classification")

    # Creating sidebar
    # you can create multiple pages this way
    st.sidebar.title("Pages")
    selection = st.sidebar.radio(label="", options=["Information", "EDA and Insights", "Prediction", "Technical"])

    # Building out the "Information" page
    if selection == "Information":
            st.info("With the change in time, consumers have become more conscious about acquiring products/services from brands that uphold certain values and ideals. They also consider the service provider's stances towards issues such as climate change. In order to appeal to these consumers, organisations should understand their sentiments. They need to understand how their products will be received whilst trying to decrease their environmental impact or carbon footprint. This can be achieved using Machine Learning.")
            
            raw = read_and_cache_csv('resources/kaggle_train.csv')
            # You can read a markdown file from supporting resources folder
            if st.button("What is Machine Learning"):
                    what_ml = (open('resources/what_is_ML.md').read())
                    st.markdown(what_ml, unsafe_allow_html=True)

            ml_image = Image.open("resources/imgs/ml_pic.jpg")
            st.image(ml_image, use_column_width=True)

            # to add info on machine learning here
            if st.button("How does the app work"):
                    app_info = markdown(open("resources/info.md").read())
                    st.markdown(app_info, unsafe_allow_html=True)

            st.subheader("Description of Sentiment Classes")
            descrip_image = Image.open("resources/imgs/climate_data_sentiment_description.png")
            st.image(descrip_image, use_column_width=True)

            st.subheader("Raw Twitter data and label")
            if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
                    st.write(raw)  # will write the df to the page

    # Building out the predication page
    if selection == "Prediction":

            raw = read_and_cache_csv('resources/kaggle_train.csv')
            st.info("Climate Change belief with ML Models utilising NLP")
            if st.button("What is NLP?"):

                    what_nlp = markdown(open("resources/what_is_nlp.md").read())
                    st.markdown(what_nlp, unsafe_allow_html=True)
                    raw = pd.read_csv("resources/train.csv")

                    nlp_img = Image.open('resources/imgs/nlp_pipeline_img.png')
                    st.image(nlp_img, use_column_width=True)

            # Detect and remove duplicate rows
            raw = raw.drop_duplicates(subset=['message'])

            # Creating a text box for user input
            tweet_text = st.text_area("Enter Text", "Type Here")

            # Apply clean_text function
            tweet_text = clean_text(tweet_text)

            # Remove stop-words
            stop_words = stopwords.words('english')  # Assign stop_words list

            def remove_stopword(text):
                    return [word for word in text.split() if word not in stop_words]
            tweet_text = remove_stopword(tweet_text)

            # Join text
            def join_text(text):
                    text = ' '.join(text)
                    return text
            tweet_text = join_text(tweet_text)

            models_dict = {'Linear Support Vector Classifier': LSVM,
                           'Non-Linear Support Vector Classifier': NL_SVM_TF1,
                           'Logistic Regression CV': LRCV,
                           'Logistic Regression TFiDF': LR_TF2}

            choice = st.selectbox("Please choose a Classification Model",
                                  list(models_dict.keys()))
            
            model = models_dict.get(choice)

            mod_vect_dict = {LSVM: CV_2, NL_SVM_TF1: TF_1, LRCV: CV_2, LR_TF2: TF_2}

            if st.button("Classify"):
                    # Transforming user input with vectorizer
                    vect = mod_vect_dict.get(model)
                    vect_text = vect.transform([tweet_text]).toarray()
                    predictor = model
                    prediction = predictor.predict(vect_text)

                    # When model has successfully run, will print prediction
                    # You can use a dictionary or similar structure to make this output
                    # more human interpretable.
                    pred_labels = {"Anti Climate Change": -1,
                                   "Neutral toward Climate Change": 0,
                                   "Pro Climate Change": 1,
                                   "News about Climate Change": 2}

                    result = get_key(prediction, pred_labels)
                    st.success("Text Categorized as: {}".format(result))

    # Building EDA and Insights page
    # eda = st.sidebar.select()
    if selection == "EDA and Insights":
            st.info('This page is dedicated to Exploratory Data Analysis and insights gained form it.')

            # load data
            raw = read_and_cache_csv("resources/kaggle_train.csv")

            # Adding to sidebar
            st.sidebar.title("EDA and Insights")
            st.sidebar.info('Use the multislect box below to view graphs by sentiment, Insight text applies to graphs with all selected sentiments.')
            sentiment = raw["label"].unique().tolist()
            select_sent = st.sidebar.multiselect('View Analysis by sentiment', sentiment, default=sentiment)

            st.markdown('### **Exploratory Data Aanalysis**')
            st.markdown('When conducting Exploratory Data Analysis, we try and look at the data from all angles, by inspecting and visualising to extract any insights that we can. This can sometimes give surprising results, and as such we try to explore any possible connections, as well as outliers, or any group/class/type that differs from the rest. In this app we will be exploring the distributions of our data from different aspects, combined with what makes it unique, or where the data is strengthened by similarities. <br> In doing so we summarize the main characters of the data and gain insight on what the data can tell us. In this regard get more understanding about what it represents and how to apply it.', unsafe_allow_html=True)
            if st.checkbox("Preview DataFrame"):
                    if st.button("Tail"):
                        st.write(raw.tail())
                    else:
                        st.write(raw.head())

            # Add image description of sentiment
            st.subheader("Description of Sentiment Classes")
            descrip_image = Image.open("resources/imgs/climate_data_sentiment_description.png")
            st.image(descrip_image, use_column_width=True)

            # mask to filter dataframe
            mask_sentiment = raw['label'].isin(select_sent)
            data = raw[mask_sentiment]

            st.markdown('### Data Distribution ###')

            # Sentiment Distribution
            fig, ax = plt.subplots(figsize=(10, 5))
            # graph = sns.countplot(x = 'sentiment', data = raw)
            graph = sns.countplot(x='label', data=data)
            plt.title('Distribution of Sentiment classes count')
            st.pyplot()

            # Insight
            st.markdown('More than half of the tweets , precisely 50,76%, belong to class 1. This indicates that the majority of tweets collected support the belief that man-made climate change exists. Conversely, 8.58% of the tweets collected are class -1, which represents tweets that do not believe in man-made climate change. Tweets that link to factual news about climate change comprise 24,89% whilst tweets which are neutral (neither supports nor refutes the belief of man-made climate change) make up 15,77% of the dataset. These are represented by the classes 2 and 0 respectively.<br>The class imbalance will need to be addressed to avoid the model being biased towards classifying sentiments as the majority class because the model will be well-versed in identifying it.', unsafe_allow_html=True)

            df = eda_data[mask_sentiment]

            st.markdown("### **Visualisations** ###")

            if st.checkbox('View Tweet length distributions'):

                    st.markdown('The first of these explorations will be in the length of various parts of the Tweet body')

                    # generate tweet length graph
                    sent_kde_plots(df, 'tweet_length', 'sentiment')
                    st.pyplot()

                    st.markdown('Looking at the number of words per tweet, although classes 0 and 1 have the same maximum number of words per tweet at 31 words, classes -1 and 1 have the highest average number of words per tweet at ~19 words. This suggests that people that sent out tweets which are anti and pro man-made climate change send out tweets with more words. News tweets generally have the least number of words with a maximum of 26 and an average of ~16 words per tweet. They do however also display more of a normal distribution, insinuating that news tweets are more consistent in the number of words. The number of words of tweets which are classified as neutral have the greatest distribution with a standard deviation of ~6 words, they vary from "few" to "many" words in a tweet.')

                    # generate character count graph
                    sent_kde_plots(df, 'character_count', 'sentiment')
                    st.pyplot()

                    st.write('A similar pattern as established by the number of words per tweet is displayed by the number of characters per tweet. Classes 1 and 0 have the the first and second maximum number of characters per tweet at at 208 and 166 characters respectively. However, classes 1 and -1 have the highest average number of characters per tweet at ~127 and ~124 characters. A slight difference is that class 2 tweets are on average longer than neutral tweets.')

                    # generate punctuation count graph
                    sent_kde_plots(df, 'punctuation_count', 'sentiment')
                    st.pyplot()

                    st.markdown('The amount of punctuaton displays a number of outliers in each class at 36, 25, 58 and 20 for classes -1, 0, 1 and 2 whilst the averages for each class are ~ 8, 7, 8 and 9. There is a miniscule difference in the means therefore the number of punctuation per tweet can not be as an unique identifier for any of the sentiment classes. <br> Despite classes -1 and 0 having tweets which have the most characters and words, the differences between these two classes and the other classes, and additonally themselves, are not significant enough to use these two characteristics as features when classifying between the four classes in question. As mentioned above, there are no punctuation patterns that are significant to either class.', unsafe_allow_html=True)

            st.markdown('### **Wordclouds!** ###')

            # call wordcloud generator
            if st.checkbox('generate wordclouds'):

                    st.markdown('Upon analysis of all the sentiment classes, "climate change", "RT", "https", "co" and "global warming" are the most popular words/phrases. Even within the individual sentiment classes, the same five words/phrases are the most common.')

                    sent = list(df['sentiment'].unique())
                    dft = eda_data.groupby('sentiment')['clean_tweet'].apply(' '.join)
                    for s in sent:
                            fig, ax = plt.subplots()
                            text = dft[s]
                            wordcloud = WordCloud(background_color='white', max_words=100,
                                                  max_font_size=50).generate(text)
                            plt.imshow(wordcloud, interpolation='bilinear')
                            plt.title('Tweets under {} Class'.format(s))
                            plt.axis('off')
                            st.pyplot()
    
    if selection == "Technical":

            ml_img = Image.open("resources/imgs/ml_img.png")
            st.image(ml_img, use_column_width=True)
 
            st.info("Here you will find a little more technical info on the models available for prediction")

            tech_inf = markdown(open('resources/vector_model_exp.md').read())
            st.markdown(tech_inf, unsafe_allow_html=True)

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
