### **Vectorization**
Word vectorisation is, in essence, taking a collection of words and through various methods transforming them into numerical features that a Machine Learning Modle can use to be trained on the context of the words. When pcombined with a 'label' or 'class' corresponding to the text, The model can then predict an unlabled piece of text and classify it into the categories that it has 'learned'

#### **Getting a bit more technical**
The Models that we have provided in the app have made use of either of the following Vectorizers.<br>  
* TFIDF Vectorizer<br>
* CountVectorizer<br>
>At its base a CountVectorizer Counts the frequency that words appear in the text and increases the value of the word the more often it appears,
where a TFIDF vectorizer, or *Term Frequency Inverse Document Frequency*, also increases the value of words proportionally to their count, but is inversely proportional to the frequency of the word in the corpus(document), explaining the Inverse Document Frequency part.
The IDF adjusts for the fact that some words appear more commonly in general, and is a bit more sensitive to unique words that appear more frequent in a specific corpus.<br>
 Additionally these vectorizers have many hyperparameters that add much more functionality than that just mentioned, but lets not get too deep into technicalities.

### **Models used**
>To see if there is a difference between the vectorizers, We have included 2 Logistic Regression models, using a CountVectorizer and the other a TFIDF vectorizer.
**Logistic Regression** uses a log odds ratio to predict the probability of a class belonging to a specific category.<br>
**Support Vector Classifiers** make use of a hyperplane which acts like a decision boundary between the various classes.
Linear Support Vector Classifiers use a straight 'line' to seperate the data into categories, where non-linear Classifiers will transform data into another dimension so that a separation of the data can be achieved.
That is as deep as we will go into the explanation of the models, lest it gets too difficult to follow.


