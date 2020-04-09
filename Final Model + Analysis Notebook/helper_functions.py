import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re as re
import scipy.stats as stats
import math
from textblob import TextBlob
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression
import scipy.stats as stats
import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Activation, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from nltk import word_tokenize


def findlength(dataframe, column_name):
    
    """
    findLength returns the length of text entries

    Parameters
    ----------
    dataframe : DataFrame
        A pandas DataFrame object. 
    column_name: String
        Name of the DataFrame column that contains text

    Returns
    -------
        Length of text entries
    """

    length_te = [len(str(words)) for words in dataframe[column_name]]
    
    return length_te

def null_plot(df, kind):
    
    """
    null_plot plots null counts of each column of the DataFrame

    Parameters
    ----------
    df : DataFrame
        A pandas DataFrame object. 
    kind : Plot kind
        The kind of plot i.e. bar

    Returns
    -------
        Plot of chosen kind of null values in DataFrame.
    """

    null_df = pd.DataFrame(columns=['column_name', 'null_counts'])
    null_df['column_name'] = df.columns
    null_df['null_counts'] = df.isnull().sum().values
    #df.isnull().sum().values
    null_df.sort_values(by=['null_counts'], ascending=True, inplace=True)
    null_df.plot(x='column_name', y='null_counts', kind=kind)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    print(df.isnull().sum().sort_values(ascending=False))
    plt.show()
    
    

def preprocess(article_text):
    
    """
    preprocess performs text preprocessing

    Parameters
    ----------
    article_text : DataFrame column
        A column containing text. 
    Returns
    -------
        A preprocessed clean text column
    """

    # Remove Line Breaks element
    article_text = article_text.str.replace("(<br/>)", "")
    # Remove NewLine element
    article_text = article_text.str.replace("(\n)", "")
    #  Remove Hyperlink element
    article_text = article_text.str.replace('(<a).*(>).*(</a>)', '')
    # Remove Ampersand 
    article_text = article_text.str.replace('(&amp)', '')
    # Remove greater than sign
    article_text = article_text.str.replace('(&gt)', '')
    # Remove less than sign
    article_text = article_text.str.replace('(&lt)', '')
    # Remove non-breaking space 
    article_text = article_text.str.replace('(\xa0)', ' ')  
    # Remove Emails
    article_text = [re.sub(r"\S*@\S*\s?", '', str(sent)) for sent in article_text]
    # Remove new line characters
    article_text = [re.sub(r"\s+", ' ', sent) for sent in article_text]
    # Remove distracting single quotes
    article_text = [re.sub("\'", "", sent) for sent in article_text]
    
    return article_text

# helper function to plot distribution along with mean and median of text polarity 
def dist_plot(df, column, title, article_type):
    
    """
    dist_plot plots distribution along with mean and median of text polarity 

    Parameters
    ----------
    df : DataFrame
        A pandas DataFrame object. 
    column : String
        The column containing data for plotting distribution and calculauting mean & median.
    title : String
        The title of the plot.
    article_type : String
        The type of articles whose distribution is being plotted i.e. all articles, real articles or fake articles.
    Returns
    -------
        Distribution Plot with mean and median plotted
    """
    
    plt.figure(figsize=(12,6))
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    sns.set_context("paper")
    sns.distplot(df[column], bins=35,hist=True)
    plt.title(title)
    real_tpmean = df[column].mean()
    real_tpmedian = df[column].median()
    print('The mean of text polarity in ' + article_type + ' is: ' + str(round(real_tpmean,3)))
    print('The median of text polarity in ' + article_type + ' is: ' + str(round(real_tpmedian,3)))
    plt.axvline(real_tpmean, color='r',alpha=0.5)
    plt.axvline(real_tpmedian, color='g',alpha=0.8)
    plt.show()
    
# Function to get top N_words
""" Following three functions were sourced from 
https://github.com/susanli2016/NLP-with-Python/blob/master/EDA%20and%20visualization%20for%20Text%20Data.ipynb"""

def get_top_n_words(corpus, n=None):
    
    """
    get_top_n_words gets the top n unigrams from the text corpus

    Parameters
    ----------
    corpus : DataFrame column
        A pandas DataFrame column. 
    n : Integer
        The number of top unigrams you want to return i.e. 10, 20.
    Returns
    -------
        List of top n unigrams
    """
    
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def get_top_n_words_nosw(corpus, n=None):
    
    """
    get_top_n_words_nosw gets the top n unigrams from the text corpus after stopwords are removed

    Parameters
    ----------
    corpus : DataFrame column
        A pandas DataFrame column. 
    n : Integer
        The number of top unigrams you want to return i.e. 10, 20.
    Returns
    -------
        List of top n unigrams
    """
    vec = CountVectorizer(stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def get_top_n_bigram(corpus, n=None):
    
    """
    get_top_n_bigram gets the top n bigrams from the text corpus

    Parameters
    ----------
    corpus : DataFrame column
        A pandas DataFrame column. 
    n : Integer
        The number of top bigrams you want to return i.e. 10, 20.
    Returns
    -------
        List of top n bigrams
    """
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def get_top_n_bigram_nosw(corpus, n=None):
    
    """
    get_top_n_bigrams_nosw gets the top n bigrams from the text corpus after stopwords are removed

    Parameters
    ----------
    corpus : DataFrame column
        A pandas DataFrame column. 
    n : Integer
        The number of top bigrams you want to return i.e. 10, 20.
    Returns
    -------
        List of top n bigrams
    """
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def get_top_n_trigram(corpus, n=None):
        
    """
    get_top_n_trigram gets the top n trigrams from the text corpus

    Parameters
    ----------
    corpus : DataFrame column
        A pandas DataFrame column. 
    n : Integer
        The number of top trigrams you want to return i.e. 10, 20.
    Returns
    -------
        List of top n trigrams
    """

    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def get_top_n_trigram_nosw(corpus, n=None):

    """
    get_top_n_bigrams_nosw gets the top n trigrams from the text corpus after stopwords are removed

    Parameters
    ----------
    corpus : DataFrame column
        A pandas DataFrame column. 
    n : Integer
        The number of top trigrams you want to return i.e. 10, 20.
    Returns
    -------
        List of top n trigrams
    """

    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

# Train logistic regression classifier and its roc-auc score, accuracy, confusion matrix & classification report.
def log_reg_classifier(model, cw, c, xtrain, ytrain, xtest, ytest):

    """
    log_reg_classifier trains, tests and evaluates a Logistic Regression Classifier

    Parameters
    ----------
    model : Model Object
        Initialize call to Logistic Regression model i.e. LogisticRegression()
    cw : String
        Class weight model parameter setting.
    c : Integer
        C model parameter setting.
    xtrain: DataFrame
        The training dataframe containing feature variables.
    ytrain: DataFrame
        The training dataframe containing predictions.
    xtest: DataFrame
        The test dataframe containing feature variables.
    ytest: DataFrame
        The test dataframe containing predicitons. 
    
    Returns
    -------
        The model's AUC-ROC score, accuracy, confusion matrix and classification report. 
    """

    log_reg = model(class_weight=cw, C=c)
    log_reg.fit(xtrain,ytrain)
    print("\n\n -- Logistic Regression Model --")
    log_reg_auc_score = roc_auc_score(ytest,log_reg.predict(xtest))
    print("-- Logistic Regression Model AUC = %2.2f --" % log_reg_auc_score)
    score_lr_cv = metrics.accuracy_score(ytest,log_reg.predict(xtest))
    print("-- Logistic Regression Model with Count Vectorizer Accuracy = %2.2f --" % score_lr_cv)
    cm = metrics.confusion_matrix(ytest,log_reg.predict(xtest), labels=[0, 1])
    print(cm)
    print(classification_report(ytest, log_reg.predict(xtest)))

# Train Multinomial Naive Bayes classifier and its roc-auc score, accuracy, confusion matrix & classification report.
def mnb_classifier(model, alpha, xtrain, ytrain, xtest, ytest):

    """
    mnb_classifier trains, tests and evaluates a Multinomial Naive Bayes Classifier

    Parameters
    ----------
    model : Model Object
        Initialize call to Multinomial Naive Bayes model i.e. Multinomial Naive Bayes()
    alpha : Integer
        alpha model parameter setting.
    xtrain: DataFrame
        The training dataframe containing feature variables.
    ytrain: DataFrame
        The training dataframe containing predictions.
    xtest: DataFrame
        The test dataframe containing feature variables.
    ytest: DataFrame
        The test dataframe containing predicitons. 
    
    Returns
    -------
        The model's AUC-ROC score, accuracy, confusion matrix and classification report. 
    """

    nb_classifier = model(alpha=alpha)
    nb_classifier.fit(xtrain, ytrain)
    pred = nb_classifier.predict(xtest)
    print("\n\n -- Multinomial NB Model with Count Vectorizer --")
    print("\n")
    mnb_cv_auc_score = roc_auc_score(ytest,pred)
    print("-- Multinomial NB Model with Count Vectorizer AUC = %2.2f --" % mnb_cv_auc_score)
    score = metrics.accuracy_score(ytest,pred)
    print("-- Multinomial NB Model with Count Vectorizer Accuracy = %2.2f --" % score)
    print("\n")
    cm = metrics.confusion_matrix(ytest,pred, labels=[0, 1])
    print(cm)
    print("\n")
    print(classification_report(ytest, pred))
    
