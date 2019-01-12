import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from wordcloud import WordCloud, STOPWORDS
#from __future__ import print_function
from time import time

def return_something():
    return 200


def return_something2(s):
    return s


def plot_df(cols, filename):
    x = np.random.randn(100)
    ax = sns.distplot(list(cols))
    plt.savefig('static/' + filename)
    return


def get_prediction_lr(df, ls, filename):
    '''
    Description: This function accepts a dataframe of house prices and a user provided list to predict the sales price for
    df: a data frame
    ls: a list containing the information for a house to predict for
    '''

    # Save a plot image
    plot_df(df['SalePrice'], filename)

    # This is a linear regression object
    linreg = LinearRegression()

    # This is our predictors dataframe
    x = df.drop('SalePrice', axis=1)

    # We fit the model using the predictors
    linreg.fit(x, df['SalePrice'])

    # This is our observation we would like to predict. Initially we use the means of all observations
    # and we overwrite the fields which are provided by the user. i.e. the default of a field
    # is the mean for that field
    pred_x = x.mean().values.reshape(1, -1)

    # Loop through the user provided list of fields and update the list for our observation
    # whenever a field is supplied
    for i in range(len(ls)):
        if ls[i] != '':
            pred_x[0][i] = float(ls[i])

    # Return the predicted Sale Price
    return list(linreg.predict(pred_x))[0]


def positivity(x):
    if x > 3:
        return 'positive'
    else:
        return 'negative'


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


def text_process2(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """

    a = stopwords.words('english')
    a.append("can't")

    nostop = [word for word in mess.split() if word.lower() not in a]
    nostopjoined = ' '.join(nostop)

    # Check characters to see if they are in punctuation
    nopunc = [char for char in nostopjoined if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    # Now just remove any stopwords
    return nopunc.split()


def print_top_words(model, feature_names, n_top_words):
    """A function to print the top n_top_words words according to frequency"""
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


def save_top_words(model, feature_names, n_top_words):
    """A function to save the top n_top_words words according to frequency as well as save the
    frequencies themselves"""
    top_words_string = []
    top_words_values = []
    for topic_idx, topic in enumerate(model.components_):
        values = ""
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        values = " ".join(str(topic.data[i]) for i in topic.argsort()[:-n_top_words - 1:-1])

        top_words_string.append(message)
        top_words_values.append(values)
    return top_words_string, top_words_values


def string_contains_term(text, term, delimiter):
    """A function to check if a text contains a given term while splitting on the given delimiter"""
    ls = text.split(delimiter)
    for word in ls:
        if term in word:
            return True

    return False


def add_dicts(dic1, dic2):
    """A function to add two dictionaries together while adding the specific
    values of the keys that exist in both dictionaries"""
    final_dict = dic1
    for word in dic2:
        if word in dic1:
            final_dict[word] = dic1[word] + dic2[word]
        else:
            final_dict[word] = dic2[word]
    return final_dict


def generate_wordcloud(text): # optionally add: stopwords=STOPWORDS and change the arg below
    """A function to create a wordcloud given some text"""
    wordcloud = WordCloud(background_color = 'white',
                          relative_scaling = 1.0,
                          stopwords = {'to', 'of'} # set or space-separated string
                          ).generate(text)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()



def generate_wordcloud_from_freq(frequency_dict): # optionally add: stopwords=STOPWORDS and change the arg below
    """A function to create a wordcloud according to the text frequencies as well as the text itself"""
    wordcloud = WordCloud(background_color = 'white',
                          relative_scaling = 1.0,
                          stopwords = {'to', 'of'} # set or space-separated string
                          ).generate_from_frequencies(frequency_dict)

    return wordcloud


def generate_wordcloud_using_tf_idf(df, interested_text):
    """Input into this function is a single column dataframe of text.
    Can also accept interested text such as 'price'"""

    # These are the necessary parameters
    n_samples = 2000
    n_features = 1000
    n_components = 10
    n_top_words = 20

    print("Loading dataset...")

    # Keep record of time
    t0 = time()

    # Store the relevant data
    data_samples = df['fullreview']

    # Print the time it has taken
    print("done in %0.4fs." % (time() - t0))

    # Use tf-idf features for NMF.
    print("Extracting tf-idf features for NMF...")

    # Get a TfidfVectorizer object with initialised parameters
    # max_df is a threshold which will omit some terms which have a large document frequency
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95,
                                       max_features=n_features,
                                       stop_words='english')

    # Keep record of time
    t0 = time()

    # Fit the TfidfVectorizer to the data
    tfidf = tfidf_vectorizer.fit_transform(data_samples)

    # Print the time it took to fit
    print("done in %0.4fs." % (time() - t0))

    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")

    # Can also use CountVectorizer but it's function is a subset of TfidfVectorizer
    tf_vectorizer = CountVectorizer(max_df=0.95,
                                    max_features=n_features,
                                    stop_words='english')

    # Keep record of time
    t0 = time()

    # Fit the CountVectorizer
    tf = tf_vectorizer.fit_transform(data_samples)

    # Print the time taken
    print("done in %0.4fs." % (time() - t0))
    print()

    # Fit the NMF model with the Frobenius norm variant
    print("Fitting the NMF model with Frobenius norm with tf-idf features, "
          "n_samples=%d and n_features=%d..."
          % (n_samples, n_features))

    # Keep track of time
    t0 = time()

    # Create the NMF object with parameters
    nmf = NMF(n_components=n_components, random_state=1,
              alpha=.1, l1_ratio=.5).fit(tfidf)

    # Print the time taken
    print("done in %0.4fs." % (time() - t0))

    print("\nTopics in NMF model (Frobenius norm):")

    # Get all the different terms involved (We limited this to 1000 features)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    # Here we print to the console, the top n_top_words. We set this to 20 per topic. The topics are contained
    # in nmf.components_
    print_top_words(nmf, tfidf_feature_names, n_top_words)

    # Whatever we printed above, we save it here
    top_words1, top_values1 = save_top_words(nmf, tfidf_feature_names, n_top_words)

    print("Fitting the NMF model (generalized Kullback-Leibler divergence) with "
          "tf-idf features, n_samples=%d and n_features=%d..."
          % (n_samples, n_features))
    t0 = time()

    # Fit the NMF model with the kullback-leibler variant
    nmf = NMF(n_components=n_components, random_state=1,
              beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
              l1_ratio=.5).fit(tfidf)

    # Print the time taken
    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")

    # Get all the different terms involved (We limited this to 1000 features)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    # Here we print to the console, the top n_top_words. We set this to 20 per topic. The topics are contained
    # in nmf.components_
    print_top_words(nmf, tfidf_feature_names, n_top_words)

    # Whatever we printed above, we save it here
    top_words2, top_values2 = save_top_words(nmf, tfidf_feature_names, n_top_words)

    print("Fitting LDA models with tf features, "
          "n_samples=%d and n_features=%d..."
          % (n_samples, n_features))

    # Now we create a LatentDirichletAllocation object with parameters
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)

    # Keep track of time
    t0 = time()

    # Fit LDA using the countvectorizer
    lda.fit(tf)

    # Print the time taken
    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in LDA model:")

    # Get all the different terms involved (We limited this to 1000 features)
    tf_feature_names = tf_vectorizer.get_feature_names()

    # Here we print to the console, the top n_top_words. We set this to 20 per topic. The topics are contained
    # in nmf.components_
    print_top_words(lda, tf_feature_names, n_top_words)

    # Whatever we printed above, we save it here
    top_words3, top_values3 = save_top_words(lda, tf_feature_names, n_top_words)

    full_string = ""
    full_values = ""
    frequency_dict = {}

    j = 0

    # For each topic, check whether it contains the interested text. If it does, get the terms in that topic along
    # with their counts. Then calculate their frequencies
    for i in range(2, len(top_words1)):
        full_string = ""
        full_values = ""
        if string_contains_term(top_words1[i], interested_text, ' '):
            # The first 2 words are like this for each topic: 'Topic' and '#0:'. We don't want these
            full_string = (full_string + ' ' + ' '.join(top_words1[i].split()[2:]))

            # top_values[i] contains the frequency values for the terms in topic i
            full_values = full_values + top_values1[i]

            # Add frequencies across topics
            frequency_dict = add_dicts(frequency_dict,
                                       dict(zip(full_string.split(), list(map(float, top_values1[9].split())))))

    # Here, we generate the actual word cloud for the first NMF model but we don't store it or print it
    generate_wordcloud_from_freq(frequency_dict)

    full_string = ""
    full_values = ""
    frequency_dict = {}

    # For each topic, check whether it contains the interested text. If it does, get the terms in that topic along
    # with their counts. Then calculate their frequencies
    j = 0
    for i in range(2, len(top_words2)):
        full_string = ""
        full_values = ""
        if string_contains_term(top_words2[i], interested_text, ' '):
            # The first 2 words are like this for each topic: 'Topic' and '#0:'. We don't want these
            full_string = (full_string + ' ' + ' '.join(top_words2[i].split()[2:]))

            # top_values[i] contains the frequency values for the terms in topic i
            full_values = full_values + top_values2[i]

            # Add frequencies across topics
            frequency_dict = add_dicts(frequency_dict,
                                       dict(zip(full_string.split(), list(map(float, top_values2[9].split())))))

    # Here, we generate the actual word cloud for the second NMF model but we don't store it or print it
    generate_wordcloud_from_freq(frequency_dict)

    full_string = ""
    full_values = ""
    frequency_dict = {}

    j = 0

    # For each topic, check whether it contains the interested text. If it does, get the terms in that topic along
    # with their counts. Then calculate their frequencies
    for i in range(2, len(top_words3)):
        full_string = ""
        full_values = ""
        if string_contains_term(top_words3[i], interested_text, ' '):
            # The first 2 words are like this for each topic: 'Topic' and '#0:'. We don't want these
            full_string = (full_string + ' ' + ' '.join(top_words3[i].split()[2:]))

            # top_values[i] contains the frequency values for the terms in topic i
            full_values = full_values + top_values3[i]

            # Add frequencies across topics
            frequency_dict = add_dicts(frequency_dict,
                                       dict(zip(full_string.split(), list(map(float, top_values3[9].split())))))

    # Here, we generate the actual word cloud for the LDA model and we return a handle to the plot
    return generate_wordcloud_from_freq(frequency_dict)


def get_word_cloud(reviews,filename):

    # We combine the Review and title columns since titles are also a source of review information and may be more
    # direct
    reviews['fullreview'] = reviews['Review'] + ' ' + reviews['title']

    # We create an outcome column which determines positivity from stars
    reviews['outcome'] = reviews['Rating'].apply(lambda x: positivity(x))

    # Get figure object of the distribution of ratings
    fig = sns.countplot(data=reviews, x='Rating')
    fig = fig.get_figure()

    # Save a plot image of the rating distribution
    fig.savefig('static/' + filename)

    # Get the wordcloud figure and save it
    wc_fig = generate_wordcloud_using_tf_idf(reviews, '')
    wc_fig.to_image().save('static/wc' + filename)
