# Default imports

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Write your solution here :
def q01_load_data_tfidf (path, max_df = 0.95, min_df = 2,no_features =1000):
    df =pd.read_csv(path)

    #df =pd.read_csv(path, encoding = "ISO-8859-1")
    #df=df.fillna('')
    train= df['talkTitle']

    vect = TfidfVectorizer()
    #from nltk.tokenize import TreebankWordTokenizer
    #tokenizer = TreebankWordTokenizer()
    #vect.set_params(tokenizer=tokenizer.tokenize)

    # remove English stop words
    vect.set_params(stop_words='english')

    # include 1-grams and 2-grams
    #vect.set_params(ngram_range=[1,2])

    # ignore terms that appear in more than 50% of the documents
    vect.set_params(max_df=max_df)

    # only keep terms that appear in at least 2 documents
    vect.set_params(min_df=(min_df/df.shape[0]))
    vect.set_params(max_features =no_features)

    vect =TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=no_features, stop_words='english')
    train_vectors = vect.fit_transform(train)

    return df, train_vectors, vect.get_feature_names()

def q01_load_data_tfidf_ga(path, max_df=0.95, min_df=2, no_features=1000):
    dataset = pd.read_csv(path)
    tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=no_features, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(dataset['talkTitle'])
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    return dataset, tfidf, tfidf_feature_names

#path = 'data/sessions.csv'
#q01_load_data_tfidf(path)
