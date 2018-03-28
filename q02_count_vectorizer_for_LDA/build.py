# Default imports

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.nlp_day_02_project.q01_load_data_tfidf.build import q01_load_data_tfidf

from sklearn.feature_extraction.text import CountVectorizer

# Write your solution here:
def q02_count_vectorizer_for_LDA (path):
    df, vectors ,features =  q01_load_data_tfidf(path)
    vect = CountVectorizer(analyzer='word', ngram_range=(1,1), min_df=0, stop_words='english')
    train_vectors = vect.fit_transform(df['talkTitle'])
    return train_vectors, vect.get_feature_names()
