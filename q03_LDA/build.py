# Default imports

from sklearn.decomposition import LatentDirichletAllocation
from greyatomlib.nlp_day_02_project.q02_count_vectorizer_for_LDA.build import q02_count_vectorizer_for_LDA


# Write your solution here :
def q03_LDA(path):
    matrix, feature_names = q02_count_vectorizer_for_LDA(path)
#     vocab = feature_names
    model =LatentDirichletAllocation(n_topics=20, random_state=1,learning_method='batch',max_iter=500)
    model.fit(matrix)
    topic_word = model.components_
    return topic_word
