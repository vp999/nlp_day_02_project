# Count Vectorization

Your job is to transform the data using count vectorizer.


## Write a function `q02_count_vectorizer_for_LDA` that :
- Makes use of output data from `q01_load_data_tfidf` question.
- Fits and transforms on the training data using `CountVectorizer`.
- Extracts the feature names after fitting with count vectorizer.




### Parameters:

| Parameter | dtype | argument type | default value | description |
| --- | --- | --- | --- | --- | 
| path | String | compulsory |  | Path of data folder |




### Returns:

| Return | dtype | description |
| --- | --- | --- | 
| variable1 | scipy.sparse.csr.csr_matrix | Output Matrix after applying count vectorizer |
| variable2 | list | Names of the features obtained after transforming with count vectorizer |


Note : While using `CountVectorizer` use the following parameters
- analyzer as 'word'
- ngram_range as (1, 1) 
- min_df as 0
- stop_words as 'english'
