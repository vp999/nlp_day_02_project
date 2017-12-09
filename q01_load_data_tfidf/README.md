# Tf -idf

This assignment comprises implementation loading data and implementing the 
advanced techniques of natural language processing where in involves calculation
of Tf -idf (term frequency–inverse document frequency)


## Write a function `q01_load_data_tfidf` that :
- Loads the data from the given path.
- Computes the `tf-idf` using function `TfidfVectorizer`.
- Fits and transform the feature `talkTitle` of data set.




### Parameters:

| Parameter | dtype | argument type | default value | description |
| --- | --- | --- | --- | --- | 
| path | String | compulsory |  | Path of data folder |
| max_df | float | optional | 0.5 | ignore terms freq that are higher than max_freq |
| min_df | int | optional | 2 | ignore terms freq that are lower than min_freq |
| no_features | int | optional | 1000 | the no. of features to extract from data |




### Returns:

| Return | dtype | description |
| --- | --- | --- | 
| variable1 | pd.core.frame.DataFrame | Data set|
| variable2 | scipy.sparse.csr.csr_matrix | computed tf-idf scores |
| variable3 | list | Names of the features obtained after tf-idf |
