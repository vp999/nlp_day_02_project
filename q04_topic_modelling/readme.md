# Helper function

This assignment comprises implementation of helper function
which outputs the topics obtained after modelling.


## Write a function `q04_topic_modelling` that :
- Makes use of output data from `q02_count_vectorizer_for_LDA` and `q03_LDA` questions.
- Iterates over the return value (i.e Array of component matrix )
from `q03_LDA` .
- Sorts the multidimensional array consisting of feature names




### Parameters:

| Parameter | dtype | argument type | default value | description |
| --- | --- | --- | --- | --- | 
| path | String | compulsory |  | Path of data folder |
| n_top_words | int | optional | 20 | no.of topics  |




### Returns:

| Return | dtype | description |
| --- | --- | --- | 
| variable1 | list | returns topics extracted after modelling |