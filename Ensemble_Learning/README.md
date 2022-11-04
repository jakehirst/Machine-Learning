Adaboost

To run Adaboost, you must first prep the training data by weighting the examples and binarizing certain columns. You do this by creating an ID3 object, and then calling prepDataquickly(filenames, "c", columns_to_binarize). Where filenames is the training and test filenames respectively, "c" is the way to weight the examples by fractional counts, and columns_to_binarize being the columns in both the test and training datasets that you'd like to be binarized.

Then, you must drop the "Unnamed: 0" and "index" axes from both the test and training dataframes. 

Finally, you must assign a weight array column of 1's to the test and training data.

Then you are ready to call AdaBoost, with the first parameter being the training data, the second being the way to calculate information gain ("Entropy" is good here), the third being the Testing dataframe, and finally the 4th being the number of rounds that adaboost goes for.


Random Forest

To run Random forest, you just have to run Get_Forest() with 6 different parameters. The first parameter is the TrainingFilename, which is the filename of where to get your training data. The second is the max_number_of_trees which is how many trees you would like to have in your forest. The third is the m_prime parameter, which is the number of examples given subsets while training. The fourth is the InfoGainMethod, which again could be "Entropy" if you would like to keep things simple. The fifth is the columns_to_binarize, which are the columns in your dataset that you would like to have as binary features instead of categorical or continuous. Finally, the num_random_attributes parameter is the number of attributes you would like to randomly select during ID3 to be able to more randomly choose attributes to split on.

Bagging

To run the Bagging algorithm, you just need to run GetBaggedTrees() which has 5 attributes. The first parameter is the TrainingFilename, which is the filename of where to get your training data. The second is the max_number_of_trees which is how many trees you would like to have in your forest. The third is the m_prime parameter, which is the number of examples given subsets while training. The fourth is the InfoGainMethod, which again could be "Entropy" if you would like to keep things simple. The fifth is the columns_to_binarize, which are the columns in your dataset that you would like to have as binary features instead of categorical or continuous. This is very much like the random forest algorithm, but without the number of random attributes. 