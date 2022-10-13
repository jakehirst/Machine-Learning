This README contains detailed instructions on how to learn decision trees with my code.

** This code works with slightly pre-processed datasets, where I included the attribute names as the first row of the csv. I have included the pre-processed datasets in the DecisionTree folder. **

Preprocess data:
\\To preprocess the data, you must follow the code in the " DATA PREPROCESSING " section of ID3.py. data is the learning data, and Testdf is the test data. 

If you would like to fill in missing attributes, call FillMissingAttributes(), with the first parameter being the dataframe you want to fill, the second parameter being the indentifier of what a cell in the csv will have in it if it is missing, and the third parameter being the way that you fill the missing attribute ("a" = Most Common value of the Attribute, "b" = the most common value of the attribute with rows that have the same label, "c" = Fractional Counts)

If you would like to binarize the numeric values in your dataset, you must first identify the columns that you would like to be binarized in a list as shown in the beginning of the " DATA PREPROCESSING " section. Then, you must call binarize_numeric_vals() with the first parameter being the dataframe you would like to binarize, and the second parameter being the list of attributes you would like to binarize.

Finally, you must identify the possible labels in the dataset as a global list. This is the last step in the " DATA PREPROCESSING " section and can be done as shown.

ALL OF THE PREPROCESS DATA STEPS MUST BE DONE FOR BOTH THE TRAINING AND TESTING DATASETS 

Running ID3 algorithm:
\\After the preprocessing is done, you can run the ID3 algorithm with runID3(), the first parameter being the preprocessed training dataset, the second parameter being the method by which you calculate information gain (string), the third being the Maximum depth you will allow for the decision tree to grow, and lastly the fourth parameter being the preprocessed Test dataframe. runID3() will not only create a decision tree for you, but will also run the code to test it agains the preprocessed Test dataframe and print out the results.

I have also included some multiprocessing in case you want to quickly run the code with a multitude of MaxDepth parameters. This is shown in the """ Running ID3 with multiple MaxDepths """ section.


