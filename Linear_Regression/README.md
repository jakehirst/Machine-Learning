LMS

To run LMS algorithm, you must first split the dataset into training and testing datasets. You can use SplitTrainingAndTesting() for this, with the first parameter being the original dataframe, and the second being the number of examples you want in your training dataframe. 

Then, you can just call RunBatchedGradientDecent() or RunStochasticGradientDecent() with both having the same parameters. The first being the Training dataframe, the second being the tolerance level of the LMS method, and finally the third being the learning rate r.