DualSVM:

To run DualSVM, simply load a dataframe with get_df(filepath), 

optimize alpha with get_alpha(), whose inputs are the dataframe, C constant that you choose, the type of kernel trick you would like to use, and gamma,

get the weight vector with get_w() whose inputs are the dataframe, the optimized alpha, the type of kernel trick, and gamma,

get the bias term beta by calling get_beta() whose inputs are the dataframe and the learned weight vector w.

Then to get the error of your model, simply call get_error(), which has inputs of the dataframe youd like to test against, the weight vector w, the learned bias term beta, the kernel trick you used, and gamma.

stochastic_sub:

To run stochastic subgradient descent SVM, simply...

load a dataframe with get_df(filepath), 

and call stoch_sub_grad_desc() whose inputs are the dataframe you want to learn, the maximum number of epochs, C constant that you choose, the method of getting your learning rate, gamma_0, and a if using learning rate method 1.