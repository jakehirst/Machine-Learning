import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#from Feature_gathering.features_to_df import create_df
import pandas as pd
# import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# import keras
# from keras import layers
import math as m
#import tensorflow_probability as tfp

"""  /Users/jakehirst/miniforge3/envs/tf/lib/python3.9  """

""" prepares the dataframe by adding the header and making binary 0 labels = -1 """
def get_df(filename):
    header = ["variance", "skewness", "curtosis", "entropy", "label"]
    dict = {0: 'variance',
            1: 'skewness',
            2: 'curtosis',
            3: 'entropy',
            4: 'label'}
    with open(os.path.join(filename), "r") as f:
        df = pd.read_csv(f, header=None)
    df.rename(columns=dict, inplace=True)
    for row in df.iterrows():
        if(row[1]["label"] == 0):
            df.at[row[0], 'label'] = -1
    return df

""" splits the data into training, testing, labels and features for NN. """
def prepare_data(df_filename, epochs, label_to_predict="label"):
    print("----- PREDICTING " + label_to_predict + " -----")
    np.set_printoptions(precision=3, suppress=True) #makes numpy easier to read with prints

    df = get_df(df_filename)


    """ sampling the dataset randomly """
    train_dataset = df.sample(frac=0.8, random_state=0)
    test_dataset = df.drop(train_dataset.index)
    columns = df.columns
    #sns.pairplot(train_dataset[columns], diag_kind='kde') #doesnt work...
    #print(train_dataset.describe().transpose())
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop(label_to_predict)
    test_labels = test_features.pop(label_to_predict)
    #print(train_dataset.describe().transpose()[['mean', 'std']])
    return [train_features, train_labels, test_features, test_labels, epochs]


def make_NN(train_features, train_labels, test_features, test_labels, epochs, depth, width, activation='relu', show=False):
    #quote from tensorflow:
    """One reason this is important is because the features are multiplied by the model weights. So, the scale of the outputs and the scale of the gradients are affected by the scale of the inputs.
    Although a model might converge without feature normalization, normalization makes training much more stable"""
    """ normalizing features and labels """
    
    if(activation == 'relu'):
        initializer = tf.keras.initializers.HeNormal() #aka He initializer
    elif(activation == 'tanh'):
        initializer = tf.keras.initializers.GlorotNormal() #aka Xavier initializer
        
        
    normalizer = tf.keras.layers.Normalization(axis=-1) #creating normalization layer
    normalizer.adapt(np.array(train_features)) #fitting the state of the preprocessing layer

    numfeatures = len(train_features.columns)

    if(depth == 3):
        model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=numfeatures),
                                    tf.keras.layers.Dense(width, activation = activation, kernel_initializer=initializer),
                                    tf.keras.layers.Dense(width, activation = activation, kernel_initializer=initializer),
                                    tf.keras.layers.Dense(1)])
    elif(depth == 5):
        model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=numfeatures),
                                    tf.keras.layers.Dense(width, activation = activation, kernel_initializer=initializer),
                                    tf.keras.layers.Dense(width, activation = activation, kernel_initializer=initializer),
                                    tf.keras.layers.Dense(width, activation = activation, kernel_initializer=initializer),
                                    tf.keras.layers.Dense(width, activation = activation, kernel_initializer=initializer),
                                    tf.keras.layers.Dense(1)])
    elif(depth == 9):
        model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=numfeatures),
                                    tf.keras.layers.Dense(width, activation = activation, kernel_initializer=initializer),
                                    tf.keras.layers.Dense(width, activation = activation, kernel_initializer=initializer),
                                    tf.keras.layers.Dense(width, activation = activation, kernel_initializer=initializer),
                                    tf.keras.layers.Dense(width, activation = activation, kernel_initializer=initializer),
                                    tf.keras.layers.Dense(width, activation = activation, kernel_initializer=initializer),
                                    tf.keras.layers.Dense(width, activation = activation, kernel_initializer=initializer),
                                    tf.keras.layers.Dense(width, activation = activation, kernel_initializer=initializer),
                                    tf.keras.layers.Dense(width, activation = activation, kernel_initializer=initializer),
                                    tf.keras.layers.Dense(1)])
        
    model.compile(
        optimizer = tf.keras.optimizers.Adam(),
        loss='mean_absolute_error')
    
    history = model.fit(tf.expand_dims(train_features, axis=-1), 
                                       train_labels, 
                                       epochs=epochs,
                                       validation_split = None,  
                                       callbacks=[
                                            tf.keras.callbacks.EarlyStopping(
                                                monitor='loss',
                                                patience=50,
                                                restore_best_weights=True
                                            )
                                        ]
                                       )

    print("minimum MAE: ")
    print(min(history.history['loss']))
    
    
    plt.plot(history.history['loss'], label='loss (mean absolute error)')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title("depth = " + str(depth) + " width = " + str(width))
    plt.legend()
    plt.grid(True)
    plt.savefig("/Users/jakehirst/Desktop/Machine_Learning/Homework5/" + "depth=" + str(depth) + "width=" + str(width) + "Min_MAE=" + str(min(history.history['loss'])) + ".jpg", dpi=100)
    plt.show()
    return [history, depth, width]






args = prepare_data("/Users/jakehirst/Desktop/Machine_Learning/SVM/train.csv", epochs=100, label_to_predict="label")

results = []
widths = [5, 10, 25, 50, 100]
for w in widths:
    results.append(make_NN(*args, depth=3, width=w, activation='tanh'))
    results.append(make_NN(*args, depth=5, width=w, activation='tanh'))
    results.append(make_NN(*args, depth=9, width=w, activation='tanh'))



