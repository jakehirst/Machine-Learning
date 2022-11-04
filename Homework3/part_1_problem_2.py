import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
import math as m
import pandas as pd

def distanceToPlane(x_1, x_2, w):
    return abs(w[0]*x_1 + w[1]*x_2) / (m.sqrt(w[0]**2 + w[1]**2))
    

""" runs standard perceptron, starting with a w of zeros... edited from standardperceptron.py such that
    the bias component is not here. I know that the bias should be 0 from the graph shown in the first graph of part A """
def run_standard_perceptron(training_df, r, T):
    w = np.zeros(len(training_df.columns)-1)
    #training_df.insert(0, 'b', np.ones(len(training_df)))
    #changing all the labels that are 0 to 1
    #training_df = morph_labels(training_df)
    for t in range(T):
        #shuffle the data
        training_df = training_df.sample(frac=1).reset_index(drop=True)
        feature_matrix = np.array(training_df.drop("label", axis=1))
        labels = np.array(training_df["label"])
        for i in range(len(labels)):
            #only change w if the guess is not equal to the label
            guess = np.matmul(w,feature_matrix[i])
            if(not np.sign(guess) == labels[i]):
                w = w + r * labels[i] * feature_matrix[i]
    return w


#part A
print("\n ---  Part A --- ")
# x_1 = np.array([[-1,0,1,0]])
# x_2 = np.array([[0,-1,0,1]])
# labels = np.array([[-1,-1,1,1]])

positive = np.array([[1,0] , [0,1]])
negative = np.array([[-1,0] , [0,-1]])

plt.plot(positive[:,0], positive[:,1], 'bo')
plt.plot(negative[:,0], negative[:,1], 'ro')
print("Because of the graph shown, we should be able to calculate the margin as the dataset looks linearly seperable.\n")
plt.show()

data = {"x_1": [-1,0,1,0],
        "x_2": [0,-1,0,1],
        "label": [-1,-1,1,1]}
df = pd.DataFrame(data, columns = ["x_1", "x_2", "label"])

w = run_standard_perceptron(df, 0.1, 10)

x_1 = np.linspace(-10, 10, 100)
x_2 = np.linspace(-10, 10, 100)

x_1, x_2 = np.meshgrid(x_1, x_2)
eq = w[0]*(x_1) + w[1]*(x_2)

fig = plt.figure()

ax = fig.gca(projection='3d')

ax.plot_surface(x_1, x_2, eq)
ax.plot(positive[:,0], positive[:,1], "bo")
ax.plot(negative[:,0], negative[:,1], "ro")
print("\nThis 3D plot (click and drag to adjust view) shows how the w vector creates a plane that splits the positive and negative labels very clearly. ")
plt.show()

#calculating margin for the weight vector w 
x = np.array([[-1,0], [0,-1], [1,0], [0,1]])
margin = None
for i in range(len(x)):
    distance = distanceToPlane(x[i][0], x[i][1], w)
    if(margin == None or margin > distance):
        margin = distance
    print(distance)
print("margin = " + str(margin))

#you need to find the hyperplane that contains the maximum margin, and that is your margin








#part B
print("\n ---  Part B --- ")
x_1 = np.array([[-1,0,1,0]])
x_2 = np.array([[0,-1,0,1]])
labels = np.array([[-1,-1,1,1]])

positive = np.array([[0,1] , [0,-1]])
negative = np.array([[-1,0] , [1,0]])

plt.plot(positive[:,0], positive[:,1], 'bo')
plt.plot(negative[:,0], negative[:,1], 'ro')
print("From the graph shown, you can see that this dataset is not linearly seperable, therefore the margin of the dataset does not exist.")
plt.show()
