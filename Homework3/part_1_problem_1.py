import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
import math as m

#use this and just check all the points to get the minimum margin for part 1
def distanceToPlane(x_1, x_2):
    return abs(2*x_1 + 3*x_2 - 4)/ m.sqrt(2**2 + 3**2)


#part A
print("\n --- Part A --- ")
w = np.array([[-4,2,3]])
u = w / numpy.linalg.norm(w)
y = np.array([1,-1,-1,1])
x = np.array([[1,1,1], [1,1,-1], [1,0,0], [1,-1,3]])

margin = None
for i in range(len(x)):
    distance = distanceToPlane(x[i][0], x[i][1])
    if(margin == None or margin > distance):
        margin = distance
    #print(distanceToPlane(x[i][0], x[i][1]))

print("margin = " + str(margin))
    
    
#part B
print("\n --- Part B --- ")
positive_points = np.array([[1,1], [-1,3], [-1,-1]])
negative_points = np.array([[1,-1], [0,0]])
x_1 = np.linspace(-10, 10, 100)
x_2 = np.linspace(-10, 10, 100)

#x_1, x_2 = np.meshgrid(x_1, x_2)
#eq = 2*(x_1) + 3*(x_2) - 4

#fig = plt.figure()

# ax = fig.gca(projection='3d')

# ax.plot_surface(x_1, x_2, eq)
# ax.plot(positive_points[:,0], positive_points[:,1], "bo")
# ax.plot(negative_points[:,0], negative_points[:,1], "ro")


plt.plot(positive_points[:,0], positive_points[:,1], "bo")
plt.plot(negative_points[:,0], negative_points[:,1], "ro")
print("from the plot shown, you can clearly see that the dataset is not linearly seperable, therefore there is no margin. ")
plt.show()
