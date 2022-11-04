import matplotlib.pyplot as plt


positive_x = [1, 1.75]
positive_y = [1, -1]

negative_x = [1.25, 1.5]
negative_y = [-2, 2]

plt.plot(positive_x, positive_y, 'bo')
plt.plot(negative_x, negative_y, 'ro')
plt.show()