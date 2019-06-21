from math import sin, exp
from scipy import linalg
from matplotlib import pylab
import numpy as np


def initial_function(x):
    return sin(x / 5.0) * exp(x / 10.0) + 5 * exp(-x / 2.0)

def polynom_1(w, x):
    return w[0][0] + w[1][0] * x

def polynom_2(w, x):
    return w[0][0] + w[1][0] * x + w[2][0] * (x ** 2)

def polynom_3(w, x):
    return w[0][0] + w[1][0] * x + w[2][0] * (x ** 2) + w[3][0] * (x ** 3)
# BONUS
def polynom_4(w, x):
    return w[0][0] + w[1][0] * x + w[2][0] * (x ** 2) + w[3][0] * (x ** 3) + w[4][0] * (x ** 4)


# calculate the first degree polynom
a_1 = np.array([[1, 1], [1, 15]])
b_1 = np.array([[initial_function(1)], [initial_function(15)]])
w_1 = linalg.solve(a_1, b_1)

# calculate the second degree polynom
a_2 = np.array([[1, 1, 1], [1, 8, 64], [1, 15, 225]])
b_2 = np.array([[initial_function(1)], [initial_function(8)], [initial_function(15)]])
w_2 = linalg.solve(a_2, b_2)

# calculate the third degree polynom
a_3 = np.array([[1, 1, 1, 1], [1, 4, 16, 64], [1, 10, 100, 1000], [1, 15, 225, 3375]])
b_3 = np.array([[initial_function(1)], [initial_function(4)], [initial_function(10)], [initial_function(15)]])
w_3 = linalg.solve(a_3, b_3)

# calculate the fourth degree polynom
a_4 = np.array([[1, 1, 1, 1, 1], [1, 4, 16, 64, 256], [1, 8, 64, 512, 4096],
                [1, 12, 144, 1728, 20736], [1, 15, 225, 3375, 50625]])
b_4 = np.array([[initial_function(1)], [initial_function(4)], [initial_function(8)],
                [initial_function(12)], [initial_function(15)]])
w_4 = linalg.solve(a_4, b_4)

# write down the result
with open('answer2.txt', 'w') as write_down:
    write_down.write(str(w_3[0, 0]) + ' ' + str(w_3[1, 0]) + ' ' + str(w_3[2, 0]) + ' ' + str(w_3[3, 0]))

# set X values
x_values = np.arange(1.0, 15.0, 0.1)
# calculate Y values for the initial function
y_i_values = [initial_function(x) for x in x_values]
# calculate Y values for the first degree polynomial
y_1_values = [polynom_1(w_1, x) for x in x_values]
# calculate Y values for the second degree polynomial
y_2_values = [polynom_2(w_2, x) for x in x_values]
# calculate Y values for the third degree polynomial
y_3_values = [polynom_3(w_3, x) for x in x_values]

# BONUS 
# calculate Y values for the fourth degree polynomial
y_4_values = [polynom_4(w_4, x) for x in x_values]

# functions graphs
pylab.plot(x_values, y_i_values, x_values, y_1_values, x_values, y_2_values,
           x_values, y_3_values, x_values, y_4_values)
pylab.legend(('the initial functions grath', 'the first degree polynomial',
              'the second degree polynomial', 'the third degree polynomial',
              'the fourth degree polynomial'), frameon=False)
pylab.show()
