from math import sin, exp
from scipy.optimize import minimize, differential_evolution
from matplotlib import pylab
import numpy as np

# set an initial function
def initial_function(x):
    return sin(x / 5.0) * exp(x / 10.0) + 5 * exp(-x / 2.0)

def non_smooth_function(x):
    return int(initial_function(x))

# Minimize smooth function
# set initial points x0 as 2, 30 and use minimize, method BFGS
# get X value for x0=2
x_x0_2 = minimize(initial_function, x0=2, method='BFGS')['x'][0].round(2)
# get Y value for x0=2
y_x0_2 = round(minimize(initial_function, x0=2, method='BFGS')['fun'], 2)
# get X value for x0=30
x_x0_30 = minimize(initial_function, x0=30, method='BFGS')['x'][0].round(2)
# get Y value for x0=30
y_x0_30 = round(minimize(initial_function, x0=30, method='BFGS')['fun'], 2)

# Minimize non-smooth function
# set initial points x0 as 30 and use minimize, method BFGS
# get Y value for x0=30
y_ns_x0_30 = round(minimize(non_smooth_function, x0=30, method='BFGS')['fun'], 2)
x_ns_x0_30 = minimize(non_smooth_function, x0=30, method='BFGS')['x'][0]

# Global optimization method
# differential evolution method
# set an interval for research = (1, 30)
y_dif_ev_1_30 = round(differential_evolution(initial_function, [(1, 30)])['fun'], 2)

# differential evolution method for non-smooth function
# set an interval for research = (1, 30)
y_ns_dif_ev_1_30 = round(differential_evolution(non_smooth_function, [(1, 30)])['fun'], 2)
x_ns_dif_ev_1_30 = differential_evolution(non_smooth_function, [(1, 30)])['x'][0]

# compare the number of the functions value calculations
min = float(minimize(initial_function, x0=30, method='BFGS')['nfev'])
di_ev = float(differential_evolution(initial_function, [(1, 30)])['nfev'])
print 'compare the number of the functions value calculations min/di_ev: ', round(min/di_ev, 2)

# write down the result for smooth function minimization
with open('w3_answer_1.txt', 'w') as write_down:
    write_down.write(str(y_x0_2) + ' ' + str(y_x0_30))
# write down the result for differential evolution method
with open('w3_answer_2.txt', 'w') as write_down:
    write_down.write(str(y_dif_ev_1_30))
# write down the result for non-smooth function minimization
# and non-smooth function differential evolution
with open('w3_answer_3.txt', 'w') as write_down:
    write_down.write(str(y_ns_x0_30) + ' ' + str(y_ns_dif_ev_1_30))

# draw initial function
# set X values
x_values = np.arange(1.0, 30.0, 0.1)
# calculate Y values for the initial function
y_i_values = [initial_function(x) for x in x_values]
# find local max & min for f(x)
y_i_min = 1000000.
x_i_min = 0.
for x in x_values:
    y_i = initial_function(x)
    if y_i < y_i_min:
        y_i_min = y_i
        x_i_min = x

# draw non-smooth function
# calculate Y values for the non-smooth function
y_ns_values = [int(initial_function(x)) for x in x_values]

# functions graphs
pylab.plot(x_values, y_i_values, x_values, y_ns_values)
# set min point for init function
pylab.annotate('init func min', xy=(x_i_min, y_i_min), xytext=(x_i_min + 2, y_i_min - 1.),
            arrowprops=dict(facecolor='blue', shrink=0.05),
            )
# set min point with minimize for x=2
pylab.annotate('minimize smooth x=2', xy=(x_x0_2, y_x0_2), xytext=(x_x0_2 + 2, y_x0_2 + 2.5),
            arrowprops=dict(facecolor='green', shrink=0.05),
            )
# set min point with minimize for x=30
pylab.annotate('minimize smooth x=30', xy=(x_x0_30, y_x0_30), xytext=(x_x0_30 - 2, y_x0_2 - 16),
            arrowprops=dict(facecolor='yellow', shrink=0.05),
            )
# set min point for non-smooth function with differential evolution for x=(1:30)
pylab.annotate('differential evolution non-smooth (1:30)', xy=(x_ns_dif_ev_1_30, y_ns_dif_ev_1_30), xytext=(x_ns_dif_ev_1_30 - 2,
            y_ns_dif_ev_1_30 + 2), arrowprops=dict(facecolor='red', shrink=0.05),
            )
# set min point for non-smooth function with minimize for x=30
pylab.annotate('minimize non-smooth x=30', xy=(x_ns_x0_30, y_ns_x0_30), xytext=(x_ns_x0_30 - 8,
            y_ns_x0_30 + 2), arrowprops=dict(facecolor='pink', shrink=0.05),
            )
pylab.legend(('the initial functions grath', 'the non-smooth grath'), frameon=False)
pylab.show()
