"""Simple linear regression.

Several functions associated with linear regression.
"""

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt


def best_fit_linear(xs, ys):
    """Return slope."""
    slope = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
             (mean(xs) ** 2 - mean(xs ** 2)))
    intercept = mean(ys) - slope * mean(xs)
    return slope, intercept


def squared_error(ys_orig, ys_line):
    """Return standard error."""
    return sum((ys_line - ys_orig) ** 2)


def coefficient_of_determination(ys_orig, ys_line):
    """Return r squared (correlation coefficient)."""
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    sq_error_regr = squared_error(ys_orig, ys_line)
    sq_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (sq_error_regr / sq_error_y_mean)


# example
xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([3, 4, 4.8, 5, 6.5, 7], dtype=np.float64)
m, b = best_fit_linear(xs, ys)
regression_line_ys = [m * x + b for x in xs]
r_squared = coefficient_of_determination(ys, regression_line_ys)

print('y = {}x + {}'.format(m, b))
print('r^2 =', r_squared)

plt.scatter(xs, ys)
plt.plot(xs, regression_line_ys)
plt.show()
