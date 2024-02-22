import numpy as np
import scipy.stats as stats

ranges = np.array([0,0])


mean = np.array([0.0, 0.0])
cov  = np.array([[0.01,0.0],\
	             [0.0,0.01]])

cdf_values = stats.multivariate_normal.cdf(ranges, mean, cov)
print (cdf_values)
