import matplotlib.pyplot as plt
import scipy.integrate
import scipy.stats
import numpy as np
import math

from variogram.variogram import ExponentialVariogram
from variogram.variogram import GeneralExponentialVariogram
from variogram.variogram import SphericalVariogram
from variogram.simulate import simulate_gaussian_field


def find_corr(x1, x2):
  if False:
      range = 500.0
      d1 = math.pow(x1[0]-x2[0], 2)
      d2 = math.pow(x1[1]-x2[1], 2)
      d = math.sqrt(d1+d2)
      d = d / range
      return(np.exp(-3.0*d))
  else:
      v1_range_x = 800.0
      v1_range_y = 500.0
      # v1_range_z = 20.0
      v1_azimuth = 30.0 * 3.141592 / 180.0 # In radians, not degrees
      v1_genexp_power = 1.5
      v1 = GeneralExponentialVariogram(v1_range_x, v1_range_y, azi=v1_azimuth, power=v1_genexp_power)
      return(v1._corr(dx=x1[0]-x2[0], dy=x1[1]-x2[1]))

p = 0.1
t = scipy.stats.norm.ppf(p)
dx = 40.0  # NBNB-AS: Slightly wrong, a bit less
dy = 40.0  # NBNB-AS: Slightly wrong, a bit less
x0 = [0.0, 0.0]
x1 = [0.0, dy]
x2 = [dx, 0.0]
x3 = [0.0, -dy]
x4 = [-dx, 0.0]
print("proportion blue = " + str(p))
# print("=>            t = " + str(t))

p1 = 1.0 / scipy.stats.norm.cdf(t)
p2 = 1.0 / scipy.stats.norm.cdf(t)
p3 = 1.0 / scipy.stats.norm.cdf(t)
p4 = 1.0 / scipy.stats.norm.cdf(t)
corr1 = find_corr(x1, x0)
corr2 = find_corr(x2, x0)
corr3 = find_corr(x3, x0)
corr4 = find_corr(x4, x0)
K1 = np.array([[1, corr1], [corr1, 1]])
K2 = np.array([[1, corr2], [corr2, 1]])
K3 = np.array([[1, corr3], [corr3, 1]])
K4 = np.array([[1, corr4], [corr4, 1]])
integral1 = scipy.stats.multivariate_normal(mean=[0, 0], cov=K1).cdf(
    np.array([t, np.inf]), lower_limit=np.array([-np.inf, t]))
integral2 = scipy.stats.multivariate_normal(mean=[0, 0], cov=K2).cdf(
    np.array([t, np.inf]), lower_limit=np.array([-np.inf, t]))
integral3 = scipy.stats.multivariate_normal(mean=[0, 0], cov=K3).cdf(
    np.array([t, np.inf]), lower_limit=np.array([-np.inf, t]))
integral4 = scipy.stats.multivariate_normal(mean=[0, 0], cov=K4).cdf(
    np.array([t, np.inf]), lower_limit=np.array([-np.inf, t]))
p1 *= integral1
p2 *= integral2
p3 *= integral3
p4 *= integral4

# TPS:
p_C1_TPS = 1.0 / scipy.stats.norm.cdf(t)
corr01 = find_corr(x0, x1)
corr02 = find_corr(x0, x2)
corr03 = find_corr(x0, x3)
corr04 = find_corr(x0, x4)
corr12 = find_corr(x1, x2)
corr13 = find_corr(x1, x3)
corr14 = find_corr(x1, x4)
corr23 = find_corr(x2, x3)
corr24 = find_corr(x2, x4)
corr34 = find_corr(x3, x4)
K = np.array([[1     , corr01, corr02, corr03, corr04],
              [corr01, 1     , corr12, corr13, corr14],
              [corr02, corr12, 1     , corr23, corr24],
              [corr03, corr13, corr23, 1     , corr34],
              [corr04, corr14, corr24, corr34, 1     ]])
lower_lim = np.array([-np.inf, t, t, t, t])
upper_lim = np.array([t, np.inf, np.inf, np.inf, np.inf])
mean_vec = np.array([0, 0, 0, 0, 0])
integral = scipy.stats.multivariate_normal(mean=mean_vec, cov=K).cdf(
    upper_lim, lower_limit=lower_lim)
p_C1_TPS *= integral

# APS:
t1 = scipy.stats.norm.ppf(1.0-p1)
t2 = scipy.stats.norm.ppf(1.0-p2)
t3 = scipy.stats.norm.ppf(1.0-p3)
t4 = scipy.stats.norm.ppf(1.0-p4)
K = np.array([[1     , corr12, corr13, corr14],
              [corr12, 1     , corr23, corr24],
              [corr13, corr23, 1     , corr34],
              [corr14, corr24, corr34, 1     ]])
lower_lim = np.array([t1, t2, t3, t4])
upper_lim = np.array([np.inf, np.inf, np.inf, np.inf])
mean_vec = np.array([0, 0, 0, 0])
integral = scipy.stats.multivariate_normal(mean=mean_vec, cov=K).cdf(
    upper_lim, lower_limit=lower_lim)
p_C1_APS = integral

print("p_C1_TPS = " + str(p_C1_TPS))
print("p_C1_APS = " + str(p_C1_APS))

exit()






























def find_integral(rho = 0.0):
  return(integrate2(rho)[0] / scipy.stats.norm.cdf(t))

x_obs = 1000.0
x_vec = np.linspace(0.0, 2000.0, num=200)
y = [find_integral(find_corr(x, x_obs)) for x in x_vec]

plt.plot(x_vec, y)
plt.show()
