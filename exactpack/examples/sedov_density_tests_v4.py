import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import csv 
from scipy.interpolate import interp1d
from ..solvers.sedov.sedov_similarity_variables import sedov
import scipy.integrate as integrate
from numba import njit, types, prange
import ctypes
from numba.extending import get_cython_function_address


# import ExactPack solvers
from exactpack.solvers.sedov.doebling import Sedov as SedovDoebling

#### parameters ####
t = 0.5
t2 = 1.5
npts = 2000001
eblast = 0.0673185
A = 1.0
rvec = np.linspace(0.0, 1.2, npts)
gamma = 1.4
rho1 = 1
rho2 = (gamma+1)/(gamma-1)*rho1
lambda_2 = 1

evaluate_lambda = np.linspace(1, 0, npts)

#### get nondim functions ####
solver_doebling_pla = SedovDoebling(geometry=1, eblast=eblast,
                                    gamma=gamma, omega=0.)

solution_doebling_pla_1 = solver_doebling_pla(r=rvec, t=t)

r = np.array(solver_doebling_pla.r_pnts)
alpha = solver_doebling_pla.alpha

rf = (eblast/A/alpha)**(1/3) * t**(2/3)

# U = (2/3)*rf/t

# p2 = 2 / (gamma + 1) * rho1 * U**2


l_fun  = np.array(solver_doebling_pla.l_fun_list)
g_fun  = np.array(solver_doebling_pla.g_fun_list)
h_fun = np.array(solver_doebling_pla.h_fun_list)
v = (np.array(solver_doebling_pla.vlist))

found_zero = False
iterator = 0
while found_zero == False:
    if g_fun[iterator] == 0:
        iterator += 1
    else:
        found_zero = True
g_fun = g_fun[iterator:]
h_fun = h_fun[iterator:]
l_fun = l_fun[iterator:]
v = v[iterator:]

l_fun_faked = np.append(l_fun[0:-1], 0.0)

l_fun_faked[0] = 1.0

g_fun_interpolated = interp1d(l_fun_faked, g_fun)

g_fun_evaluated= g_fun_interpolated(evaluate_lambda)

plt.plot(evaluate_lambda, g_fun_evaluated)
plt.show()





