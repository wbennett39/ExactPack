import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import csv 
from scipy.interpolate import interp1d
from ..solvers.sedov.sedov_similarity_variables import sedov

# import ExactPack solvers
from exactpack.solvers.sedov.doebling import Sedov as SedovDoebling

#### parameters ####
t = 1.0
t2 = 1.0
npts = 200001
#eblast = 0.0673185
eblast = .851072
A = 1.0
rvec = np.linspace(0.0, 1.2, npts)
gamma = 1.4
rho1 = 1
rho2 = (gamma+1)/(gamma-1)*rho1
lambda_2 = 1



#### solver object ####
solver_doebling_pla = SedovDoebling(geometry=1, eblast=eblast,
                                    gamma=gamma, omega=0.)

solution_doebling_pla_1 = solver_doebling_pla(r=rvec, t=t)
solution_doebling_pla_2 = solver_doebling_pla(r=rvec, t=t2)

r = np.array(solver_doebling_pla.r_pnts)
alpha = solver_doebling_pla.alpha

rf = (eblast/A/alpha)**(1/3) * t**(2/3)
rf2 = (eblast/A/alpha)**(1/3) * t2**(2/3)

r2 = lambda_2*(eblast/A/alpha)**(1/3)*t**(2/3)
r3 = lambda_2*(eblast/A/alpha)**(1/3)*t2**(2/3)
U = (2/3)*r2/t
U2 = (2/3)*r3/t2
p2 = 2 / (gamma + 1) * rho1 * U**2
p3 = 2 / (gamma + 1) * rho1 * U2**2


# rr1 = xi*rf


### grab self similar solutions

g_fun = np.array(solver_doebling_pla.g_fun_list)
l_fun  = np.array(solver_doebling_pla.l_fun_list)
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

test_index = np.argmin(np.abs(l_fun-1))


v1 = v[0]
v0 = v[-1]


zs = 0.5 * (gamma-1) * v**2*(v-2/3)/ (2/3/gamma-v)
rr1 = l_fun * rf
rr2 = l_fun * rf2

rho = np.array(g_fun * rho2)
p = h_fun * p2
p_t2 = h_fun * p3
e = p / rho/(gamma-1)
r_eval = rr1[0:-1]
r_eval = np.append(r_eval, 0.0)
r_eval = np.flip(r_eval)
r_eval2 = rr2[0:-1]
r_eval2 = np.append(r_eval2, 0.0)
r_eval2 = np.flip(r_eval2)
npnts2 = 200


r_eval = np.append(r_eval[:], np.linspace(rf, 1.2, npnts2))
r_eval2 = np.append(r_eval2[:], np.linspace(rf2, 1.2, npnts2))
p = np.append(np.flip(p),np.zeros(npnts2))
p_t2 = np.append(np.flip(p_t2),np.zeros(npnts2))
e = np.append(np.flip(e),np.zeros(npnts2))


rho = np.append(np.flip(rho),np.ones(npnts2)*rho1)
p_interp = interp1d(r_eval, p)
p_interp_2 = interp1d(r_eval2, p_t2)
e_interp = interp1d(r_eval, e)
rho_interp = interp1d(r_eval, rho)

pressure = p_interp(rvec)
pressure_2 = p_interp_2(rvec)
density = rho_interp(rvec)
energy = e_interp(rvec)


### plot tests ####

plt.figure(1)
plt.plot(rvec, pressure, "o", mfc = 'none', label = 'pressure')
plt.plot(rvec, pressure_2, "o", mfc = 'none', label = 'pressure')
solution_doebling_pla_1.plot('pressure')
solution_doebling_pla_2.plot('pressure')


plt.legend()
plt.show()

plt.figure(2)
plt.plot(-rvec, density , "o", mfc = 'none', label = "density")
solution_doebling_pla_1.plot('density')
plt.legend()
plt.show()

plt.figure(3)
e = p / rho/(gamma-1)

solution_doebling_pla_1.plot('specific_internal_energy')
plt.semilogy(rvec, energy , "--", label = 'internal energy')
plt.legend()
plt.show()
#
