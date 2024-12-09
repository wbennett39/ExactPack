'''Example demonstrating Sedov solvers. Reproduces plots from Kamm & Timmes,
"On Efficient Generation of Numerically Robust Sedov Solutions," LA-UR-07-2849

Uses Doebling and (if available) Timmes Sedov solvers.
'''


# import standard Python packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import csv 
from scipy.interpolate import interp1d
from ..solvers.sedov.sedov_similarity_variables import sedov

# import ExactPack solvers
from exactpack.solvers.sedov.doebling import Sedov as SedovDoebling

timmes_import = True
try:
    from exactpack.solvers.sedov.timmes import Sedov as SedovTimmes
except ImportError:
    timmes_import = False

# pyplot default settings

npts = 200001
rvec = np.linspace(0.0, 1.0, npts)
t = 1
t3 = 0.4
t2 = 1
eblast = 0.0673185
#
# Figure 8doebling: Standard test cases, Doebling Solver
#

solver_doebling_pla = SedovDoebling(geometry=1, eblast=eblast,
                                    gamma=1.4, omega=0.)


solution_doebling_pla_1 = solver_doebling_pla(r=rvec, t=t)
r = np.array(solver_doebling_pla.r_pnts)


# solution_doebling_pla_2 = solver_doebling_pla(r=rvec, t=t2)
# solution_doebling_pla_3 = solver_doebling_pla(r=rvec, t=t3)

density_dim = []
velocity_dim = []
rdim = []
pressure_dim = []
specific_internal_energy = []
with open(f'sedov_t={t}.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                density_dim.append(float(row['density']))
                rdim.append(float(row['position']))
                velocity_dim.append(float(row['velocity']))
                pressure_dim.append(float(row['pressure']))
                specific_internal_energy.append(float(row['specific_internal_energy']))

alpha = solver_doebling_pla.alpha
vwanto = solver_doebling_pla.vwanto
print(alpha, 'alpha')
A = 1.0
# r = np.array(r)
print(len(r))

rf = (eblast/A/alpha)**(1/3) * t**(2/3)
print(rf, 'rf')
rf2 = (eblast/A/alpha)**(1/3) * t2**(2/3)
rf3 = (eblast/A/alpha)**(1/3) * t3**(2/3)
xi = r/rf
gamma = 1.4
rr1 = xi*rf
rr2 = np.array(xi*rf2)
rr3 = np.array(xi*rf3)

# P =  (r+1e-16)**(-2) * t**2 * pressure_dim/A


# f_fun = solver_doebling_pla.f_fun_list

h_fun = solver_doebling_pla.h_fun_list

g_fun = np.array(solver_doebling_pla.g_fun_list)
l_fun = np.array(solver_doebling_pla.l_fun_list)

R = g_fun * 6


vlist = (np.array(solver_doebling_pla.vlist))
print(vlist, 'v there')
# zlist = 0.5 * (gamma-1) * vlist**2*(vlist-2/3)/ (2/3/gamma-vlist)


# P_interp = interp1d(vlist, P)
p = h_fun * solver_doebling_pla.p2
rho = np.array(g_fun * solver_doebling_pla.rho2)



v = f_fun * solver_doebling_pla.u2
# plt.show()
tlist = [1]
plt.ion()

term = 9 * (1.4+1)/8

# P = h_fun / (r**2+1e-16) * rf**2/term
# print(vPsorted[0])
# plt.plot(vPsorted[0], vPsorted[1])
# plt.show()

p_interp = interp1d(rdim, pressure_dim)


Ptest, rhotest, ztest, lambda_var = sedov()


for tt in tlist:
    solution_doebling_pla = solver_doebling_pla(r=rvec, t=tt)
    plt.figure(1)
    solution_doebling_pla.plot('density')
    rf = (eblast/A/alpha)**(1/3) * tt**(2/3)
    rr1 = xi*rf
    rtest = rf * lambda_var
    rfindex =  np.argmin(np.abs(rr1-rf)) - rr1.size

    rho[0:rfindex+1] = 1

    plt.plot(rr1, rho, 'o', mfc = 'none', label = 'nondim scaled')
    plt.show()

    
    # plt.figure(2)
    # P = zlist * rho / 1.4
    # # plt.plot(vPsorted[0], vPsorted[1])
    # plt.plot(rr1,  rr1**2*P/(tt**2), "o", mfc = 'none')
    # solution_doebling_pla.plot('pressure')
    # plt.figure(3)
    # plt.plot(rr1[:-1], p_interp(rr1[:-1]) - rr1[:-1]**2*P[:-1]/(tt**2) )
    # plt.show()
    # plt.figure(4)
    # # plt.plot(ztest, Ptest)
    # pressure_test = rtest**2*Ptest/(tt**2)
    # solution_doebling_pla.plot('specific_internal_energy')
    # plt.semilogy(rtest, pressure_test / rhotest /(gamma-1) , "--")
    # # plt.plot(zlist, P)
    # # plt.figure(5)
    # # plt.plot(zlist, P, 'o', mfc = 'none')
    # print(max(zlist), 'max zs')

    print(np.sqrt(np.mean((p_interp(rr1[:-1]) - rr1[:-1]**2*P[:-1]/(tt**2))**2)))










