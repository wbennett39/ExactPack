'''Example demonstrating Sedov solvers. Reproduces plots from Kamm & Timmes,
"On Efficient Generation of Numerically Robust Sedov Solutions," LA-UR-07-2849

Uses Doebling and (if available) Timmes Sedov solvers.
'''


# import standard Python packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import csv 

# import ExactPack solvers
from exactpack.solvers.sedov.doebling import Sedov as SedovDoebling

timmes_import = True
try:
    from exactpack.solvers.sedov.timmes import Sedov as SedovTimmes
except ImportError:
    timmes_import = False

# pyplot default settings


# set domain variables for plots
npts = 201
rvec = np.linspace(0.0, 1, npts)
t = 1.0
t3 = 1.5
t2 = 1
eblast = 0.0673185
#
# Figure 8doebling: Standard test cases, Doebling Solver
#

solver_doebling_pla = SedovDoebling(geometry=1, eblast=eblast,
                                    gamma=1.4, omega=0.)
solution_doebling_pla_1 = solver_doebling_pla(r=rvec, t=t)
solution_doebling_pla_2 = solver_doebling_pla(r=rvec, t=t2)
solution_doebling_pla_3 = solver_doebling_pla(r=rvec, t=t3)

# write out solution
solution_doebling_pla_1.dump(f'sedov_t={t}.csv')
# read in solution
density_dim = []
velocity_dim = []
r = []
pressure_dim = []
specific_internal_energy = []
with open(f'sedov_t={t}.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                density_dim.append(float(row['density']))
                r.append(float(row['position']))
                velocity_dim.append(float(row['velocity']))
                pressure_dim.append(float(row['pressure']))
                specific_internal_energy.append(float(row['specific_internal_energy']))
# for ix, xx in enumerate(pressure_dim):
#     if abs(pressure_dim[ix] / (density_dim[ix]*0.4) - specific_internal_energy[ix]) >= 1e-8:

#         print(r[ix], pressure_dim[ix] / (density_dim[ix]*0.4) - specific_internal_energy[ix]) 

                

pressure_dim = np.array(pressure_dim)
velocity_dim = np.array(velocity_dim)
r = np.array(r)
density_dim = np.array(density_dim)
specific_internal_energy = np.array(specific_internal_energy)
# print(velocity_dim)
alpha = solver_doebling_pla.alpha
A = 1.0

rf = (eblast/A/alpha)**(1/3) * t**(2/3)
rf2 = (eblast/A/alpha)**(1/3) * t2**(2/3)
rf3 = (eblast/A/alpha)**(1/3) * t3**(2/3)

V = velocity_dim * t / (r+1e-10)
R = density_dim / A
P =  (r+1e-16)**(-2) * t**2 * pressure_dim/A
# e =  r**2 / t**2/R*P/(0.4)
e = pressure_dim / 0.4 /(density_dim+1e-5)
E = t**2/(r**2+1e-10) * specific_internal_energy
# print(E)
xi = r/rf
rr1 = xi*rf

# print(rf, 'rf')
# print(rf2, 'rf')

for ix, xx in enumerate(pressure_dim):
    if A *r[ix]**2/t**2 * P[ix] - pressure_dim[ix] > 1e-15:
        print(r[ix],A *r**2/t**2 * P[ix] - pressure_dim[ix] )
    elif abs(A * R[ix] - density_dim[ix])>1e-15:
        print('rho error')
    elif abs(P[ix]*A*rr1[ix]**2/t**2 - pressure_dim[ix] >1e-15):
        print('pressure error')






# solver_doebling_cyl = SedovDoebling(geometry=2, eblast=0.311357,
#                                     gamma=1.4, omega=0.)
# solution_doebling_cyl = solver_doebling_cyl(r=rvec, t=t)

# solver_doebling_sph = SedovDoebling(geometry=3, eblast=0.851072,
#                                     gamma=1.4, omega=0.)
# solution_doebling_sph = solver_doebling_sph(r=rvec, t=t)

fig = plt.figure(figsize=(10, 10))
plt.suptitle('''Sedov solutions for $\gamma=1.4$, standard cases, Doebling solver.
    Compare to Fig. 8 from Kamm & Timmes 2007''')

plt.subplot(223)

plt.plot(xi*rf2, R*A, "o", mfc = 'none')

solution_doebling_pla_1.plot('density')
solution_doebling_pla_2.plot('density')
solution_doebling_pla_3.plot('density')

plt.subplot(222)
plt.plot(xi*rf3, V*xi*rf2/t3, "o", mfc = 'none')
plt.plot(xi*rf, V*xi*rf/t, "o", mfc = 'none')
plt.plot(xi*rf2, V*xi*rf3/(t2), "o", mfc = 'none')
solution_doebling_pla_1.plot('velocity')
solution_doebling_pla_2.plot('velocity')
solution_doebling_pla_3.plot('velocity')

#solution_doebling_cyl.plot('density')
#solution_doebling_sph.plot('density')
#plt.xlim(0.0, 1.2)
#plt.ylim(0.0, 6.5)
plt.xlabel('Position (cm)')
plt.ylabel('Density (g/cc)')
plt.grid(True)
L = plt.legend(loc='upper left', bbox_to_anchor=(0.25, 1.25), ncol=3,
               fancybox=True, shadow=True)
L.get_texts()[0].set_text('t=1')
L.get_texts()[1].set_text('t=2')
L.get_texts()[2].set_text('t=3')
# plt.legend()
# plt.show()

plt.subplot(221)

solution_doebling_pla_1.plot('pressure')
solution_doebling_pla_2.plot('pressure')
solution_doebling_pla_3.plot('pressure')


rr = xi*rf2
plt.plot(rr,  P*A/t2**2*rr**2, "o", mfc = 'none')

plt.subplot(224)
# solution_doebling_pla_1.plot('specific_internal_energy')
# solution_doebling_pla_2.plot('specific_internal_energy')
# solution_doebling_pla_3.plot('specific_internal_energy')

# solution_doebling_pla_1.plot('specific_internal_energy')g
rr1 = np.array(xi*rf)
rr2 = np.array(xi*rf2)
rr3 = np.array(xi*rf3)
gamm1 = 1.4-1
pressure = P*A*rr1**2/t**2
density = R*A 
# print(P[0:5], 'P')
# print(pressure[0], density[0], 'pressure, density')
# specific_internal_energy = pressure / (gamm1 * density)
solution_doebling_pla_1.plot('specific_internal_energy')
# plt.plot(r, e, '^', mfc = 'none')
# solution_doebling_pla_2.plot('specific_internal_energy')
# solution_doebling_pla_3.plot('specific_internal_energy')


plt.plot(rr1,  specific_internal_energy, "o", mfc = 'none')
plt.gca().set_yscale('log')

plt.show()

def RMSE(l1, l2):
    diff = l1**2-l2**2
    return np.sqrt(np.mean(diff))

plt.figure(1)
# solution_doebling_pla_1.plot('specific_internal_energy')
# solution_doebling_pla_2.plot('specific_internal_energy')
# solution_doebling_pla_3.plot('specific_internal_energy')
# plt.plot(r, specific_internal_energy)
# plt.plot(rr1,  E*rr1**2/t**2, "o", mfc = 'none')
plt.plot(r, np.abs(E*rr1**2/t**2 - specific_internal_energy))
# plt.plot(rr3,  E*rr3**2/t3**2, "o", mfc = 'none')
# plt.plot(rr2,  E*rr2**2/t2**2, "o", mfc = 'none')
# plt.plot(rr1,  E*rr1**2/t**2, "o", mfc = 'none')
# print(rr1)
# plt.plot(rr2,  specific_internal_energy, "o", mfc = 'none')

# plt.plot(r, e, '^', mfc = 'none')
plt.gca().set_yscale('log')
plt.xlim(0,0.25)
plt.xlabel('r', fontsize = 16)
plt.ylabel('RMSE', fontsize = 16)
plt.savefig('6-12_etest.pdf')
plt.show()

print(RMSE(specific_internal_energy,E*rr1**2/t**2 ), 'RMSE')

# plt.show()
#
#plt.subplot(222)
#solution_doebling_pla.plot('velocity')
#solution_doebling_cyl.plot('velocity')
#solution_doebling_sph.plot('velocity')
#plt.xlim(0.0, 1.2)
#plt.ylim(0.0, 0.4)
#plt.xlabel('Position (cm)')
#plt.ylabel('Particle velocity (cm/s)')
#plt.grid(True)
#plt.gca().legend().set_visible(False)
#
plt.plot(1)


plt.show()
# solution_doebling_cyl.plot('specific_internal_energy')
# solution_doebling_sph.plot('specific_internal_energy')

#plt.xlim(0.0, 1.2)
#plt.ylim(1.e-2, 1.e5)
#plt.xlabel('Position (cm)')
#plt.ylabel('Specific internal energy (erg/g)')
#plt.grid(True)
#plt.gca().set_yscale('log')
#plt.gca().legend().set_visible(False)
#
#plt.subplot(224)
#solution_doebling_pla.plot('pressure')
#solution_doebling_cyl.plot('pressure')
#solution_doebling_sph.plot('pressure')
#plt.xlim(0.0, 1.2)
#plt.ylim(0.0, 0.15)
#plt.xlabel('Position (cm)')
#plt.ylabel('Pressure (erg/cc)')
#plt.grid(True)
#plt.gca().legend().set_visible(False)
#
#plt.tight_layout()
#fig.subplots_adjust(top=0.85)  # Makes room for suptitle
##plt.savefig('fig08doebling.pdf')
#
#
#if timmes_import:
#
#    #
#    # Figure 8timmes: Standard test cases, Timmes Solver
#    #
#
#    solver_timmes_pla = SedovTimmes(geometry=1, eblast=0.0673185,
#                                        gamma=1.4)
#    solution_timmes_pla = solver_timmes_pla(r=rvec, t=t)
#
#    solver_timmes_cyl = SedovTimmes(geometry=2, eblast=0.311357,
#                                        gamma=1.4)
#    solution_timmes_cyl = solver_timmes_cyl(r=rvec, t=t)
#
#    solver_timmes_sph = SedovTimmes(geometry=3, eblast=0.851072,
#                                        gamma=1.4)
#    solution_timmes_sph = solver_timmes_sph(r=rvec, t=t)
#
#    fig = plt.figure(figsize=(10, 10))
#    plt.suptitle('''Sedov solutions for $\gamma=1.4$, standard cases, Timmes solver.
#        Compare to Fig. 8 from Kamm & Timmes 2007''')
#
#    plt.subplot(221)
#    solution_timmes_pla.plot('density')
#    solution_timmes_cyl.plot('density')
#    solution_timmes_sph.plot('density')
#    plt.xlim(0.0, 1.2)
#    plt.ylim(0.0, 6.5)
#    plt.xlabel('Position (cm)')
#    plt.ylabel('Density (g/cc)')
#    plt.grid(True)
#    L = plt.legend(loc='upper left', bbox_to_anchor=(0.25, 1.25), ncol=3,
#                   fancybox=True, shadow=True)
#    L.get_texts()[0].set_text('planar')
#    L.get_texts()[1].set_text('cylindrical')
#    L.get_texts()[2].set_text('spherical')
#
#    plt.subplot(222)
#    solution_timmes_pla.plot('velocity')
#    solution_timmes_cyl.plot('velocity')
#    solution_timmes_sph.plot('velocity')
#    plt.xlim(0.0, 1.2)
#    plt.ylim(0.0, 0.4)
#    plt.xlabel('Position (cm)')
#    plt.ylabel('Particle velocity (cm/s)')
#    plt.grid(True)
#    plt.gca().legend().set_visible(False)
#
#    plt.subplot(223)
#    solution_timmes_pla.plot('specific_internal_energy')
#    solution_timmes_cyl.plot('specific_internal_energy')
#    solution_timmes_sph.plot('specific_internal_energy')
#    plt.xlim(0.0, 1.2)
#    plt.ylim(1.e-2, 1.e5)
#    plt.xlabel('Position (cm)')
#    plt.ylabel('Specific internal energy (erg/g)')
#    plt.grid(True)
#    plt.gca().set_yscale('log', nonposy='clip')
#    plt.gca().legend().set_visible(False)
#
#    plt.subplot(224)
#    solution_timmes_pla.plot('pressure')
#    solution_timmes_cyl.plot('pressure')
#    solution_timmes_sph.plot('pressure')
#    plt.xlim(0.0, 1.2)
#    plt.ylim(0.0, 0.15)
#    plt.xlabel('Position (cm)')
#    plt.ylabel('Pressure (erg/cc)')
#    plt.grid(True)
#    plt.gca().legend().set_visible(False)
#
#    plt.tight_layout()
#    fig.subplots_adjust(top=0.85)  # Makes room for suptitle
#    #plt.savefig('fig08timmes.pdf')
#
##
## Figure 9: Singular test cases
##
#
#solver_doebling_cyl = SedovDoebling(geometry=2, eblast=2.45749,
#                                    gamma=1.4, omega=1.66667)
#solution_doebling_cyl = solver_doebling_cyl(r=rvec, t=t)
#
#solver_doebling_sph = SedovDoebling(geometry=3, eblast=4.90875,
#                                    gamma=1.4, omega=2.33333)
#solution_doebling_sph = solver_doebling_sph(r=rvec, t=t)
#
#fig = plt.figure(figsize=(10, 10))
#plt.suptitle('''Sedov solutions for $\gamma=1.4$, singular cases, Doebling solver.
#    Compare to Fig. 9 from Kamm & Timmes 2007''')
#
#plt.subplot(221)
#solution_doebling_cyl.plot('density')
#solution_doebling_sph.plot('density')
#plt.xlim(0.0, 1.2)
#plt.ylim(0.0, 12.0)
#plt.xlabel('Position (cm)')
#plt.ylabel('Density (g/cc)')
#plt.grid(True)
#L = plt.legend(loc='upper left', bbox_to_anchor=(0.25, 1.25), ncol=2,
#               fancybox=True, shadow=True)
#L.get_texts()[0].set_text('cylindrical')
#L.get_texts()[1].set_text('spherical')
#
#plt.subplot(222)
#solution_doebling_cyl.plot('velocity')
#solution_doebling_sph.plot('velocity')
#plt.xlim(0.0, 1.2)
#plt.ylim(0.0, 0.8)
#plt.xlabel('Position (cm)')
#plt.ylabel('Particle velocity (cm/s)')
#plt.grid(True)
#plt.gca().legend().set_visible(False)
#
#plt.subplot(223)
#solution_doebling_cyl.plot('specific_internal_energy')
#solution_doebling_sph.plot('specific_internal_energy')
#plt.xlim(0.0, 1.2)
#plt.ylim(1.e-5, 1.e0)
#plt.xlabel('Position (cm)')
#plt.ylabel('Specific internal energy (erg/g)')
#plt.grid(True)
#plt.gca().set_yscale('log')
#plt.gca().legend().set_visible(False)
#
#plt.subplot(224)
#solution_doebling_cyl.plot('pressure')
#solution_doebling_sph.plot('pressure')
#plt.xlim(0.0, 1.2)
#plt.ylim(0.0, 0.7)
#plt.xlabel('Position (cm)')
#plt.ylabel('Pressure (erg/cc)')
#plt.grid(True)
#plt.gca().legend().set_visible(False)
#
#plt.tight_layout()
#fig.subplots_adjust(top=0.85)  # Makes room for suptitle
##plt.savefig('fig09.pdf')
#
##
## Figure 10: Vacuum test cases
##
#
#solver_doebling_cyl = SedovDoebling(geometry=2, eblast=2.67315,
#                                    gamma=1.4, omega=1.7)
#solution_doebling_cyl = solver_doebling_cyl(r=rvec, t=t)
#
#solver_doebling_sph = SedovDoebling(geometry=3, eblast=5.45670,
#                                    gamma=1.4, omega=2.4)
#solution_doebling_sph = solver_doebling_sph(r=rvec, t=t)
#
#fig = plt.figure(figsize=(10, 10))
#plt.suptitle('''Sedov solutions for $\gamma=1.4$, vacuum cases, Doebling solver
#    Compare to Fig. 10 from Kamm & Timmes 2007''')
#
#plt.subplot(221)
#solution_doebling_cyl.plot('density')
#solution_doebling_sph.plot('density')
#plt.xlim(0.0, 1.2)
#plt.ylim(0.0, 20.0)
#plt.xlabel('Position (cm)')
#plt.ylabel('Density (g/cc)')
#plt.grid(True)
#L = plt.legend(loc='upper left', bbox_to_anchor=(0.25, 1.25), ncol=2,
#               fancybox=True, shadow=True)
#L.get_texts()[0].set_text('cylindrical')
#L.get_texts()[1].set_text('spherical')
#
#plt.subplot(222)
#solution_doebling_cyl.plot('velocity')
#solution_doebling_sph.plot('velocity')
#plt.xlim(0.0, 1.2)
#plt.ylim(0.0, 0.8)
#plt.xlabel('Position (cm)')
#plt.ylabel('Particle velocity (cm/s)')
#plt.grid(True)
#plt.gca().legend().set_visible(False)
#
#plt.subplot(223)
#solution_doebling_cyl.plot('specific_internal_energy')
#solution_doebling_sph.plot('specific_internal_energy')
#plt.xlim(0.0, 1.2)
#plt.ylim(1.e-5, 1.e0)
#plt.xlabel('Position (cm)')
#plt.ylabel('Specific internal energy (erg/g)')
#plt.grid(True)
#plt.gca().set_yscale('log')
#plt.gca().legend().set_visible(False)
#
#plt.subplot(224)
#solution_doebling_cyl.plot('pressure')
#solution_doebling_sph.plot('pressure')
#plt.xlim(0.0, 1.2)
#plt.ylim(0.0, 0.7)
#plt.xlabel('Position (cm)')
#plt.ylabel('Pressure (erg/cc)')
#plt.grid(True)
#plt.gca().legend().set_visible(False)
#
#plt.tight_layout()
#fig.subplots_adjust(top=0.85)  # Makes room for suptitle
##plt.savefig('fig10.pdf')
#
plt.show()
