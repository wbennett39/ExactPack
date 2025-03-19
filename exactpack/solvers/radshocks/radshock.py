'''
The docstring for radshock.
'''
import matplotlib.pyplot as plt
import os, copy, numpy, scipy, pickle, scipy.integrate, scipy.interpolate
import matplotlib.pyplot, importlib
try:
    import utils
except ImportError:
  from exactpack.solvers.radshocks import utils  
import numpy as np
from chaospy.quadrature import clenshaw_curtis
importlib.reload(utils)

# TOC:
# RadShock(object)
#   def __init__(self, various variables)
#
# IEShock(object)
#   def __init__(self, various variables)
#
# greyED_RadShock(RadShock)
#   def ED_driver(self)
#
# greyNED_RadShock(RadShock)
#   def nED_driver(self, epsilon = 1.,
#                  eps_precursor_ASP = 1.e-6, eps_relaxation_ASP = 1.e-6)
#
# greySn_RadShock(greyNED_RadShock)
#   def Sn_driver(self, Sn = 16, f_tol = 1.e-4)
#
# Shock_2Tie(IEShock)
#   def IE_driver(self)

class RadShock(object):
    def __init__(self, M0 = 1.2, rho0 = 1.,
                 Tref = 100., Cv = 1.4472799784454e12, gamma = 5. / 3.,
                 sigA = 577.35, sigS = 0.,
                 expDensity_abs = 0., expTemp_abs = 0.,
                 expDensity_scat = 0., expTemp_scat = 0.,
                 problem = 'nED', epsilon = 1., print_the_sources = 'False', 
                 eps_precursor_equil = 1.e-6,
                 eps_relaxation_equil = 1.e-6,
                 int_tol = 1.e-10, use_jac = 'True',
                 dxset = 5, numlev = 1, runtime = 1.e-7,
                 freezeWidth = 10., dumpnum = 100):
#: The initial Mach number.
        self.M0 = M0
#: The ambient, upstream equilibrium material density
        self.rho0 = rho0
#: The ambient, upstream equilibrium temperature
        self.Tref = Tref
#: The constant volume specific heat
        self.Cv = Cv
#: The adiabatic index, which is Cv / Cp
        self.gamma = gamma
#: The multiplicative factor for the absorption cross section
        self.sigA = sigA
#: The exponential power of density in the absorption cross section
        self.expDensity_abs = expDensity_abs
#: The exponential power of temperature in the absorption cross section
        self.expTemp_abs = expTemp_abs
#: The multiplicative factor for the scattering cross section
        self.sigS = sigS
#: The exponential power of density in the scattering cross section
        self.expDensity_scat = expDensity_scat
#: The exponential power of temperature in the scattering cross section
        self.expTemp_scat = expTemp_scat
        self.T0 = 1.
#: The speed of light in units of cm / s
        self.c = 2.99792458e10# [cm / s]
#: The radiation constant in units of erg / cm^3 / eV^4
        self.ar = 137.20172# [erg / cm^3 - eV^4]
#: The sound speed for an ideal gas
        self.sound = numpy.sqrt(self.gamma * (self.gamma - 1.) * self.Cv * \
                                self.Tref)
#: The ratio of the speed of light to the sound speed
        self.C0 = self.c / self.sound
#: A ratio proportional to the radiation pressure over an ideal kinetic energy
        self.P0 = self.ar * self.Tref**4 / (self.rho0 * self.sound**2)
#: A flag to print the residual source error
        self.print_the_sources = print_the_sources
#: The problem identifier allows the solution to differentiate between different nonequilibrium-diffusion rad-hydro models.  For example, the Lowrie-Morel source model as implemented in RAGE ("LM_nED"), the nonequilibrium-diffusion model that included first-order velocity corrections ("nED"), the flux-limited diffusion model (FLD) of Levermore-Pomraning ("FLD_LP"), the FLD model of Wilson's sum limiter ("FLD_1"), the FLD model of Larsen's square-root limiter ("FLD_2")
        self.problem = problem
        self.use_jac = use_jac
        self.eps_precursor_equil = eps_precursor_equil
        self.eps_relaxation_equil = eps_relaxation_equil
        self.int_tol = int_tol
        self.numlev = numlev
        self.runtime = runtime
        self.freezeWidth = freezeWidth
        self.dumpnum = dumpnum

class IEShock(object):
    def __init__(self, M0 = 1.4, rho0 = 1., Z = 1., Tref = 100., gamma = 5./3.,
                 Cv = 1.4472799784454e12, problem = 'ie',
                 print_the_sources = 'False', 
                 eps_precursor_equil = 1.e-6,
                 eps_relaxation_equil = 1.e-6,
                 eps_precursor_ASP = 1.e-6,
                 eps_relaxation_ASP = 1.e-6,
                 int_tol = 1.e-10, use_jac = False, epsilon = 1.):
        self.M0 = M0
        self.rho0 = rho0
        self.Z = Z
        self.Tref = Tref
        self.Cv = Cv
        self.gamma = gamma
        self.T0 = 1.
        self.Te0 = self.T0
        self.sound = numpy.sqrt(gamma * (gamma - 1.) * Cv * Tref)
        top = (3. * gamma - 1.) * Z + (gamma + 1.) * gamma
        bot = (3. - gamma) * Z + gamma + 1.
        self.Mc = numpy.sqrt(top / bot / gamma)
        self.print_the_sources = print_the_sources
        self.problem = problem
        self.eps_precursor_equil = eps_precursor_equil
        self.eps_relaxation_equil = eps_relaxation_equil
        self.eps_precursor_ASP = eps_precursor_ASP
        self.eps_relaxation_ASP = eps_relaxation_ASP
        self.int_tol = int_tol
        self.use_jac = use_jac
        self.epsilon = epsilon

class greyED_RadShock(RadShock):
    '''
    Define the grey equilibrium-diffusion radiative-shock problem,
    and drive the solution.
    '''
    def ED_driver(self):
        self.problem = 'ED'
        self.ED_profile = utils.ED_ShockProfiles(self)
        self.ED_profile.downstream_equilibrium()
        self.ED_profile.make_ED_solution()

class greyNED_RadShock(RadShock):
    '''
    Define the grey nonequilibrium-diffusion radiative-shock problem,
    and drive the solution.
    '''
# prob = radshock.greyNED_RadShock(M0 = 3., sigA = 44.93983839817290, sigS = 0.4006, expDensity_abs = 1, expTemp_abs = -3.5)
    def nED_driver(self, epsilon = 1.,
                   eps_precursor_ASP = 1.e-6, eps_relaxation_ASP = 1.e-6):
        self.Pr0 = self.T0**4 / 3.
        self.epsilon = epsilon
        self.eps_precursor_ASP = eps_precursor_ASP
        self.eps_relaxation_ASP = eps_relaxation_ASP
        self.eps_precursor_ASP_initially = min(eps_precursor_ASP, 1.e-3)
        self.eps_relaxation_ASP_initially = min(eps_relaxation_ASP, 1.e-3)
        self.nED_profile = utils.nED_ShockProfiles(self)
        self.nED_profile.downstream_equilibrium()
        self.nED_profile.make_2T_solution()

class greySn_RadShock(greyNED_RadShock):
    '''
    Define the grey-Sn radiative-shock problem, and drive the solution.
    '''
    def Sn_driver(self, Sn = 16, f_tol = 1.e-4, epsilon = 1.):
        self.nED_driver(epsilon = epsilon)
        self.Sn_profile = utils.Sn_ShockProfiles(self.nED_profile)
        self.Sn_profile.Sn = Sn
        self.Sn_profile.f_tol = f_tol
        self.Sn_profile.make_RT_solution()
        self.Sn_profile.continue_running()
        self.Sn_profile.update_dictionaries()
        if (self.Sn_profile.f_err[-1] > self.Sn_profile.f_tol):
            print_stmnt  = 'The greySn_RadShock solution for M0 = '
            print_stmnt += str(self.M0) + ' failed to converge.'
            print(print_stmnt)
        print(np.shape(self.Sn_profile.Im))
        self.Im = self.Sn_profile.Im
        Nested_Class = NestedSn()
        nested_phi = np.zeros((Nested_Class.ns_list.size, self.Sn_profile.x.size))
        nested_P = np.zeros((Nested_Class.ns_list.size, self.Sn_profile.x.size))
        VEF = np.zeros((Nested_Class.ns_list.size, self.Sn_profile.x.size))
        for ix in range(self.Sn_profile.x.size):
                res = Nested_Class.make_nested_phi(self.Im[:, ix]) 
                # print(np.shape(res[0]), 'res shape')

                nested_phi[:,ix] = 4 * np.pi * res[0]
                nested_P[:,ix] = 4 * np.pi * res[2]
                VEF[:,ix] = nested_P[:,ix] /nested_phi[:,ix]

        Tr = self.Tref *(nested_phi  )**.25
        # print(nested_phi)
        print(self.ar, 'ar')
        print(np.shape(nested_phi), 'nested phi shape')
        x = self.Sn_profile.x
        plt.plot(-x, Tr[0, :])
        plt.plot(-x, Tr[1, :])
        plt.plot(-x, Tr[2, :])
        plt.plot(-x, self.Sn_profile.Tr * self.Tref, 'k-')
        plt.xlim(-0.002, 0.004)
        print(self.Sn_profile.Tr/Tr)
        # assert 0


        plt.figure(2)
        RMSE_vals = np.zeros(Nested_Class.ns_list.size-1)
        Richardson_vals = np.zeros((Nested_Class.ns_list.size-1, self.Sn_profile.x.size))
        VEF_vals_Richardson = np.zeros((Nested_Class.ns_list.size-1))
        VEF_Q = np.zeros((Nested_Class.ns_list.size-1))
        for ix in range(Nested_Class.ns_list.size-1):
            RMSE_vals[ix] = RMSE(nested_phi[:,ix], nested_phi[:,-1])
        for ix in range(2,Nested_Class.ns_list.size):
            xdata = Nested_Class.ns_list[:ix]
            VEF_Q[ix-1] = np.sum(VEF[ix,:]-1/3)
            VEF_vals_Richardson[ix-1] = convergence_estimator(xdata, VEF_Q[:ix], target = Nested_Class.ns_list[ix-1], method = 'richardson')

            for ixx in range(self.Sn_profile.x.size):
                ydata = nested_phi[:ix, ixx]
                Richardson_vals[ix-1, ixx] = convergence_estimator(xdata, ydata, target = Nested_Class.ns_list[ix-1], method = 'richardson')
        plt.loglog(Nested_Class.ns_list[:-1], RMSE_vals)
        plt.savefig('NestedSn_Tr_converge.pdf')
        # plt.loglog(Nested_Class.ns_list[1:], Richardson_vals[:, int(self.Sn_profile.x.size/2)])
        plt.show()
        plt.figure(3)
        meanRichardson = np.zeros(Nested_Class.ns_list.size-1)
        for ix in range(Nested_Class.ns_list.size-1):
            meanRichardson[ix] = np.mean(Richardson_vals[ix,:])
        plt.loglog(Nested_Class.ns_list[1:], meanRichardson)
        plt.savefig('NestedSn_meanTr_converge.pdf')
        plt.show()
        print(VEF_Q, 'VEF_Q')
        plt.figure(4)
        plt.loglog(Nested_Class.ns_list[1:], VEF_vals_Richardson)
        plt.show()


        
        # VEF = np.sum()


            
        # plt.plot(x, nested_phi[3, :])
        # plt.plot(x, nested_phi[4, :])
        # plt.plot(x, nested_phi[5, :])
            

class Shock_2Tie(IEShock):
    
    
    '''
    Define the ion-electron shock problem, and drive the solution.
    '''
    def IE_driver(self):
        if (self.M0 <= self.Mc):
          self.IE_profile = utils.IE_continuousShockProfiles(self)
          self.IE_profile.downstream_equilibrium()
          self.IE_profile.make_continuous_solution()
        else:
          self.IE_profile = utils.IE_discontinuousShockProfiles(self)
          self.IE_profile.downstream_equilibrium()
          self.IE_profile.make_2T_solution()




class NestedSn:
    
    def __init__(self):
        self.ns_list = np.array([2,6,16, 46])
        self.xs_mat = np.zeros((self.ns_list.size, self.ns_list[-1] ))
        self.N_ang = 46
        self.index_mat = np.zeros((self.ns_list.size, self.N_ang ))
        self.w_mat = np.zeros((self.ns_list.size, self.ns_list[-1] ))
        self.mus, self.ws = cc_quad(self.N_ang)
        self.weights_matrix()
    
    def weights_matrix(self):
        for i in range(self.ns_list.size):
                self.w_mat[i, 0:self.ns_list[i]] = cc_quad(self.ns_list[i])[1]
                self.xs_mat[i, 0:self.ns_list[i]] = cc_quad(self.ns_list[i])[0]
                igrab = False
                ig = 0
    
    def make_nested_phi(self, psi):
                phi_list = np.zeros(self.ns_list.size)
                Jp_list = np.zeros(self.ns_list.size)
                P_list = np.zeros(self.ns_list.size)
                # self.make_phi(psi, self.w_mat[-1])
                # phi = np.sum(psi[:]*self.w_mat[-1])*0.5
                phi = np.sum(psi[:]*self.ws)*0.5
                J = np.sum(psi[:]*self.ws*self.mus)*0.5
                P = np.sum(psi[:]*self.ws*self.mus*self.mus)*0.5


                phi_list[-1] = phi
                Jp_list[-1] = J
                P_list[-1] = P
                psi_old = psi 
                for ix in range(2, self.ns_list.size+1):
              
                        psi_lower = self.make_lower_order_fluxes(psi_old)
                        
                        # print(psi_lower.size, 'psi l')
                        # print(self.w_mat[-ix-1, 0:self.ns_list[-ix-1]])
                        # phi_list[-ix] = self.make_phi(psi_lower, self.w_mat[-ix, 0:self.ns_list[ix]])
                        phi_list[-ix] = np.sum(psi_lower[:]*self.w_mat[-ix, 0:self.ns_list[-ix]])*0.5
                        Jp_list[-ix] = np.sum(psi_lower[:]*self.w_mat[-ix, 0:self.ns_list[-ix]] * self.xs_mat[-ix, 0:self.ns_list[-ix]])*0.5
                        P_list[-ix] = np.sum(psi_lower[:]*self.w_mat[-ix, 0:self.ns_list[-ix]] * self.xs_mat[-ix, 0:self.ns_list[-ix]]* self.xs_mat[-ix, 0:self.ns_list[-ix]])*0.5

                        # print(self.ns_list[-ix-1])
                        # print(self.w_mat[-ix-1, 0:self.ns_list[-ix-1]], 'ws')
                        
                        

                        psi_old = psi_lower
                print(phi_list, 'phi_list')
                # print(phi_list[-1]-phi)
                # print(len(phi_list))
                return np.array(phi_list), np.array(Jp_list), np.array(P_list)

    def make_lower_order_fluxes(self, psi):
        psi_new = []
        xs_test = []
        count = 1
        psi_new.append(psi[0])
        xs_test.append(self.mus[0])
        if psi.size == 6:
            psi_new = np.array([psi[0], psi[-1]])
        else:
            for ix in range(1,psi.size):
                if count%3 == 0:
                    psi_new.append(psi[count])
                    xs_test.append(self.mus[count])
                    # count = 0
                count += 1
        # xs_test.append(self.mus[-1])
        xs_test = np.array(xs_test)
        # print(xs_test.size, 'xs size')
        # psi_new.append(psi[-1])
        # print(xs_test, 'xs')
        return np.array(psi_new)
    
    def  wynn_epsilon_algorithm(self, S):
        n = S.size
        width = n-1
        # print(width)
        tableau = np.zeros((n + 1, width + 2))
        tableau[:,0] = 0
        tableau[1:,1] = S.copy() 
        for w in range(2,width + 2):
            for r in range(w,n+1):
                #print(r,w)
                # if abs(tableau[r,w-1] - tableau[r-1,w-1]) <= 1e-15:
                #     print('potential working precision issue')
                tableau[r,w] = tableau[r-1,w-2] + 1/(tableau[r,w-1] - tableau[r-1,w-1])
        return tableau


def cc_quad(N):
    x, w= clenshaw_curtis(N-1,(-1,1))
    return x[0], w


def convergence_estimator(xdata, ydata, target = 256, method = 'linear_regression'):
    if method == 'linear_regression':
        # lastpoint = ydata[-1]
        # ynew = np.log(np.abs(ydata[1:]-ydata[:-1]))
        # xnew = np.log(np.abs(xdata[1:]-xdata[:-1]))
        # a, b = np.polyfit(xnew, ynew,1)
        # err_estimate = (np.exp(b) * np.abs(target-xdata[:-1])**a)[-1]
        # print(err_estimate, 'err estimate')
        ynew = np.log(np.abs(ydata[-1]-ydata[:-1]))
        xnew = np.log(xdata[:-1])
        a, b = np.polyfit(xnew, ynew,1)
        c1 = np.exp(b)
        err_estimate = c1 * target ** a

        
    elif method == 'difference':
        # err_estimate = np.abs(ydata[-1] - ydata[-2]) /(xdata[-1]-xdata[-2]) 
        
        # alpha = np.abs(ydata[-1] - ydata[-2]) *xdata[-2]
        # err_estimate = alpha/target

        # err_estimate = np.abs(ydata[-1] - ydata[-2]) / np.abs(xdata[-2]-xdata[-1])/target*xdata[-2]*xdata[-1]
        err_estimate = np.abs(ydata[-1]-ydata[-2])
    
    
    elif method == 'richardson':
        ynew = np.log(np.abs(ydata[-1]-ydata[:-1]))
        xnew = np.log(xdata[:-1])
        a, b = np.polyfit(xnew, ynew,1)
        c1 = np.exp(b)
        k0 = -a
        h = 1/ xdata[-2]
        t = h * xdata[-1]
        A1 = (t**k0 * ydata[-1] -ydata[-2]) / (t**k0 - 1)
        err_estimate = np.abs(ydata[-1] - A1)

    return err_estimate    # return a



def RMSE(l1,l2):
    return np.sqrt(np.mean((l1-l2)**2))