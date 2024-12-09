import numpy as np
from .doebling import Sedov 


def sedov(gamma = 1.4, eblast = 0.0673185, npnts = 201):
    
    solver_doebling_pla = Sedov(geometry=1, eblast=eblast,
                                    gamma=gamma, omega=0.)
    rvec = np.linspace(0,1)
    t=1
    solution_doebling_pla_1 = solver_doebling_pla(r=rvec, t=t)
    alpha = solver_doebling_pla.alpha
    vzero = solver_doebling_pla.vwanto

    v2 = 4  / (3*(gamma+1))
    v0 = 2/3/gamma
    vstar = 2 /((gamma-1) + 2)
    v= np.linspace(v0, v2, npnts)
    v[0] = vzero
    # v = np.linspace(0, min(v0, v2), npnts)
    gfun = np.zeros(npnts)
    lambda_var = np.zeros(npnts)

    # zs = 0.5*(gamma-1) * v**2 * (v-2/3) / (2/(3*gamma) - v)
    zs = 0.5 * (gamma-1) * v**2*(v-2/3)/ (2/3/gamma-v)

    # find rho/rho1
    # a = 3 * (gamma+1) / 4
    # b = (gamma+1) / (gamma-1)
    # c = 3 * gamma / 2
    # d = (3 * (gamma + 1)) / (3 * (gamma+1) - 2 *(2 + gamma-1 ))
    # e = (2 + (gamma - 1))/2
    a = solver_doebling_pla.a_val
    b = solver_doebling_pla.b_val
    c = solver_doebling_pla.c_val
    d = solver_doebling_pla.d_val
    e = solver_doebling_pla.e_val

    # alpha0 = 2/3
    # alpha2 = - (gamma - 1) / (2*(gamma-1)+1)
    # alpha1 = 3 * gamma / (2+ (gamma -1)) *((2 * (2-gamma))/9/gamma/gamma - alpha2)
    # alpha3 = 1 / (2*(gamma-1)+1)
    # alpha4 = 3 / (2-gamma) * alpha1
    # alpha5 = -2 / (2-gamma)

    alpha0 = solver_doebling_pla.a0
    alpha1 = solver_doebling_pla.a1
    alpha2 = solver_doebling_pla.a2
    alpha3 = solver_doebling_pla.a3
    alpha4 = solver_doebling_pla.a4
    alpha5 = solver_doebling_pla.a5

    
    for iv, vv in enumerate(v):
        x1 = a * vv
        x2 = b*(c*vv-1)
        x3 = d*(1-e*vv)
        x4 = b*(1-c/gamma*vv)
        # print(alpha3, alpha4, alpha5)

        gfun[iv] = (x2**alpha3)*(x3**alpha4)*(x4**alpha5)
        lambda_var[iv] = x1**(-alpha0) * x2 ** (-alpha2) * x3 ** (-alpha1)
        # print(x2, alpha3)

        # print(gfun[iv])

    
    rho = gfun * 6

    P = zs * rho / gamma
    print(max(zs), 'max z')
    return P, rho, zs, lambda_var



        