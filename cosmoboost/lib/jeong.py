import numpy as np

########################################
#     Jeong approximate functions
########################################


def jeong_boost_Cl_1storder(L, Cl, beta, cos_avg=None, only_dCl=True):
    if cos_avg == None:
        from scipy.integrate import dblquad
        cos_avg = dblquad(lambda ph, th: np.cos(th) * np.sin(th), 0, np.pi, lambda th: 0,
                          lambda th: 2 * np.pi)[0]

        print("using <cos> = {}".format(cos_avg))
    dCl = np.gradient(Cl)
    dCl_1st = - beta * cos_avg * L * dCl

    if only_dCl:
        return dCl_1st
    else:
        return Cl + dCl_1st


def jeong_boost_Cl_2ndorder(L, Cl, beta, cos2_avg=None, only_dCl=True):
    if cos2_avg == None:
        from scipy.integrate import dblquad
        cos2_avg = dblquad(lambda ph, th: np.cos(th) ** 2 * np.sin(th), 0, np.pi, lambda th: 0,
                           lambda th: 2 * np.pi)[0]/4/np.pi
        print("using <cos2> = {}".format(cos2_avg))
    dCl = np.gradient(Cl, L)
    d2Cl = np.gradient(dCl, L)

    dCl_2nd = (beta ** 2 / 2) * (L * dCl + d2Cl * L ** 2 * cos2_avg)

    if only_dCl:
        return dCl_2nd
    else:
        return Cl + dCl_2nd


def jeong_boost_Cl(L, Cl, beta, cos_avg=None, cos2_avg=None):

    dCl_1st = jeong_boost_Cl_1storder(L, Cl, beta, cos_avg=cos_avg, only_dCl=True)
    dCl_2nd = jeong_boost_Cl_2ndorder(L, Cl, beta, cos2_avg=cos2_avg, only_dCl=True)

    return Cl + dCl_1st + dCl_2nd
