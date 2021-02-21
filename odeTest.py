import numpy as np
import cosmoboost as cb

lmaxSimulation = 6000
pars = cb.DEFAULT_PARS

# cosmoboost parameters
lmax = pars['lmax'] = lmaxSimulation
delta_ell = pars['delta_ell'] = 8
pars['d'] = 1
beta = pars['beta']
T_0 = pars["T_0"]
pars['method'] = 'ODE'

kernel = cb.Kernel(pars)


