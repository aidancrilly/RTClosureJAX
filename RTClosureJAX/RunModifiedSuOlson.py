from core.solver import *
from core.utils import *
from core.RT_equations import *
from core.closure_funcs import Levermore_fluxlimiter, Larsen_2_fluxlimiter, diffusion_fluxlimter

from time import time

import matplotlib.pyplot as plt

# Create Su Olson problem set up
Nx = 1200
SuOlson_RT_args, SuOlson_sim_params = initialise_ModifiedSuOlson_problem(Nx)

# Initialise diffrax models
SuOlson_sim_params = initialise_diffrax_defaults(SuOlson_sim_params)

#
# Following steps are made for any solution
#
# 1. Select and initialise radiation transport model
# 2. Create parameterised version of model
# 3. Specify model free parameters
# 4. Compute solution
#

# Discrete Ordinates solution
SN_RT_args, SN_sim_params, SN_equations = initialise_DiscreteOrdinates(SuOlson_RT_args, SuOlson_sim_params)
param_RT_solve = create_params_lambda_solver_function(SN_equations, SN_RT_args, SN_sim_params)
a = None
b = None
start = time()
SN_sol = param_RT_solve(a,b)
print(f'SN simulation complete in {time()-start} s')
# fin.

# Third Order Moment solution
TMC_RT_args, TMC_sim_params, TMC_equations = initialise_ThirdOrderMoment(SuOlson_RT_args, SuOlson_sim_params)
param_RT_solve = create_params_lambda_solver_function(TMC_equations, TMC_RT_args, TMC_sim_params)
a = np.array([-0.05065705, -0.05513487,  0.84024453])
b = np.array([0.04694922,  0.08595726, -0.17512433])
start = time()
TMC_sol = param_RT_solve(a,b)
print(f'TMC simulation complete in {time()-start} s')
# fin.

# Variable Eddington Factor solution
VEF_RT_args, VEF_sim_params, VEF_equations = initialise_VariableEddingtonFactor(SuOlson_RT_args, SuOlson_sim_params)
param_RT_solve = create_params_lambda_solver_function(VEF_equations, VEF_RT_args, VEF_sim_params)
a = jnp.array([-0.5403172,  1.3603048])
b = jnp.array([-0.5747044 , -0.57543194,  0.18687488])
start = time()
VEF_sol = param_RT_solve(a,b)
print(f'VEF simulation complete in {time()-start} s')
# fin.

# Flux Limited Diffusion solution
FLD_RT_args, FLD_sim_params, FLD_equations = initialise_FluxLimitedDiffusion(SuOlson_RT_args, SuOlson_sim_params,Levermore_fluxlimiter)
param_RT_solve = create_params_lambda_solver_function(FLD_equations, FLD_RT_args, FLD_sim_params)
a = np.array([-0.05065705, -0.05513487,  0.84024453])
b = np.array([0.04694922,  0.08595726, -0.17512433])
start = time()
FLD_sol = param_RT_solve(a,b)
print(f'FLD simulation complete in {time()-start} s')
# fin.

# Plot results
fig = plt.figure(dpi=200,figsize=(6,9))
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412,sharex=ax1,sharey=ax1)
ax3 = fig.add_subplot(413,sharex=ax1)
ax4 = fig.add_subplot(414,sharex=ax1)

ax1.set_ylabel("W")
ax2.set_ylabel("V")
ax3.set_ylabel("F")
ax4.set_ylabel("f")

ax4.set_xlabel("x")

ax4.axhline(1./3.,c='b',ls='--')
ax4.set_ylim(0.24,1.1)
ax1.set_xlim(0.0,SuOlson_sim_params['x'][-1])

Wsol,Fsol,Psol,Asol,Vsol = process_discrete_ordinates_sim(SN_sol)

for i in range(SuOlson_sim_params['Nt']):
    # Discrete Ordinates
    ax1.plot(SuOlson_sim_params['x'],Wsol[i,:],'k')
    ax2.plot(SuOlson_sim_params['x'],Vsol[i,:],'k')
    ax3.plot(SuOlson_sim_params['x'],Fsol[i,:],'k')
    ax4.plot(SuOlson_sim_params['x'],Psol[i,:]/(Wsol[i,:]+1e-10),'k')

    # Third Order Moment
    ax1.plot(SuOlson_sim_params['x'],TMC_sol[i,0,:],'k--')
    ax2.plot(SuOlson_sim_params['x'],TMC_sol[i,2,:],'k--')
    ax3.plot(SuOlson_sim_params['x'],TMC_sol[i,1,:],'k--')
    ax4.plot(SuOlson_sim_params['x'],TMC_sol[i,3,:]/(TMC_sol[i,0,:]+1e-10),'k--')

    # Variable Eddington Factor
    ax1.plot(SuOlson_sim_params['x'],VEF_sol[i,0,:],'k:')
    ax2.plot(SuOlson_sim_params['x'],VEF_sol[i,2,:],'k:')
    ax3.plot(SuOlson_sim_params['x'],VEF_sol[i,1,:],'k:')

    # Flux Limited Diffusion
    ax1.plot(SuOlson_sim_params['x'],FLD_sol[i,0,:],'k-.')
    ax2.plot(SuOlson_sim_params['x'],FLD_sol[i,1,:],'k-.')
    ax4.plot(SuOlson_sim_params['x'],FLD_sol[i,2,:]/(FLD_sol[i,0,:]+1e-10),'k-.')

fig.tight_layout()

plt.show()