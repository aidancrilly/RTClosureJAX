from RTClosureJAX.solver import *
from RTClosureJAX.utils import *
from RTClosureJAX.RT_equations import *
from RTClosureJAX.closure_funcs import ML_Levermore_fluxlimiter

from time import time

import matplotlib.pyplot as plt

# Create Su Olson problem set up
Nx = 400
SuOlson_RT_args, SuOlson_sim_params = initialise_ModifiedSuOlson_problem(Nx)

# Load previously trained parameters
optimal_params = load_optimal_params("opt_closure_params.json")

# Initialise diffrax models
SuOlson_sim_params = initialise_diffrax(SuOlson_sim_params)

#
# Following steps are made for any solution
#
# 1. Select and initialise radiation transport model
# 2. Create parameterised version of model
# 3. Specify model free parameters
# 4. Compute solution
#

# Logical switches
l_DiscreteOrdinates    = True
l_ThirdOrderMoment     = False
l_VariableEddington    = False
l_FluxLimitedDiffusion = True

if(l_DiscreteOrdinates):
    # Discrete Ordinates solution
    SN_RT_args, SN_sim_params, SN_equations = initialise_DiscreteOrdinates(SuOlson_RT_args, SuOlson_sim_params)
    param_RT_solve = create_params_lambda_solver_function(SN_equations, SN_RT_args, SN_sim_params)
    a = None
    b = None
    start = time()
    SN_sol = param_RT_solve(a,b)
    print(f'SN simulation complete in {time()-start} s')
    Wsol,Fsol,Psol,Asol,Vsol = process_discrete_ordinates_sim(SN_sol)
    # fin.

if(l_ThirdOrderMoment):
    # Third Order Moment solution
    TMC_RT_args, TMC_sim_params, TMC_equations = initialise_ThirdOrderMoment(SuOlson_RT_args, SuOlson_sim_params)
    param_RT_solve = create_params_lambda_solver_function(TMC_equations, TMC_RT_args, TMC_sim_params)
    a = jnp.array(optimal_params['TMC']['a'])
    b = jnp.array(optimal_params['TMC']['b'])
    start = time()
    TMC_sol = param_RT_solve(a,b)
    print(f'TMC simulation complete in {time()-start} s')
    # fin.

if(l_VariableEddington):
    # Variable Eddington Factor solution
    VEF_RT_args, VEF_sim_params, VEF_equations = initialise_VariableEddingtonFactor(SuOlson_RT_args, SuOlson_sim_params)
    param_RT_solve = create_params_lambda_solver_function(VEF_equations, VEF_RT_args, VEF_sim_params)
    a = jnp.array(optimal_params['VEF']['a'])
    b = jnp.array(optimal_params['VEF']['b'])
    start = time()
    VEF_sol = param_RT_solve(a,b)
    print(f'VEF simulation complete in {time()-start} s')
    # fin.

if(l_FluxLimitedDiffusion):
    # Flux Limited Diffusion solution
    ga = jnp.array(optimal_params['TMC']['a'])
    gb = jnp.array(optimal_params['TMC']['b'])
    FLD_RT_args, FLD_sim_params, FLD_equations = initialise_FluxLimitedDiffusion(SuOlson_RT_args, SuOlson_sim_params, ML_Levermore_fluxlimiter, ga, gb)
    param_RT_solve = create_params_lambda_solver_function(FLD_equations, FLD_RT_args, FLD_sim_params)
    a = jnp.array(optimal_params['FLD']['a'])
    b = jnp.array(optimal_params['FLD']['b'])
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
ax4.set_ylabel("p")

ax4.set_xlabel("x")

ax4.axhline(1./3.,c='b',ls='--')
ax4.set_ylim(0.24,1.1)
ax1.set_xlim(0.0,SuOlson_sim_params['x'][-1])

for i in range(SuOlson_sim_params['Nt']):
    if(l_DiscreteOrdinates):
        # Discrete Ordinates
        ax1.plot(SuOlson_sim_params['x'],Wsol[i,:],'k')
        ax2.plot(SuOlson_sim_params['x'],Vsol[i,:],'k')
        ax3.plot(SuOlson_sim_params['x'],Fsol[i,:],'k')
        ax4.plot(SuOlson_sim_params['x'],Psol[i,:]/(Wsol[i,:]+1e-10),'k')

    if(l_ThirdOrderMoment):
        # Third Order Moment
        ax1.plot(SuOlson_sim_params['x'],TMC_sol[i,0,:],'k--')
        ax2.plot(SuOlson_sim_params['x'],TMC_sol[i,2,:],'k--')
        ax3.plot(SuOlson_sim_params['x'],TMC_sol[i,1,:],'k--')
        ax4.plot(SuOlson_sim_params['x'],TMC_sol[i,3,:]/(TMC_sol[i,0,:]+1e-10),'k--')

    if(l_VariableEddington):
        # Variable Eddington Factor
        ax1.plot(SuOlson_sim_params['x'],VEF_sol[i,0,:],'k:')
        ax2.plot(SuOlson_sim_params['x'],VEF_sol[i,2,:],'k:')
        ax3.plot(SuOlson_sim_params['x'],VEF_sol[i,1,:],'k:')

    if(l_FluxLimitedDiffusion):
        # Flux Limited Diffusion
        ax1.plot(SuOlson_sim_params['x'],FLD_sol[i,0,:],'k-.')
        ax2.plot(SuOlson_sim_params['x'],FLD_sol[i,1,:],'k-.')
        ax4.plot(SuOlson_sim_params['x'],FLD_sol[i,2,:]/(FLD_sol[i,0,:]+1e-10),'k-.')

fig.tight_layout()

plt.show()