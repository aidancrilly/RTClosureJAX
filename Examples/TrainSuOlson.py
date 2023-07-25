from RTClosureJAX.solver import *
from RTClosureJAX.utils import *
from RTClosureJAX.RT_equations import *
from RTClosureJAX.closure_funcs import *
from RTClosureJAX.optimizer import *

import json
from time import time
import matplotlib.pyplot as plt

# Create Su Olson problem set up
Nx = 400
SuOlson_RT_args, SuOlson_sim_params = initialise_SuOlson_problem(Nx)

# Initialise diffrax models
SuOlson_sim_params = initialise_diffrax(SuOlson_sim_params)
# Initialise optax learning rate schedule
Ntrain_steps = 100
decay_rate   = 0.5
boundaries   = [0.5]
lr_schedule  = create_piecewise_learning_rate_schedule(1e-2,Ntrain_steps,decay_rate,boundaries)

# Get analytic solution
SuOlson_analytic_solution = get_SuOlson_analytic_solution(SuOlson_sim_params)

#
# Following steps are made for any solution
#
# 1. Select and initialise radiation transport model
# 2. Create parameterised version of model
# 3. Specify model free parameters
# 4. Compute solution
#

# Logical switches
l_ThirdOrderMoment     = False
l_VariableEddington    = True
l_FluxLimitedDiffusion = False
l_plot_results         = False
l_save_results         = False

if(l_ThirdOrderMoment):
    # Third Order Moment solution
    TMC_RT_args, TMC_sim_params, TMC_equations = initialise_ThirdOrderMoment(SuOlson_RT_args, SuOlson_sim_params)
    param_RT_solve = create_params_lambda_solver_function(TMC_equations, TMC_RT_args, TMC_sim_params)
    param_ClosureLoss = create_params_lambda_loss_function(param_RT_solve,SuOlson_analytic_solution,TMC_sim_params)
    # Starting values
    a0 = np.array([0.0,0.0,0.0])
    b0 = np.array([0.0,0.0,0.0])
    start = time()
    TMC_loss_history, TMC_opt_params = learn_closure(a0,b0,param_ClosureLoss,Ntrain_steps,lr_schedule)
    print(f'TMC training complete in {time()-start} s')
    TMC_sol = param_RT_solve(TMC_opt_params['a'],TMC_opt_params['b'])
    # fin.

if(l_VariableEddington):
    # Variable Eddington Factor solution
    VEF_RT_args, VEF_sim_params, VEF_equations = initialise_VariableEddingtonFactor(SuOlson_RT_args, SuOlson_sim_params)
    param_RT_solve = create_params_lambda_solver_function(VEF_equations, VEF_RT_args, VEF_sim_params)
    param_ClosureLoss = create_params_lambda_loss_function(param_RT_solve,SuOlson_analytic_solution,VEF_sim_params)
    # Starting values
    a0 = jnp.array([0.0,-4.0/3.0,2.0])
    b0 = jnp.array([0.0,0.0,0.0])
    start = time()
    VEF_loss_history, VEF_opt_params = learn_closure(a0,b0,param_ClosureLoss,Ntrain_steps,lr_schedule)
    print(f'VEF training complete in {time()-start} s')
    VEF_sol = param_RT_solve(VEF_opt_params['a'],VEF_opt_params['b'])
    # fin.

if(l_FluxLimitedDiffusion):
    # Flux Limited Diffusion solution
    FLD_RT_args, FLD_sim_params, FLD_equations = initialise_FluxLimitedDiffusion(SuOlson_RT_args, SuOlson_sim_params, Levermore_fluxlimiter, dt_mult = 1e-3)
    param_RT_solve = create_params_lambda_solver_function(FLD_equations, FLD_RT_args, FLD_sim_params)
    param_ClosureLoss = create_params_lambda_loss_function(param_RT_solve,SuOlson_analytic_solution,FLD_sim_params)
    # Starting values
    a0 = np.array([0.0,0.0,0.0])
    b0 = np.array([0.0,0.0,0.0])
    start = time()
    FLD_loss_history, FLD_opt_params = learn_closure(a0,b0,param_ClosureLoss,Ntrain_steps,lr_schedule)
    print(f'FLD training complete in {time()-start} s')
    FLD_sol = param_RT_solve(FLD_opt_params['a'],FLD_opt_params['b'])
    # fin.

# Plot results
if(l_plot_results):
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

    for i in range(SuOlson_sim_params['Nt']):

        if(l_ThirdOrderMoment):
            if(i == 0):
                print(TMC_opt_params)
            # Third Order Moment
            ax1.plot(SuOlson_sim_params['x'],TMC_sol[i,0,:],'k--')
            ax2.plot(SuOlson_sim_params['x'],TMC_sol[i,2,:],'k--')
            ax3.plot(SuOlson_sim_params['x'],TMC_sol[i,1,:],'k--')
            ax4.plot(SuOlson_sim_params['x'],TMC_sol[i,3,:]/(TMC_sol[i,0,:]+1e-10),'k--')

        if(l_VariableEddington):
            if(i == 0):
                print(VEF_opt_params)
            # Variable Eddington Factor
            ax1.plot(SuOlson_sim_params['x'],VEF_sol[i,0,:],'k:')
            ax2.plot(SuOlson_sim_params['x'],VEF_sol[i,2,:],'k:')
            ax3.plot(SuOlson_sim_params['x'],VEF_sol[i,1,:],'k:')

        if(l_FluxLimitedDiffusion):
            if(i == 0):
                print(FLD_opt_params)
            # Flux Limited Diffusion
            ax1.plot(SuOlson_sim_params['x'],FLD_sol[i,0,:],'k-.')
            ax2.plot(SuOlson_sim_params['x'],FLD_sol[i,1,:],'k-.')
            ax4.plot(SuOlson_sim_params['x'],FLD_sol[i,2,:]/(FLD_sol[i,0,:]+1e-10),'k-.')

        # Analytic
        ax1.scatter(SuOlson_analytic_solution['x'],SuOlson_analytic_solution['W'][i,:],c='r',marker='D')
        ax2.scatter(SuOlson_analytic_solution['x'],SuOlson_analytic_solution['V'][i,:],c='r',marker='D')

    fig.tight_layout()

    plt.show()

if(l_save_results):
    json_filename = SimDataDir+'new_opt_closure_params.json'
    json_write_dict = {}
    if(l_ThirdOrderMoment):
        json_write_dict['TMC'] = {'a' : TMC_opt_params['a'].tolist(), 'b' : TMC_opt_params['b'].tolist()}

    if(l_VariableEddington):
        json_write_dict['VEF'] = {'a' : VEF_opt_params['a'].tolist(), 'b' : VEF_opt_params['b'].tolist()}

    if(l_FluxLimitedDiffusion):
        json_write_dict['FLD'] = {'a' : FLD_opt_params['a'].tolist(), 'b' : FLD_opt_params['b'].tolist()}
    
    json_write = json.dumps(json_write_dict, indent=4)
    with open(json_filename,'w') as f:
        print(json_write, file=f)
