import jax.numpy as jnp
import jax
import diffrax

def initialise_diffrax_defaults(sim_params):
    sim_params['saveat'] = diffrax.SaveAt(ts=sim_params['tsave'])
    sim_params['solver'] = diffrax.Heun()
    sim_params['adjoint'] = diffrax.RecursiveCheckpointAdjoint()
    return sim_params

def create_params_lambda_solver_function(RT_equations,RT_args,sim_params):
    # Set up functions to function of params only
    params_RT_solve = lambda a,b : RT_solve(RT_equations,{'a' : a, 'b' : b},RT_args,sim_params)
    return params_RT_solve

def create_params_lambda_loss_function(params_RT_solve,analyticsol,RT_args):
    # Set up functions to function of params only
    params_ClosureLoss = lambda params : ClosureLoss(params,params_RT_solve,analyticsol,RT_args)
    return params_ClosureLoss

def RT_solve(RT_equations,params,RT_args,sim_params,nsteps = 1000000):
    """ Solve the provided RT equations using diffrax """
    term = diffrax.ODETerm(RT_equations)
    combined_args = params | RT_args

    Nt = sim_params['Nt']
    Np = RT_args['Np']
    Nx = RT_args['Nx']

    y0 = jnp.zeros((Np,Nx)).flatten()

    tsave   = sim_params['tsave']
    dt0     = sim_params['dt0']
    saveat  = sim_params['saveat']
    solver  = sim_params['solver']
    adjoint = sim_params['adjoint']

    solution = diffrax.diffeqsolve(term, solver, t0=0, t1=tsave[-1], dt0=dt0, y0=y0, saveat=saveat, args = combined_args, adjoint=adjoint, max_steps=nsteps)
    return solution.ys.reshape(Nt,Np,Nx)

def ClosureLoss(params,model,analyticsol,sim_params):
    pred_sol = model(params['a'],params['b'])
    Wsol = pred_sol[:,0,:]

    loss = 0.0
    for it in range(analyticsol['Nt']):
        loss += jnp.sum((jnp.interp(analyticsol['x'], sim_params['x'], Wsol[it,:])-analyticsol['W'][it,:])**2)

    return loss