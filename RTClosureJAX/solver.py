import jax.numpy as jnp
import jax
import diffrax

def initialise_diffrax(sim_params, diffrax_solver = diffrax.Heun(), diffrax_adjoint = diffrax.RecursiveCheckpointAdjoint()):
    """
    
    Sets up diffrax solvers, save points and adjoint problem solver. Defaults are those tested by author.

    """
    sim_params['saveat'] = diffrax.SaveAt(ts=sim_params['tsave'])
    sim_params['solver'] = diffrax_solver
    sim_params['adjoint'] = diffrax_adjoint
    return sim_params

def create_params_lambda_solver_function(RT_equations,RT_args,sim_params,nsteps = 1000000):
    """
    
    Wrapper on the radiation transport solver such that it is a callable function of the free parameters a and b only
    
    """
    params_RT_solve = lambda a,b : RT_solve(RT_equations,{'a' : a, 'b' : b},RT_args,sim_params,nsteps)
    return params_RT_solve

def RT_solve(RT_equations,params,RT_args,sim_params,nsteps):
    """
    
    Use diffrax to solve the provided radiation transport equations using provided settings
    
    """
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

