from RTClosureJAX.solver import *
from RTClosureJAX.utils import *
from RTClosureJAX.RT_equations import *

"""

Tests if total energy is conserved by the numerical scheme

We use the Su Olson problem as a test bed with tmax > 10 so we also test in the absence of external sources

"""

test_tol = 1e-5

def initialise(Nx = 400,L=8.0,Nt = 100,tmax=15.0):
    # Create diffusion problem set up
    SuOlson_RT_args, SuOlson_sim_params = initialise_SuOlson_problem(Nx)
    SuOlson_sim_params['tsave'] = jnp.linspace(0.0,tmax,Nt)
    SuOlson_sim_params['Nt'] = Nt

    # Load previously trained parameters
    optimal_params = load_optimal_params("opt_closure_params.json")

    # Initialise diffrax models
    SuOlson_sim_params = initialise_diffrax(SuOlson_sim_params)

    # Total energy analytic solution
    analytic_solution = jnp.where(SuOlson_sim_params['tsave'] < 10.0, 0.5*SuOlson_sim_params['tsave'], 5.0)
    return SuOlson_RT_args, SuOlson_sim_params, optimal_params, analytic_solution

def run_SN(SuOlson_RT_args, SuOlson_sim_params, optimal_params):
    # Discrete Ordinates solution
    SN_RT_args, SN_sim_params, SN_equations = initialise_DiscreteOrdinates(SuOlson_RT_args, SuOlson_sim_params)
    param_RT_solve = create_params_lambda_solver_function(SN_equations, SN_RT_args, SN_sim_params)
    a = None
    b = None
    SN_sol = param_RT_solve(a,b)
    Wsol,Fsol,Psol,Asol,Vsol = process_discrete_ordinates_sim(SN_sol)
    return jnp.sum(Wsol+Vsol,axis=1)*SN_RT_args['dx']

def run_TMC(SuOlson_RT_args, SuOlson_sim_params, optimal_params):
    # Third Order Moment solution
    TMC_RT_args, TMC_sim_params, TMC_equations = initialise_ThirdOrderMoment(SuOlson_RT_args, SuOlson_sim_params)
    param_RT_solve = create_params_lambda_solver_function(TMC_equations, TMC_RT_args, TMC_sim_params)
    a = jnp.array(optimal_params['TMC']['a'])
    b = jnp.array(optimal_params['TMC']['b'])
    TMC_sol = param_RT_solve(a,b)
    return jnp.sum(TMC_sol[:,0,:]+TMC_sol[:,2,:],axis=1)*TMC_RT_args['dx']
    
def run_VEF(SuOlson_RT_args, SuOlson_sim_params, optimal_params):
    # Variable Eddington Factor solution
    VEF_RT_args, VEF_sim_params, VEF_equations = initialise_VariableEddingtonFactor(SuOlson_RT_args, SuOlson_sim_params)
    param_RT_solve = create_params_lambda_solver_function(VEF_equations, VEF_RT_args, VEF_sim_params)
    a = jnp.array(optimal_params['VEF']['a'])
    b = jnp.array(optimal_params['VEF']['b'])
    VEF_sol = param_RT_solve(a,b)
    return jnp.sum(VEF_sol[:,0,:]+VEF_sol[:,2,:],axis=1)*VEF_RT_args['dx']

def run_HFLD(SuOlson_RT_args, SuOlson_sim_params, optimal_params):
    # Flux Limited Diffusion solution
    ga = jnp.array(optimal_params['TMC']['a'])
    gb = jnp.array(optimal_params['TMC']['b'])
    FLD_RT_args, FLD_sim_params, FLD_equations = initialise_FluxLimitedDiffusion(SuOlson_RT_args, SuOlson_sim_params, ML_Levermore_fluxlimiter, ga, gb)
    param_RT_solve = create_params_lambda_solver_function(FLD_equations, FLD_RT_args, FLD_sim_params)
    a = jnp.array(optimal_params['FLD']['a'])
    b = jnp.array(optimal_params['FLD']['b'])
    FLD_sol = param_RT_solve(a,b)
    return jnp.sum(FLD_sol[:,0,:]+FLD_sol[:,1,:],axis=1)*FLD_RT_args['dx']

def test_SN():
    SuOlson_RT_args, SuOlson_sim_params, optimal_params, analytic_solution = initialise()
    test_solution = run_SN(SuOlson_RT_args, SuOlson_sim_params, optimal_params)
    assert (jnp.mean((test_solution-analytic_solution)**2) < test_tol)
    
def test_TMC():
    SuOlson_RT_args, SuOlson_sim_params, optimal_params, analytic_solution = initialise()
    test_solution = run_TMC(SuOlson_RT_args, SuOlson_sim_params, optimal_params)
    assert (jnp.mean((test_solution-analytic_solution)**2) < test_tol)

def test_VEF():
    SuOlson_RT_args, SuOlson_sim_params, optimal_params, analytic_solution = initialise()
    test_solution = run_VEF(SuOlson_RT_args, SuOlson_sim_params, optimal_params)
    assert (jnp.mean((test_solution-analytic_solution)**2) < test_tol)

def test_HFLD():
    SuOlson_RT_args, SuOlson_sim_params, optimal_params,analytic_solution = initialise()
    test_solution = run_HFLD(SuOlson_RT_args, SuOlson_sim_params, optimal_params)
    assert (jnp.mean((test_solution-analytic_solution)**2) < test_tol)
