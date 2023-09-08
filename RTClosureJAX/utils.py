import jax.numpy as jnp
import jax
import numpy as np
import json

SuOlsonDataDir = r"C:\\Users\\Aidan Crilly\\Documents\\GitHub\\RTClosureJAX\\SuOlsonData\\"
SimDataDir = r"C:\\Users\\Aidan Crilly\\Documents\\GitHub\\RTClosureJAX\\SimData\\"

def DualExternalSource(x,tau,x0,tau0,dx):
    """
    
    Source term in the 'modified' Su Olson problem

    Tanh used over Heaviside in spatial dimension to smooth discontinuous behaviour

    """
    S = 0.5*(jnp.tanh((x0-x)/(dx))+1+jnp.tanh((x0-(x[-1]-x))/(dx))+1)*jnp.heaviside(tau0-tau,0.5)/(2*x0)
    return S

def ExternalSource(x,tau,x0,tau0,dx):
    """
    
    Source term in the original Su Olson problem

    Tanh used over Heaviside in spatial dimension to smooth discontinuous behaviour

    """
    S = 0.5*(jnp.tanh((x0-x)/(dx))+1)*jnp.heaviside(tau0-tau,0.5)/(2*x0)
    return S

def initialise_SuOlson_problem(Nx,L=6.0,Nt=5):
    """
    
    Set up for the Su Olson problem

    Nx : number of spatial cells

    """
    xb = jnp.linspace(0.0,L,Nx+1)
    x = 0.5*(xb[1:]+xb[:-1])
    dx = x[1]-x[0]

    x0 = 0.5
    tau0 = 10.0
    SourceTerm = lambda tau : ExternalSource(x,tau,x0,tau0,dx)
    epsilon = 1.0

    t_data = np.loadtxt(f"{SuOlsonDataDir}SuOlsont.dat")

    tsave  = jnp.array(t_data[:Nt])

    # Arguments needed explicitly by the RT equations
    SuOlson_RT_args = {
        'Nx' : Nx,
        'dx' : dx,
        'SourceTerm' : SourceTerm,
        'epsilon' : epsilon
    }

    # Additional simulation parameters needed by the solver
    SuOlson_sim_params = {
        'x' : x,
        'tsave' : tsave,
        'Nt' : tsave.shape[0]
    }

    return SuOlson_RT_args,SuOlson_sim_params

def initialise_ModifiedSuOlson_problem(Nx,L=4.0):
    """
    
    Set up for the modified Su Olson problem

    Nx : number of spatial cells

    """

    xb = jnp.linspace(0.0,L,Nx+1)
    x = 0.5*(xb[1:]+xb[:-1])
    dx = x[1]-x[0]

    x0 = 0.5
    tau0 = 100.0
    SourceTerm = lambda tau : DualExternalSource(x,tau,x0,tau0,dx)
    epsilon = 1.0

    tsave  = jnp.array([1.0,1.5,2.0,4.0,10.0])

    # Arguments needed explicitly by the RT equations
    ModifiedSuOlson_RT_args = {
        'Nx' : Nx,
        'dx' : dx,
        'SourceTerm' : SourceTerm,
        'epsilon' : epsilon
    }

    # Additional simulation parameters needed by the solver
    ModifiedSuOlson_sim_params = {
        'x' : x,
        'tsave' : tsave,
        'Nt' : tsave.shape[0]
    }

    return ModifiedSuOlson_RT_args,ModifiedSuOlson_sim_params

def get_SuOlson_analytic_solution(SuOlson_sim_params):
    """
    
    Loads Su Olson analytic solution and returns data dictionary
    
    """
    x_data = np.loadtxt(f"{SuOlsonDataDir}SuOlsonx.dat")
    W_data  = np.loadtxt(f"{SuOlsonDataDir}SuOlson_W_transport.dat")
    V_data  = np.loadtxt(f"{SuOlsonDataDir}SuOlson_V_transport.dat")
    WD_data = np.loadtxt(f"{SuOlsonDataDir}SuOlson_W_diffusion.dat")
    VD_data = np.loadtxt(f"{SuOlsonDataDir}SuOlson_V_diffusion.dat")

    Nt = SuOlson_sim_params['Nt']

    analyticsol = {'x' : x_data, 'W' : W_data.T[:Nt,:], 'V' : V_data.T[:Nt,:], 'W_diffusion' : WD_data.T[:Nt,:], 'V_diffusion' : VD_data.T[:Nt,:]}

    return analyticsol

def load_optimal_params(opt_param_file):
    """
    
    Loads previously trained closure parameters from .json file
    
    """
    if(opt_param_file.split('.')[-1] == 'json'):
        with open(SimDataDir+opt_param_file, 'r') as json_file:
            json_dict = json.load(json_file)
        return json_dict
    else:
        print("Expected JSON file for optimal parameter data...")
        print("> Returning None")
        return None