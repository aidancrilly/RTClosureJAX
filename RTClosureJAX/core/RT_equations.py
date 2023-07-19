import jax.numpy as jnp
import jax
import numpy as np

SimDataDir = r"C:\\Users\\Aidan Crilly\\Documents\\GitHub\\RTClosureJAX\\SimData\\"

from .closure_funcs import *

""" 
Radiation transport equations (pure absorption case)
W = Radiation energy density
F = Radiation energy flux
V = Material energy density
Q = Radiation source term
P = Radiation pressure
epsilon = 4 (radiation constant)/(heat capacity prefactor)

x = normalised position coordinate
tau/t = normalised time coordinate

Reflective boundaries
- for Su Olson problem, left hand boundary placed sufficiently far away

"""

# Divide by zero safety factor
delta = 1e-10

# Variable Eddington factor

def initialise_VariableEddingtonFactor(RT_args, sim_params):
    VEF_RT_args    = RT_args.copy()
    VEF_sim_params = sim_params.copy()

    VEF_RT_args['Np'] = 3
    VEF_sim_params['dt0'] = 1e-1*VEF_RT_args['dx']

    VEF_RT_args['Closure'] = create_lambda_params_constrained_pade(1.0/3.0,1.0,2.0)

    return VEF_RT_args, VEF_sim_params, VEF_RT_equations

def VEF_RT_equations(t,y,args):
    # Unpack parameters
    a = args['a']
    b = args['b']
    SourceTerm = args['SourceTerm']
    epsilon = args['epsilon']
    dx = args['dx']
    Np = args['Np']
    Nx = args['Nx']
    Closure = args['Closure']

    y = y.reshape(Np,Nx)
    W,F,V = y[0,:],y[1,:],y[2,:]
    Q = SourceTerm(t)

    # Add ghost cell
    F_ghost = jnp.insert(F,0,0.0)
    W_ghost = jnp.insert(W,0,W[0])
    W_ghost = jnp.append(W_ghost,W[-1])

    # Radiation energy density update
    divF = jnp.diff(F_ghost,n=1)/dx
    dWdt = (Q+V-W-divF)/epsilon

    # Eddington factor calculation
    W_face   = 0.5*(W_ghost[1:]+W_ghost[:-1])
    VEF_face = jnp.abs(F_ghost)/(W_face+delta)
    p_face   = Closure(VEF_face,a,b)
    p = 0.5*(p_face[1:]+p_face[:-1])
    p_ghost = jnp.append(p,p[-1])
    p_ghost = jnp.insert(p_ghost,0,p[0])
    p = 0.25*(p_ghost[:-2]+2*p_ghost[1:-1]+p_ghost[2:])

    # Radiation energy flux update
    pW = p*W
    pW_ghost = jnp.append(pW,pW[-1])
    F_ghost = jnp.append(F_ghost,F_ghost[1])
    dFdt = (-jnp.diff(pW_ghost,n=1)/dx-F)/epsilon

    # Material energy density update
    dVdt = W-V

    dydt = jnp.vstack((dWdt,dFdt,dVdt))
    return dydt.flatten()

# Third order moment equations

def initialise_ThirdOrderMoment(RT_args, sim_params):
    TMC_RT_args    = RT_args.copy()
    TMC_sim_params = sim_params.copy()

    TMC_RT_args['Np'] = 4
    TMC_sim_params['dt0'] = 1e-1*TMC_RT_args['dx']

    TMC_RT_args['Closure'] = create_lambda_params_constrained_pade(0.0,1.0,1.0)

    return TMC_RT_args, TMC_sim_params, TMC_RT_equations

def TMC_RT_equations(t,y,args):
    # Unpack parameters
    a = args['a']
    b = args['b']
    SourceTerm = args['SourceTerm']
    epsilon = args['epsilon']
    dx = args['dx']
    Np = args['Np']
    Nx = args['Nx']
    Closure = args['Closure']

    y = y.reshape(Np,Nx)
    W,F,V,P = y[0,:],y[1,:],y[2,:],y[3,:]
    Q = SourceTerm(t)

    # Add ghost cells
    W_ghost = jnp.insert(W,0,W[1])
    W_ghost = jnp.append(W_ghost,W[-1])
    P_ghost = jnp.insert(P,0,P[1])
    P_ghost = jnp.append(P_ghost,P[-1])
    F_ghost = jnp.insert(F,0,0.0)

    # Radiation energy density update
    divF = jnp.diff(F_ghost,n=1)/dx
    dWdt = (Q+V-W-divF)/epsilon

    # Radiation energy flux update
    dFdt = (-jnp.diff(P_ghost[1:],n=1)/dx-F)/epsilon

    # Eddington factor calculation
    EF = P_ghost/(W_ghost+delta)
    EF_face = jnp.where(F_ghost < 0, EF[1:] , EF[:-1])

    # Radiation pressure factor update
    divaF = jnp.diff(Closure(EF_face,a,b)*F_ghost,n=1)/dx
    dPdt = ((Q+V-3*P)/3.0-divaF)/epsilon

    # Material energy density update
    dVdt = W-V

    dydt = jnp.vstack((dWdt,dFdt,dVdt,dPdt))
    return dydt.flatten()

# Flux-Limited Diffusion

def initialise_FluxLimitedDiffusion(RT_args, sim_params, fluxlimiter):
    FLD_RT_args    = RT_args.copy()
    FLD_sim_params = sim_params.copy()

    FLD_RT_args['Np'] = 3
    FLD_sim_params['dt0'] = 1e-1*FLD_RT_args['dx']

    FLD_RT_args['Closure'] = create_lambda_params_constrained_pade(0.0,1.0,1.0)
    FLD_RT_args['FluxLimiter'] = fluxlimiter

    return FLD_RT_args, FLD_sim_params, FLD_RT_equations

def FLD_RT_equations(t,y,args):
    # Unpack parameters
    a = args['a']
    b = args['b']
    SourceTerm = args['SourceTerm']
    epsilon = args['epsilon']
    dx = args['dx']
    Np = args['Np']
    Nx = args['Nx']
    Closure = args['Closure']
    FluxLimiter = args['FluxLimiter']

    y = y.reshape(Np,Nx)
    W,V,P = y[0,:],y[1,:],y[2,:]
    Q = SourceTerm(t)

    # Add ghost cells
    W_ghost = jnp.insert(W,0,W[1])
    W_ghost = jnp.append(W_ghost,W[-1])
    P_ghost = jnp.insert(P,0,P[1])
    P_ghost = jnp.append(P_ghost,P[-1])

    # Eddington factor & 
    EF = P_ghost/(W_ghost+delta)
    EF_face = 0.5*(EF[1:]+EF[:-1])
    W_face = 0.5*(W_ghost[:-1]+W_ghost[1:])
    # Normalised gradient
    diff_W = jnp.diff(W_ghost,n=1)
    R = diff_W/dx/(W_face+delta)
    R_sq = R**2
    # Flux limited radiative flux calculation
    p        = EF_face
    lamb     = FluxLimiter(p,R_sq)
    F_ghost  = -lamb*R*jnp.where(diff_W < 0, W_ghost[:-1], W_ghost[1:])

    # Radiation energy density update
    divF = jnp.diff(F_ghost,n=1)/dx
    dWdt = (Q+V-W-divF)/epsilon

    # Radiation pressure factor update
    EF_face = jnp.where(F_ghost < 0, EF[1:] , EF[:-1])
    divaF = jnp.diff(Closure(EF_face,a,b)*F_ghost,n=1)/dx
    dPdt = ((Q+V-3*P)/3.0-divaF)/epsilon

    # Material energy density update
    dVdt = W-V

    dydt = jnp.vstack((dWdt,dVdt,dPdt))
    return dydt.flatten()

# Discrete ordinates
# S32

mun = jnp.array(np.loadtxt(f"{SimDataDir}S32_mu.dat"))
wn = jnp.array(np.loadtxt(f"{SimDataDir}S32_w.dat"))

def initialise_DiscreteOrdinates(RT_args, sim_params):
    SN_RT_args    = RT_args.copy()
    SN_sim_params = sim_params.copy()

    Nm = mun.shape[0]
    SN_RT_args['Np'] = Nm+1
    SN_sim_params['dt0'] = 1e-1*SN_RT_args['dx']

    return SN_RT_args, SN_sim_params, DiscreteOrdinates_RT_equations


def SN_forward_sweep(m,dpsidt_arr,psi,psi0,V,Q,epsilon,dx):
    psim = psi[m,:]
    psi_ghost = jnp.insert(psim,0,psi0)
    dpsidt = (-mun[m]*jnp.diff(psi_ghost,n=1)/dx-psim+V/2+Q/2)/epsilon
    dpsidt_arr = dpsidt_arr.at[m,:].add(dpsidt)
    return dpsidt_arr

def SN_backward_sweep(m,dpsidt_arr,psi,psi0,V,Q,epsilon,dx):
    psim = psi[m,:]
    psi_ghost = jnp.append(psim,psi0)
    dpsidt = (-mun[m]*jnp.diff(psi_ghost,n=1)/dx-psim+V/2+Q/2)/epsilon
    dpsidt_arr = dpsidt_arr.at[m,:].add(dpsidt)
    return dpsidt_arr

def DiscreteOrdinates_RT_equations(t,y,args):
    # Unpack parameters
    SourceTerm = args['SourceTerm']
    epsilon = args['epsilon']
    dx = args['dx']
    Np = args['Np']
    Nx = args['Nx']

    y = y.reshape(Np,Nx)
    Nm = Np-1
    Q = SourceTerm(t)
    V = y[-1,:]
    psi = y[:-1,:]
    W = jnp.sum(wn[:,None]*psi,axis=0)

    # Backward sweeps
    dpsidt_arr = jnp.zeros_like(psi)
    backward_sweep_i = lambda i, dpsidt_arr : SN_backward_sweep(i,dpsidt_arr,psi,psi[Nm-i-1,-1],V,Q,epsilon,dx)
    dpsidt_arr = jax.lax.fori_loop(0, Nm//2, backward_sweep_i, dpsidt_arr)

    # Forward sweeps
    forward_sweep_i = lambda i, dpsidt_arr : SN_forward_sweep(i,dpsidt_arr,psi,psi[Nm-i-1,0],V,Q,epsilon,dx)
    dpsidt_arr = jax.lax.fori_loop(Nm//2, Nm, forward_sweep_i, dpsidt_arr)

    # Material energy density update
    dVdt = W-V

    dydt = jnp.vstack((dpsidt_arr,dVdt))
    return dydt.flatten()

def process_discrete_ordinates_sim(sol):
    """
    
    Converts a discrete ordinates solution into moment data via quadrature integration

    """
    psi = sol[:,:-1,:]
    Wsol = jnp.sum(wn[None,:,None]*psi,axis=1)
    Fsol = jnp.sum(mun[None,:,None]*wn[None,:,None]*psi,axis=1)
    Psol = jnp.sum(mun[None,:,None]**2*wn[None,:,None]*psi,axis=1)
    Asol = jnp.sum(mun[None,:,None]**3*wn[None,:,None]*psi,axis=1)
    Vsol = sol[:,-1,:]
    return Wsol,Fsol,Psol,Asol,Vsol