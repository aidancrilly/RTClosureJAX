import jax.numpy as jnp
import jax
import numpy as np

from .utils import SimDataDir
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

def initialise_VariableEddingtonFactor(RT_args, sim_params, dt_mult = 1e-1, stability_const = 0.1):
    VEF_RT_args    = RT_args.copy()
    VEF_sim_params = sim_params.copy()

    VEF_RT_args['Np'] = 3
    VEF_sim_params['dt0'] = dt_mult*VEF_RT_args['dx']
    VEF_RT_args['stability_const'] = stability_const

    VEF_RT_args['Closure'] = create_lambda_params_constrained_pade(1.0/3.0,1.0,2.0)
    VEF_RT_args['GradClosure'] = jax.vmap(jax.grad(VEF_RT_args['Closure']),in_axes = (0,None,None))

    return VEF_RT_args, VEF_sim_params, VEF_RT_equations

def VEFRoeSolve(delta_u,Fl,Fr,C,D,stability_const):
    """
    
    Following "One-dimensional Riemann solvers and the maximum entropy closure" by Brunner and Holloway

    Using the Roe matrix:

    R(u_l,u_r) = ( 0, 1)
                 ( D, C)

    Extra dissipation is added as eigenvalues can go through zero
    
    """

    # Eigenvalues
    det = C**2+4*D
    sqrtdet = jnp.sqrt(jnp.where(det > 0.0, det , 0.0))
    lambda_plus  = 0.5*(C+sqrtdet)
    lambda_minus = 0.5*(C-sqrtdet)
    # Right eigenvectors
    r_plus  = jnp.array((-lambda_minus,D))/jnp.sqrt(1+lambda_minus**2)
    r_minus = jnp.array((-lambda_plus,D))/jnp.sqrt(1+lambda_plus**2)
    # Left eigenvectors
    l_plus  = jnp.array(( lambda_plus-C ,1.0))/jnp.sqrt(1+(lambda_plus-C)**2)
    l_minus = jnp.array(( lambda_minus-C,1.0))/jnp.sqrt(1+(lambda_minus-C)**2)

    CorrectionTerm = jnp.abs(lambda_plus)*jnp.outer(r_plus,l_plus)+jnp.abs(lambda_minus)*jnp.outer(r_minus,l_minus)
    # Adding dissipation since eigenvalues can go to zero
    CorrectionTerm += stability_const*jnp.eye(2)

    RoeTerm = jnp.matmul(CorrectionTerm,delta_u)

    flux = 0.5*(Fl+Fr)-0.5*RoeTerm
    return flux

vectorised_VEFRoeSolve = jax.vmap(VEFRoeSolve,in_axes=(1,1,1,0,0,None),out_axes=1)

def VEFComputeRoeCoefficients(Closure,GradClosure,FR,p,a,b):
    """

    D = (p_l f_r - p_r f_l)/(f_r - f_l)
    C = (p_r - p_l)/(f_r - f_l)

    where p is the Eddington factor and f is the flux ratio
    
    """

    f_l = FR[:-1]
    f_r = FR[1:]
    p_l = p[:-1]
    p_r = p[1:]

    face_f  = 0.5*(f_r+f_l)
    delta_f = (f_r-f_l)
    sign_f_f = jnp.sign(face_f)
    abs_d_f = jnp.abs(delta_f)

    C = jnp.where(abs_d_f > 1e-5, (p_r-p_l)/(delta_f+delta), sign_f_f*GradClosure(jnp.abs(face_f),a,b))
    D = jnp.where(abs_d_f > 1e-5, (p_l*f_r-p_r*f_l)/(delta_f+delta), Closure(jnp.abs(face_f),a,b)-sign_f_f*face_f*GradClosure(jnp.abs(face_f),a,b))

    return C,D

def VEF_RT_equations(t,y,args):
    # Unpack parameters
    a = args['a']
    b = args['b']
    SourceTerm = args['SourceTerm']
    epsilon = args['epsilon']
    dx = args['dx']
    Np = args['Np']
    Nx = args['Nx']
    GradClosure = args['GradClosure']
    Closure = args['Closure']
    stability_const = args['stability_const']

    y = y.reshape(Np,Nx)
    W,F,V = y[0,:],y[1,:],y[2,:]
    Q = SourceTerm(t)
    # N.B. flux F is now cell centred for use in the Riemann solver

    # Add ghost cell
    F_ghost = jnp.insert(F,0,-F[0])
    F_ghost = jnp.append(F_ghost,-F[-1])
    W_ghost = jnp.insert(W,0,W[0])
    W_ghost = jnp.append(W_ghost,W[-1])

    # Source and sink terms
    dWdt = (Q+V-W)/epsilon
    dFdt = (-F)/epsilon
    dVdt = W-V

    # Eddington factor calculation
    FR = F_ghost/(W_ghost+delta) 
    p = Closure(jnp.abs(FR),a,b)

    # Riemann problem using Roe-type solver
    C,D = VEFComputeRoeCoefficients(Closure,GradClosure,FR,p,a,b)
    # Assemble left and right properties
    delta_u = jnp.vstack((W_ghost[1:]-W_ghost[:-1],F_ghost[1:]-F_ghost[:-1]))
    Fl = jnp.vstack((F_ghost[:-1],p[:-1]*W_ghost[:-1]))
    Fr = jnp.vstack((F_ghost[1:],p[1:]*W_ghost[1:]))
    RoeFlux = vectorised_VEFRoeSolve(delta_u,Fl,Fr,C,D,stability_const)
    # Add advective terms onto time derivatives
    dWdt += -jnp.diff(RoeFlux[0,:],n=1)/dx/epsilon
    dFdt += -jnp.diff(RoeFlux[1,:],n=1)/dx/epsilon

    dydt = jnp.vstack((dWdt,dFdt,dVdt))
    return dydt.flatten()

# Third order moment equations

def initialise_ThirdOrderMoment(RT_args, sim_params, dt_mult = 1e-1):
    TMC_RT_args    = RT_args.copy()
    TMC_sim_params = sim_params.copy()

    TMC_RT_args['Np'] = 4
    TMC_sim_params['dt0'] = dt_mult*TMC_RT_args['dx']

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

def initialise_FluxLimitedDiffusion(RT_args, sim_params, fluxlimiter, dt_mult = 1e-2):
    FLD_RT_args    = RT_args.copy()
    FLD_sim_params = sim_params.copy()

    FLD_RT_args['Np'] = 3
    # Smaller time step needed to control the diffusive branch
    FLD_sim_params['dt0'] = dt_mult*FLD_RT_args['dx']

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

    # Normalised gradient
    diff_W = jnp.diff(W_ghost,n=1)
    W_face = jnp.where(diff_W < 0, W_ghost[:-1], W_ghost[1:])
    R      = diff_W/(W_face+delta)/dx
    R_sq   = R**2
    # Eddington factor
    EF       = P_ghost/(W_ghost+delta)
    EF_face  = jnp.where(diff_W < 0, EF[:-1] , EF[1:])
    p        = EF_face
    # Flux limited radiative flux calculation
    lamb      = FluxLimiter(p,R_sq)
    F_ghost   = -lamb*(diff_W/dx)

    # Radiation energy density update
    divF = jnp.diff(F_ghost,n=1)/dx
    dWdt = (Q+V-W-divF)/epsilon

    # Radiation pressure factor update
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

def initialise_DiscreteOrdinates(RT_args, sim_params,dt_mult = 1e-1):
    SN_RT_args    = RT_args.copy()
    SN_sim_params = sim_params.copy()

    Nm = mun.shape[0]
    SN_RT_args['Np'] = Nm+1
    SN_sim_params['dt0'] = dt_mult*SN_RT_args['dx']

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