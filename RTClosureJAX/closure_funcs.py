import jax.numpy as jnp
import jax

# Divide by zero safety factor
delta = 1e-10

def create_lambda_params_constrained_pade(y0,y1,dfdy1):
    """
    
    Returns constrained Pade form which is function of x and free parameters only
    
    """
    return lambda x,a,b : constrained_pade(x,a,b,y0,y1,dfdy1)

def constrained_pade(x,a,b,f0,f1,dfdy1):
    """
    
    Constrained Pade approximant with 0 < x < 1

    a, b : free parameters
    f0 : constraint on value at x = 0
    f1 : constraint on value at x = 1
    dfdy1 : constraint on gradient at x = 1
    
    """
    y = 1-x
    y = jnp.where(y > 1, 1, y)
    y = jnp.where(y < 0, 0, y)
    b_dom = jnp.append(b,1.0)
    a_num = jnp.append(a,f1)
    a_num = jnp.insert(a_num,-1,-dfdy1+b_dom[-2])
    a_sum = jnp.sum(a_num)
    b_sum = jnp.sum(b_dom)
    a_num = jnp.insert(a_num,0,f0*b_sum-a_sum)
    f = jnp.polyval(a_num,y)/jnp.polyval(b_dom,y)
    return f

def unconstrained_pade(x,a,b):
    """
    
    Pade approximant with 0 < x < 1

    a, b : free parameters
    
    """
    y = 1-x
    y = jnp.where(y > 1, 1, y)
    y = jnp.where(y < 0, 0, y)
    f = jnp.polyval(a,y)/jnp.polyval(b,y)
    return f

def Levermore_fluxlimiter(p,R_sq):
    """
    
    Levermore's relationship between flux limiter, Eddington factor and normalised energy gradient

    p : Eddington factor (0 < p < 1)
    R_sq : Squared normalised gradient
    
    """
    p = jnp.where(p > 1, 1, p)
    p = jnp.where(p < 0, 0, p)
    R_sq = jnp.where(R_sq < 0, 0, R_sq)
    det      = 1+4*R_sq*p
    sqrt_det = jnp.sqrt(det)
    return (sqrt_det-1)/(2*(R_sq+delta))

def Larsen_2_fluxlimiter(p,R_sq):
    """
    
    Larsen's flux limiter with n = 2

    N.B. independent of Eddington factor, p
    
    """
    return 1.0/jnp.sqrt(9+R_sq)

def diffusion_fluxlimter(p,R_sq):
    """
    
    Returns diffusive limit i.e. no flux limiter

    N.B. independent of all inputs

    """
    return 1.0/3.0*jnp.ones_like(p)