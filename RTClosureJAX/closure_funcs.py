import jax.numpy as jnp
import jax

# Divide by zero safety factor
delta = 1e-10

def create_lambda_params_constrained_pade(f0, f1, dfdy1 = None, fx = None):
    """
    
    Returns constrained Pade form which is function of x and free parameters only
    
    """
    if(dfdy1 is None):
        pade = lambda x,a,b : constrained_pade(x,a,b,f0,f1,fx)
        coeffs = lambda a,b : constrained_pade_coeff(a,b,f0,f1,fx)
    else:
        pade = lambda x,a,b : constrained_pade_w_grad(x,a,b,f0,f1,dfdy1)
        coeffs = lambda a,b : constrained_pade_coeff_w_grad(a,b,f0,f1,dfdy1)
    return pade,coeffs

def constrained_pade_w_grad(x,a,b,f0,f1,dfdy1):
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

def constrained_pade_coeff_w_grad(a,b,f0,f1,dfdy1):
    """
    
    Returns the coefficients of the constrained Pade approximant

    a, b : free parameters
    f0 : constraint on value at x = 0
    f1 : constraint on value at x = 1
    dfdy1 : constraint on gradient at x = 1
    
    """
    b_dom = jnp.append(b,1.0)
    a_num = jnp.append(a,f1)
    a_num = jnp.insert(a_num,-1,-dfdy1+b_dom[-2])
    a_sum = jnp.sum(a_num)
    b_sum = jnp.sum(b_dom)
    a_num = jnp.insert(a_num,0,f0*b_sum-a_sum)
    return a_num,b_dom

def constrained_pade(x,a,b,f0,f1,fx):
    """
    
    Constrained Pade approximant with 0 < x < 1

    a, b : free parameters
    f0 : constraint on value at x = 0
    f1 : constraint on value at x = 1
    fx : constraint on value at x, comes as x,y pair
    
    """
    y = 1-x
    y = jnp.where(y > 1, 1, y)
    y = jnp.where(y < 0, 0, y)
    b_dom = jnp.append(b,1.0)
    b_sum = jnp.sum(b_dom)

    a_num = jnp.append(a,f1)

    xs,fs = fx
    ys = 1-xs
    N = a.size+2
    M = b_dom.size
    ysN = ys**N
    bm_sum = fs*jnp.sum(b_dom[::-1]*ys**(jnp.arange(0,M)))-f0*b_sum*ysN
    an_term = jnp.append(jnp.array([f1,0.0]),a[::-1])
    delta_an = ys**(jnp.arange(0,N))-ysN
    an_sum = jnp.sum(an_term*delta_an)
    a1 = (bm_sum-an_sum)/(ys-ysN)
    a_num = jnp.insert(a_num,-1,a1)

    a_sum = jnp.sum(a_num)
    a_num = jnp.insert(a_num,0,f0*b_sum-a_sum)
    f = jnp.polyval(a_num,y)/jnp.polyval(b_dom,y)
    return f

def constrained_pade_coeff(a,b,f0,f1,fx):
    """
    
    Returns the coefficients of the constrained Pade approximant

    a, b : free parameters
    f0 : constraint on value at x = 0
    f1 : constraint on value at x = 1
    fx : constraint on value at x, comes as x,y pair
    
    """
    b_dom = jnp.append(b,1.0)
    b_sum = jnp.sum(b_dom)

    a_num = jnp.append(a,f1)

    xs,fs = fx
    ys = 1-xs
    N = a.size+2
    M = b_dom.size
    ysN = ys**N
    bm_sum = fs*jnp.sum(b_dom[::-1]*ys**(jnp.arange(0,M)))-f0*b_sum*ysN
    an_term = jnp.append(jnp.array([f1,0.0]),a[::-1])
    delta_an = ys**(jnp.arange(0,N))-ysN
    an_sum = jnp.sum(an_term*delta_an)
    a1 = (bm_sum-an_sum)/(ys-ysN)
    a_num = jnp.insert(a_num,-1,a1)

    a_sum = jnp.sum(a_num)
    a_num = jnp.insert(a_num,0,f0*b_sum-a_sum)
    return a_num,b_dom

def unconstrained_pade(x,a,b):
    """
    
    Pade approximant with 0 < x < 1

    a, b : free parameters
    
    """
    x = jnp.where(x > 1, 1, x)
    x = jnp.where(x < 0, 0, x)
    f = jnp.polyval(a,x)/jnp.polyval(b,x)
    return f

def ML_Levermore_fluxlimiter(p,R_sq,Closure,a,b):
    """
    
    Levermore's relationship between flux limiter, Eddington factor and normalised energy gradient

    p : Eddington factor (0 < p < 1)
    R_sq : Squared normalised gradient
    
    """
    p = jnp.where(p > 1, 1, p)
    p = jnp.where(p < 0, 0, p)
    R_sq = jnp.where(R_sq < 0, 0, R_sq)
    det      = 1+4*R_sq*(Closure(p,a,b))
    sqrt_det = jnp.sqrt(det)
    return (sqrt_det-1)/(2*(R_sq+delta))

def Levermore_fluxlimiter(p,R_sq,Closure,a,b):
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

def Larsen_2_fluxlimiter(p,R_sq,Closure,a,b):
    """
    
    Larsen's flux limiter with n = 2

    N.B. independent of Eddington factor, p
    
    """
    return 1.0/jnp.sqrt(9+R_sq)

def diffusion_fluxlimter(p,R_sq,Closure,a,b):
    """
    
    Returns diffusive limit i.e. no flux limiter

    N.B. independent of all inputs

    """
    return 1.0/3.0*jnp.ones_like(p)