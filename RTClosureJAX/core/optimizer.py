import jax.numpy as jnp
import jax
import optax

def create_params_lambda_loss_function(params_RT_solve,analyticsol,RT_args):
    """
    
    Wrapper on the loss function such that it is a callable function of the free parameters a and b only
    
    """
    params_ClosureLoss = lambda params : ClosureLoss(params,params_RT_solve,analyticsol,RT_args)
    return params_ClosureLoss


def ClosureLoss(params,model,analyticsol,sim_params):
    """
    
    Loss function used to optimise closure parameters e.g. Mean Squared Error

    """
    pred_sol = model(params['a'],params['b'])
    Wsol = pred_sol[:,0,:]

    loss = 0.0
    for it in range(sim_params['Nt']):
        loss += jnp.sum((jnp.interp(analyticsol['x'], sim_params['x'], Wsol[it,:])-analyticsol['W'][it,:])**2)

    return loss

def create_piecewise_learning_rate_schedule(init_value,total_steps,decay_rate,boundaries):
    boundaries_and_scales = {int(total_steps*b) : decay_rate for b in boundaries}
    learning_rate_schedule = optax.piecewise_constant_schedule(init_value,boundaries_and_scales=boundaries_and_scales)
    return learning_rate_schedule

def learn_closure(a0,b0,params_ClosureLoss,nsteps,learning_rate_schedule,verbose=True):
    """
    
    Simple wrapper for the optimisation process

    Adam optimizer is used to perform gradient descent

    a0, b0 : initial guess for a and b parameters
    params_ClosureLoss : loss function which is function of parameters (a,b) only
    nsteps : number of gradient descent steps taken
    learning_rate_schedule : step size schedule to be taken in gradient descent

    """

    optimizer = optax.adam(learning_rate=learning_rate_schedule)
    params = {'a': a0,'b' : b0}
    opt_state = optimizer.init(params)

    loss_history = []
    for i in range(nsteps):
        loss, grad_loss = jax.value_and_grad(params_ClosureLoss)(params)

        updates, opt_state = optimizer.update(grad_loss, opt_state)
        params = optax.apply_updates(params, updates)
        loss_history.append(loss)

        if(verbose):
            print(f"Step: {i}, Loss {loss}")
            print(params)
            print("~~~~~~~~~~~~~~~~~~~~~~~~")

        # Check for NaN'ed output
        if(jnp.isnan(loss)):
            from sys import exit
            print('WARNING - optimisation created NaN loss...')
            exit()

    return loss_history, params