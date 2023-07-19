import jax.numpy as jnp
import jax
import optax

def learn_closure(a0,b0,params_ClosureLoss,nsteps,learning_rate = 5e-4,verbose=True):
    """
    
    Simple wrapper for the optimisation process

    Adam optimizer is used to perform gradient descent

    a0, b0 : initial guess for a and b parameters
    params_ClosureLoss : loss function which is function of parameters (a,b) only
    nsteps : number of gradient descent steps taken
    learning_rate : step size taken in gradient descent

    """

    optimizer = optax.adam(learning_rate)
    params = {'a': a0,'b' : b0}
    opt_state = optimizer.init(params)

    loss_history = []
    for i in range(nsteps):
        loss, grad_loss = jax.value_and_grad(params_ClosureLoss)(params)

        updates, opt_state = optimizer.update(grad_loss, opt_state)
        params = optax.apply_updates(params, updates)
        loss_history.append(loss)

        if(verbose):
            print(loss,params)

        # Check for NaN'ed output
        if(jnp.isnan(loss)):
            from sys import exit
            print('WARNING - optimisation created NaN loss...')
            exit()

    return loss_history, params