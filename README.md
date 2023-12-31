![tests workflow](https://github.com/aidancrilly/RTClosureJAX/actions/workflows/tests.yml/badge.svg?event=push)

# RTClosureJAX

This repository contains the code base for following publication: 
" Learning closure relations using differentiable programming: an example in radiation transport "

Uses JAX, diffrax and optax libraries.

## Author:
- Aidan Crilly

E-mail: ac116@ic.ac.uk

## Reproducing results

The 'Examples/' directory provides a number scripts that can be used to reproduce solutions to the original and modified Su & Olson problem, as well as the training process:

- RunSuOlson.py

This contains the default settings used to run solutions to the original Su & Olson problem for the various models. By default, it reads the closure parameters from the SimData/opt_closure_params.json file which contains optimal parameters found by the author using TrainSuOlson.py

- TrainSuOlson.py

This contains the default settings used to train the closure relations to the original Su & Olson problem with the various models. N.B. the higher order flux limited diffusion (HFLD) model requires an accurate g-closure from the third order moment model to run. Therefore, if training from scratch, the third order model must be trained before HFLD. Optimal parameters are saved to a json file in 'SimData/'

- RunModifiedSuOlson.py

This contains the default settings used to run solutions to the modified Su & Olson problem for the various models. By default, it reads the closure parameters from the SimData/opt_closure_params.json file.

## The Su & Olson test problem

The training data for the closures comes from the Su & Olson test problem. In this problem, the coupled system of equations to be solved are the following:

$$
\begin{aligned}
\left(\epsilon \frac{\partial}{\partial \tau} + \mu \frac{\partial}{\partial x} + 1\right) U(x,\mu,\tau) &= \frac{1}{2}V(x,\tau)+\frac{1}{2}Q(x,\tau) \ , \\
\frac{\partial V}{\partial \tau} &= W(x,\tau)-V(x,\tau) \ ,
\end{aligned}
$$

where:

$$
\begin{aligned}
W(x,\tau) \equiv \int^{+1}_{-1}U(x,\mu,\tau) d\mu \ .
\end{aligned}
$$

Here $U$, $W$, $V$ and $Q$ are the scaled radiation intensity, radiation energy density, material energy density and external radiation source respectively. The coordinates $x$, $\mu$, and $\tau$ are the normalised distance, cosine of angle of propagation and the normalised time respectively.

The analytic solutions given are performed for a source term of the following form:

$$
Q(x,\tau) = \frac{1}{2x_{0}} \left[\Theta(x+x_{0})-\Theta(x-x_{0})\right]\left[\Theta(\tau)-\Theta(\tau-\tau_{0})\right] \ ,
$$

where $\Theta$ is the Heaviside function.

Su and Olson provide analytic solutions at various $x$ and $\tau$ values for $\epsilon = 1$, $x_0$ = 0.5 and $\tau_0$ = 10. Tabular data is given in the 'SuOlsonData' directory.

## Code overview

We solve radiation transport for the Su & Olson type problems using various reduced models:
- Pure diffusion
- Flux-limited diffusion
- Variable Eddington factor
- Third order moment
- Higher order flux-limited diffusion

Examples of running and training the models are given in the 'Examples' directory

## Finite differencing details

All models are solved using finite volume methods fully explicitly. Reflective boundary conditions are applied. Time stepping was performed using Huen's method, which solves:

$$
\frac{\partial y}{\partial \tau} = f(y,\tau)
$$

by

$$
\begin{aligned}
\hat{y} &= y^j + h f(y^j,\tau^j) \ ,\\
y^{j+1} &= y^j + \frac{h}{2}\left[f(y^j,\tau^j)+f(\hat{y},\tau^{j+1})\right] \ ,
\end{aligned}
$$

where $j$ is the temporal index and $h$ is the time step.

### Flux-limited diffusion
The flux-limited diffusion equations:

$$
\begin{aligned}
\epsilon \frac{\partial W}{\partial \tau} + \frac{\partial F}{\partial x} + W  &= V+Q \ , \\
F &= - \lambda(R) \frac{\partial W}{\partial x} \ , \\
R &= \frac{1}{W}\frac{\partial W}{\partial x} \ ,
\end{aligned}
$$

were solved using the forward-time-centred-space method - making the fluxes face-centred. Therefore, the flux limiter is also needed at the face. A donor cell approach was used to do this:

$$
\begin{aligned}
\epsilon \frac{\partial W_i}{\partial \tau}  &= -\frac{F_{i+1/2}-F_{i-1/2}}{\Delta x} - W_i+V_i+Q_i \ , \\
F_{i+1/2} &= - \lambda(R_{i+1/2}) \frac{W_{i+1}-W_{i}}{\Delta x} \ , \\
R_{i+1/2} &= \frac{1}{W_{i+1/2}}\frac{W_{i+1}-W_{i}}{\Delta x}
\end{aligned}
$$

if $W_{i+1}-W_{i} > 0$ then $W_{i+1/2} = W_{i+1}$ else $W_{i+1/2} = W_i$; $i$ here is the spatial index and $\Delta x$ is the spatial step. This scheme has a stability which is poorer than the advective CFL condition ($\Delta \tau < \Delta x$) due to the diffusive terms.

### Variable Eddington factor
The Variable Eddington factor equations:

$$
\begin{aligned}
\epsilon\frac{\partial W}{\partial \tau} + \frac{\partial F}{\partial x}  &= V+Q-W \ , \\
\epsilon\frac{\partial F}{\partial \tau} + \frac{\partial}{\partial x}(p W)  &= -F \ , 
\end{aligned}
$$

with closure

$$
p\left(f = \frac{\left|F\right|}{W}\right) 
$$

were solved using a Roe scheme, following [Brunner and Holloway](https://www.sciencedirect.com/science/article/pii/S0022407300000996), with the RHS treated as source terms.

$$
\begin{aligned}
\epsilon\frac{\partial W_i}{\partial \tau} &= - \frac{R_{W,i+1/2}-R_{W,i-1/2}}{\Delta x} -W_i+V_i+Q_i \ , \\
\epsilon\frac{\partial F_i}{\partial \tau} &= - \frac{R_{F,i+1/2}-R_{F,i-1/2}}{\Delta x}-F_i \ , 
\end{aligned}
$$

where $R_W$ and $R_F$ are the numerical fluxes calculated by the Roe scheme:
<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cbg_white%20%5Cleft%28%5Cbegin%7Barray%7D%7Bc%7D%20R_%7BW%2Ci&plus;1/2%7D%20%5C%5C%20R_%7BF%2Ci&plus;1/2%7D%20%5Cend%7Barray%7D%5Cright%29%20%3D%20%5Cfrac%7B1%7D%7B2%7D%20%5Cleft%28%5Cbegin%7Barray%7D%7Bc%7D%20F_%7Bi&plus;1%7D&plus;F_%7Bi%7D%20%5C%5C%20p%28f_%7Bi&plus;1%7D%29W_%7Bi&plus;1%7D&plus;p%28f_%7Bi%7D%29W_%7Bi%7D%20%5Cend%7Barray%7D%5Cright%29%20&plus;%20%5Cfrac%7B1%7D%7B2%7D%20%5Cunderline%7BC%7D%28f_%7Bi&plus;1%7D%2Cf_%7Bi%7D%29%20%5Ccdot%20%5Cleft%28%5Cbegin%7Barray%7D%7Bc%7D%20W_%7Bi&plus;1%7D-W_%7Bi%7D%20%5C%5C%20F_%7Bi&plus;1%7D-F_%7Bi%7D%20%5Cend%7Barray%7D%5Cright%29%20%5C%20%2C" />
</p>

where the correction matrix, $\underline{C}$, is given in Brunner and Holloway. In this scheme both flux and energy density are cell centred. The scheme's stability follows the advective CFL condition.

### Third order moment
The Third Order Moment equations:

$$
\begin{aligned}
\epsilon \frac{\partial W}{\partial \tau} + \frac{\partial F}{\partial x} + W  &= V+Q \ , \\
\epsilon \frac{\partial F}{\partial \tau} + \frac{\partial P}{\partial x} + F  &= 0 \ , \\
\epsilon \frac{\partial P}{\partial \tau} + \frac{\partial}{\partial x}(g F) + P  &= \frac{1}{3}(V+Q) \ ,
\end{aligned}
$$

with closure

$$
g\left(p = \frac{P}{W}\right)
$$

were solved using face-centred fluxes and first order finite differencing. The donor cell Eddington factor ($p$) was used to get a face centred value of the closure in the pressure equation:

$$
\epsilon \frac{\partial P_i}{\partial \tau} = -\frac{g(p_{i+1/2})F_{i+1/2}-g(p_{i-1/2})F_{i-1/2}}{\Delta x} - P_i + \frac{1}{3}(V_i+Q_i)
$$

if $F_{i+1/2} > 0$ then $p_{i+1/2} = \frac{P_{i}}{W_{i}}$ else $p_{i+1/2} = \frac{P_{i+1}}{W_{i+1}}$. The scheme's stability follows the advective CFL condition.

### Higher order flux-limited diffusion
The higher order flux-limited diffusion model:

$$
\begin{aligned}
\epsilon \frac{\partial W}{\partial \tau} + \frac{\partial F}{\partial x} + W  &= V+Q \ , \\
F &= - \lambda(R,p) \frac{\partial W}{\partial x} \ , \\
\epsilon \frac{\partial P}{\partial \tau} + \frac{\partial}{\partial x}(g F) + P  &= \frac{1}{3}(V+Q) \ ,
\end{aligned}
$$

used a similar donor cell method to obtain face-centred values, as in flux-limited diffusion and third order moment models. This scheme has a stability which is poorer than the advective CFL condition due to the diffusive terms.

### Discrete Ordinates

Discrete ordinates numerically solve the full radiation transport equation using a Gauss-Legendre quadrature set for the angular coordinate, $\mu$. This goes as follows, for $\mu > 0$:

$$
\epsilon \frac{\partial U_{i,n}}{\partial \tau}= - \mu_n \frac{U_{i,n}-U_{i-1,n}}{\Delta x}-U_{i,n}+\frac{1}{2}V_i+\frac{1}{2}Q_i \ , 
$$

for $\mu$ < 0:

$$
\epsilon \frac{\partial U_{i,n}}{\partial \tau}= - \mu_n \frac{U_{i+1,n}-U_{i,n}}{\Delta x}-U_{i,n}+\frac{1}{2}V_i+\frac{1}{2}Q_i \ ,
$$

where $n$ is the angular index. For each ordinate $\mu_n$, there is an ordinate weight, $w_n$. These are used to compute angular moments of $U_{i,n}$, for example:

$$
W_{i} = \sum_n w_n U_{i,n} \ .
$$

The $\mu_n$ and $w_n$ values for a 32 ordinates set are stored in files under the 'SimData' directory.

## Tests

A number of tests of the numerical schemes are given in the 'Tests/' directory.

## Installation

- Clone git repository and pip install local copy

```
git clone https://github.com/aidancrilly/RTClosureJAX.git
cd RTClosureJAX
pip install -e .
```
