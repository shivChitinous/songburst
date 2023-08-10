# Izhikevich Neurons

A type of quadratic IF neurons given by the non-dimensionalised equations:


$$\dfrac{dv}{dt} = k_1v^2 + k_2v + k_3 - u + I$$

$$\dfrac{du}{dt} = a(bv-u)
$$


if 
$v>v_{th}$


then $v \rightarrow c, u \rightarrow u + d$. 



$I$ is capacitance normalised current

$v$ is the membrane potential of the neuron

$u$ is the recovery variable

$v_{th}$ signifies the peak value of an action potential and not the traditional _voltage of no return_. Choosing the threshold just implies that we specify the maximum voltage in an action potential.

1. `IZN`: class that defines IZ neuron objects

2. `modulate`: changes parameters of IZ neurons to make them more excitable; takes parameter $m \in [0,1]$.
If `reset` is set to `True`, it resets neuron to original parameters before modulating (original parameters need to be passed as `params`)

3. `modulate_net`: changes parameters of $N$ IZ neurons in a network to make them more excitable; takes parameter $m \in [0,1]$. It takes in each parameter as well as $m$ as an $N$-D vector.

4. `evolve`: evolves a single IZ neuron object in time given the time vector contained in `df` and a `psc` defined by the `PSC` function from the `nrn` package.

5. `evolve_net`: evolves a network of IZN objects `X` in time given the time vector contained in `df`, a `psc` defined by the `PSC` function from the `nrn` package, a stimulus matrix `Iec_df` and a connectivity matrix `conn`. It also accounts for axonal conduction delays given a delay matrix `delays`.
