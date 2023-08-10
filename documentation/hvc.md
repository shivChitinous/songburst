# HVC Neuron Types and Parameters

## Izhikevich HVC Neurons

1. `parametershvc`: returns parameters for a bursting neuron adapted from _Giordiani et al. (2018)_

2. `parametershvcra`: returns optimised parameters for an $HVC_{RA}$ neuron given a heterogeneity parameter `h`

3. `parametershvcx`: returns optimised parameters for an $HVC_{X}$ neuron given a heterogeneity parameter `h`

4. `parametershvcint`: returns optimised parameters for an $HVC_{int}$ neuron given a heterogeneity parameter `h`

5. `neuron_gen`: generates a network of IZ neuron objects `X` given a connectivity matrix `conn`, an optional neuron-type vector `typ` and 3 optional heterogeneity parameters `hra`, `hx`, `hint` for each cell type.

6. `parametrize`: parametrizes an HVC neuron in terms of its mean-firing rate `mean_fr` and its resting membrane potential `rmp` given the neuron object once it has been evolved to step stimulus (*standard = 200 pA*), a time period `t_r` during which the neuron is at resting membrane potential and the main dataframe `df`

## Conductance HVC Neurons

Adapted from _Ross et al. (2019)_

1. `params_int_HH`: returns optimised parameters for an $HVC_{int}$ conductance neuron given a temperature `T`

2. `params_x_HH`: returns optimised parameters for an $HVC_{X}$ conductance neuron given a temperature `T`

3. `params_ra_HH`: returns optimised parameters for an $HVC_{RA}$ conductance neuron given a temperature `T`
