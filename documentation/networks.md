# Functions to generate large networks of Izhikevich Neurons

## 1. Network Generation

1. `random_net`: gives connectivity matrix of a random network of HVC projection neurons and interneurons based on experimental values. The probabilities `p_ra`, `p_x`, `p_int_ra`, `p_ra_int`, `p_int_x`, `p_x_int`, `p_ra_ra` and the synaptic strengths `int_ra`, `int_x`, `ra_int`, `x_int`, `ra_ra` can also be specified by the user

2. `synfire_chain`: gives a synfire chain connectivity matrix of HVC projection neurons and interneurons. The parameters `groups`, `p_ra`, `p_int_ra`, `p_ra_int`, `starter` (random seed) set the connections and `int_ra`, `ra_int`, `ra_ra` set the synaptic weights
               
3. `branched_chain`: gives a branched chain connectivity matrix of HVC projection neurons and interneurons with two branches. The parameters `groups`, `p_ra`, `p_int_ra`, `p_ra_int`, `frac` (branch parameter), `starter` (random seed) set the connections and `int_ra`, `ra_int`, `ra_ra` set the synaptic weights

## 2. Plotting

1. `plot_neuron`: plots a given neuron in a network along with relevant properties. `df`,`X` have to be passed and `Neu`,`tmin`,`tmax` have to be specified

2. `plot_raster`: plots a raster for the entire network given `df`, `X`, `tmin` and `tmax`

3. `plot_interneuron` : plots the mean and standard deviation of the interneuron firing rate over time given `df`, `X`, `smoothed_activity`, `tmin`, `tmax`
