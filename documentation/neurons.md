# Functions to simulate dynamics of neurons

## 1. Implementing Neurons

1. `lif_eu`: evolves a Leaky-IF neuron given parameters `R`, `E`, `Vth`, `V_r`, `tm`, a `time` vector, resolution `dt` and an input vector `I` using forward-euler method

2. `AP`: defines the shape of an AP given parameters `dur`, `spikeM`, `spikem` and `Vth`

3. `spi_lif_eu`: evolves a spiking Leaky-IF neuron; needs a functions `AP` to be defined and parameters `ref_per`, `R`, `E`, `Vth`, `V_r`, `tm`, `apDur`, `spM`, `spm`.

4. `PSC`: defines the shape of a PSC given parameters `t_rise`, `t_fall`, `mag` and resolution `dt`

5. `VI_lif_eu`: evolves a spiking Leaky-IF neuron given a stimulus vector `I`, a time vector `time`, parameters `R`, `E`, `Vth`, `V_r`, `tm`, `ref_per`, `apDur`, `spM`, `spm`, `t_rise_PSC`, `t_fall_PSC`, `mag` (of PSC), `w` (weight) and resolution `dt` and generates an output post-synaptic current vector `Io`.

6. `my_lif_eu`: evolves a network of spiking Leaky-IF neurons given a stimulus matrix `Iec_df`, a time vector `time`, parameters `R`, `E`, `Vth`, `V_r`, `tm`, `ref_per`, `apDur`, `spM`, `spm`, `t_rise_PSC`, `t_fall_PSC`, `mag` (of PSC), connectivity matrix `C` and resolution `dt`.

## 2. Generating Stimuli

Every stimulus is generated as a vector stored as a column in the main dataframe `df`. 

1. `stimuli`: generates a stimulus based on some parameters. Takes `df` and `mag` as required parameters and `stim`, `dur`, `st`, `pwidth`, `gap`, `base`, `noise`, `ramp`, `rampup_t`, `rampdown_t`, `psp_dur`, `freq`, `synaptize` and `noisy` as optional parameters. The stimulus type (`stim`) can be - 'step', 'pulse', 'lin', 'sin' or 'bump'. If `synaptize` is set to `True` then the stimulus is composed of short pulses of duration `psp_dur` and frequency `freq` with an envelope as per the stimulus type.

2. `inject` : injects a generated stimulus into a network of neurons given an index array `neurons` and a vector of neuron objects `X`; see `stimuli` for other parameters. Outputs a stimulus matrix `Iec_df`. `overlay` set to true writes stimuli on top of the previous `Iec_df`.

3. `release` : modulates a network of neurons given an index array `neurons` and a vector of neuron objects `X`; see `stimuli` for other parameters. Outputs a stimulus matrix `Mod_df`. `overlay` set to true writes stimuli on top of the previous `Mod_df`.

4. `inject_single` : injects a generated stimulus into a single neuron given an object `neuron`. Can also plot the stimulus.

5. `release_single` : modulates a single neuron in real time given an object `neuron`. Can also plot the modulating input.
