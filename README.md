# songburst
Package to construct models of songbird nucleus HVC

## files

1. `neurons.py` - basic package containing functions to generate post-synaptic currents, action potentials, implement LIF neurons etc.
2. `izneurons.py` - package describing Izhikevich neurons and containing functions to evolve a single or a network of IZ neurons in time
3. `hvc.py` - functions for implementing HVC neurons
4. `conductance.py` - functions for analysing conductance model simulations
5. `networks.py` - functions for simulating large networks of Izhikevich neurons
6. `poly.py` - functions for constructing and evolving polychronous chains of neurons

### Installing virtual environment
1. Clone repo, navigate into folder
2. If you don't already have poetry, install poetry. You may need to close command window and open a new one.
3. Create conda environment:
`conda env create -f environment.yml`
4. Activate environment:
`conda activate songburst`
5. Make sure you are in the top folder of the cloned repo, then install dependencies:
`poetry install`
6. Setup the new environment as an ipython kernel:
`conda install -c anaconda ipykernel`
then
`python -m ipykernel install --user --name=songburst`
