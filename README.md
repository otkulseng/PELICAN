# Nano Pelican
Keras implementation based on the nano-pelican code in
 https://github.com/abogatskiy/PELICAN-nano.
The reason for rewriting is the possibility of using qkeras for QAT.


## Installing
It is set up like a python package, so can be pip installed (run command inside folder containing setup.py)
```
$ C:\users\PELICAN\ pip install .
```
## Usage

The folder [scripts](nanopelican/scripts) has code that probably wont be useful for anyone else using this package. The rest of the codebase, however, has been implemented with modularity in mind.

In each subfolder of [Experiments](experiments), you will find a file called `model`. These (many different models) may be useful as examples.

All the pelican-related code is implemented as layers. There is quite a few of them

### Inner Product
The beginning of any Pelican model. The behaviour of this layer can be adjusted. See [here](nanopelican/layers/inner_product.py) for a complete list of all parameters

**Input:**
(Samples x particles x 4 (3)), where the last dimension is the fourvector of the particle in (E, px, py, pz) or (pt, eta, phi).

**Output:**
(Samples x particles x particles), matrix with all the inner products.
Example command:

### Lineq2v2
The behaviour of this layer can be adjusted. See [here](nanopelican/layers/inner_product.py) for a complete list of all parameters

**Input:**
(N x N x L), where each N by N matrix is assumed to be permutation equivariant.

**Output:**
(N x N x 15L), for each input matrix, output the 15 general linear permutation equivariant matrices that can be made from it.

If the input has some extra symmetry, some of the 15 outputs may be redundant. Set
`hollow=True` if the input matrices have zero trace and `symmetric=True` if the input matrix is symmetric.


Example command:
```
$ python train --config=model.yml
```
See model.yml for example of a config file

