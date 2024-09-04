import argparse
import h5py
import numpy as np
from pathlib import Path
from qkeras import quantized_bits
import yaml

def load_yaml(filename):
    with open(filename, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    return config

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--folder", type=str, required=True)
parser.add_argument("--quantize", type=bool, default=True)
parser.add_argument("--out_file", type=str, default='weights.h')
args = parser.parse_args()


folder = Path(args.folder)

filename = folder / 'best_acc.weights.h5'
config = load_yaml(folder / 'config.yml')['model']

if args.quantize:
    print("Using quantizer")
    q_func = quantized_bits(bits=config['n_bits'], integer=config['n_int'])
    quantizer = lambda x : np.array([q_func(x)])[0]
else:
    print("Not using quantizer")
    quantizer = lambda x : x

weights = h5py.File(filename)

all_weights = {}
for k, v in weights.items():
    if 'layer' not in k:
        continue

    if 'vars' not in v:
        continue

    v = v['vars']
    if len(v) == 0:
        continue

    all_weights[k] = []

    for _, val in v.items():
        all_weights[k].append(np.array(val))
    all_weights[k] = all_weights[k]


np.set_printoptions(precision=15, floatmode='fixed')

for k, v in all_weights.items():
    if '2v0' in k:
        const2v0 = v[0]
        continue
    if '2v2' in k:
        const2v2 = v[0]
        continue
    if 'q_batch' in k:
        #gamma, beta, mean, var

        if 'normalization_1' in k:
            # This is the second qbatch layer, bnorm 2v0
            gamma, beta, mean, var = [quantizer(elem) for elem in v]
            batch2= np.array((mean, gamma/np.sqrt(var), beta)).T
        else:
            # This is the first qbatch layer, bnorm 2v2
            gamma, beta, mean, var = [quantizer(elem) for elem in v]
            batch1= np.array([mean, gamma/np.sqrt(var), beta]).T[0]
        continue

    if 'q_dense_1' in k:
        #This is 2v0 last layer
        w2v0, b2v0 = [quantizer(elem) for elem in v]
        continue
    if 'q_dense' in k:
        w2v2, b2v2 = [quantizer(elem) for elem in v]

        bdiag2v2 = w2v2[-1]
        w2v2 = w2v2[:-1]
        print("WARNING: these are not the weights if not using diag_bias in lineq2v2. Make sure it corresponds")
        continue

    raise ValueError(f"Cannot interpret weight {k}")

#write file
f = open(args.out_file,"w")

f.write('#include "../nPELICAN.h"\n')
# f.write('//model: ' + m['args'].prefix + '\n')
# f.write('//nobj: ' + str(m['args'].nobj) + '\n\n')


f.write('\n\n// WARNING: Have been trained with the following assumptions: \n')

if 'inp' in config:
    conf = config['inp']
    f.write('// input_t == ap_fixed<{},{}> \n'.format(
        conf['n_bits'], conf['n_int']+1
    ))
else:
    # Full precision
    f.write('// input_t == ap_fixed<32, 9> \n')


if 'internal' in config:
    conf = config['internal']
    f.write('// internal_t == ap_fixed<{},{}> \n'.format(
        conf['n_bits'], conf['n_int']+1
    ))


f.write('// weight_t=bias_t == ap_fixed<{},{}> \n\n'.format(
    config['n_bits'], config['n_int']+1
))



#calculate normalization constants
f.write('//normalization constants\n')
f.write('//these are currently NOT quantized, so should be FP\n')
# f.write('//nobj avg = {}\n'.format(m['args'].nobj_avg))

f.write('internal_t const2v0 = {};\n'.format(1/const2v0))
f.write('internal_t const2v02 = {};\n\n'.format((1/const2v0)**2))

f.write('internal_t const2v2 = {};\n'.format(1/const2v2))
f.write('internal_t const2v22 = {};\n\n'.format((1/const2v2)**2))



f.write('//first batchnorm [mean, weight/sqrt(var), bias]\n')
f.write('weight_t batch1_2to2[3] = ' + np.array2string(batch1, separator=', ').replace('\n', '').replace('[', '{').replace(']', '}') + ';\n\n')


f.write('//2to2 linear layer\n')
f.write('weight_t w1_2to2[NHIDDEN*6] = ' + np.array2string(np.ravel(w2v2), separator=', ').replace('\n', '').replace('[', '{').replace(']', '}') + ';\n')
f.write('bias_t b1_2to2[NHIDDEN] = ' + np.array2string(b2v2, separator=', ').replace('\n', '').replace('[', '{').replace(']', '}') + ';\n')
f.write('bias_t b1_diag_2to2[NHIDDEN] = ' + np.array2string(bdiag2v2, separator=', ').replace('\n', '').replace('[', '{').replace(']', '}') + ';\n\n')


f.write('//second batchnorm [channel][mean, weight/sqrt(var), bias]\n')
f.write('weight_t batch2_2to0[NHIDDEN][3] = ' + np.array2string(batch2, separator=', ').replace('\n', '').replace('[', '{').replace(']', '}') + ';\n\n')


f.write('//2to1 linear layer\n')
f.write('weight_t w2_2to0[NHIDDEN*2*NOUT] = ' + np.array2string(np.ravel(w2v0), separator=', ').replace('\n', '').replace('[', '{').replace(']', '}') + ';\n')
f.write('bias_t b2_2to0[NOUT] = ' + np.array2string(b2v0, separator=', ').replace('\n', '').replace('[', '{').replace(']', '}') + ';\n')


f.close()