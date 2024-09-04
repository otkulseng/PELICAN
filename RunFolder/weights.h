#include "../nPELICAN.h"


// WARNING: Have been trained with the following assumptions: 
// input_t == ap_fixed<32, 9> 
// weight_t=bias_t == ap_fixed<8,3> 

//normalization constants
//these are currently NOT quantized, so should be FP
internal_t const2v0 = 0.12868403450003923;
internal_t const2v02 = 0.016559580735207288;

internal_t const2v2 = 0.11692787773386222;
internal_t const2v22 = 0.013672128591345032;

//first batchnorm [mean, weight/sqrt(var), bias]
weight_t batch1_2to2[3] = { 3.968750000000000,  0.156863957643509, -0.187500000000000};

//2to2 linear layer
weight_t w1_2to2[NHIDDEN*6] = {-0.406250000000000,  0.343750000000000,  0.093750000000000, -0.281250000000000,  0.312500000000000, -0.093750000000000,  0.125000000000000, -0.218750000000000,  0.281250000000000, -0.531250000000000, -0.156250000000000, -1.156250000000000};
bias_t b1_2to2[NHIDDEN] = {-0.468750000000000, -0.125000000000000};
bias_t b1_diag_2to2[NHIDDEN] = {-0.312500000000000,  0.062500000000000};

//second batchnorm [channel][mean, weight/sqrt(var), bias]
weight_t batch2_2to0[NHIDDEN][3] = {{1.968750000000000, 0.756367385387421, 0.250000000000000}, {0.125000000000000, 4.796917438507080, 0.093750000000000}};

//2to1 linear layer
weight_t w2_2to0[NHIDDEN*2*NOUT] = { 0.906250000000000,  1.125000000000000,  0.312500000000000, -1.343750000000000};
bias_t b2_2to0[NOUT] = {0.281250000000000};
