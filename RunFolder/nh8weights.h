#include "../nPELICAN.h"


// WARNING: Have been trained with the following assumptions: 
// input_t == ap_fixed<24,12> 
// internal_t == ap_fixed<10,7> 
// weight_t=bias_t == ap_fixed<8,3> 

//normalization constants
//these are currently NOT quantized, so should be FP
internal_t const2v0 = 0.17669570293210196;
internal_t const2v02 = 0.031221371434669624;

internal_t const2v2 = 0.09544623821617948;
internal_t const2v22 = 0.00910998438961968;

//first batchnorm [mean, weight/sqrt(var), bias]
weight_t batch1_2to2[3] = { 3.968750000000000,  0.515625000000000, -0.187500000000000};

//2to2 linear layer
weight_t w1_2to2[NHIDDEN*6] = {-0.843750000000000,  0.156250000000000,  0.187500000000000, -0.937500000000000, -0.281250000000000,  0.843750000000000, -0.531250000000000, -0.562500000000000,  0.437500000000000,  0.500000000000000,  0.000000000000000, -0.437500000000000, -0.156250000000000,  0.343750000000000,  0.250000000000000, -0.531250000000000, -1.000000000000000, -0.593750000000000, -0.218750000000000,  0.531250000000000,  0.531250000000000, -0.718750000000000, -0.062500000000000, -0.343750000000000,  0.156250000000000, -0.187500000000000,  0.218750000000000,  0.062500000000000,  0.781250000000000,  0.062500000000000,  0.812500000000000, -0.281250000000000,  0.093750000000000,  0.656250000000000,  0.000000000000000,  0.312500000000000, -1.093750000000000, -0.187500000000000, -0.781250000000000,  0.093750000000000, -0.250000000000000, -0.562500000000000, -0.343750000000000, -0.468750000000000, -0.875000000000000,  0.343750000000000, -0.656250000000000,  0.187500000000000};
bias_t b1_2to2[NHIDDEN] = {-0.312500000000000, -0.375000000000000, -0.687500000000000, -0.468750000000000,  1.156250000000000, -0.843750000000000,  1.375000000000000, -0.625000000000000};
bias_t b1_diag_2to2[NHIDDEN] = { 0.812500000000000, -1.406250000000000,  0.156250000000000, -0.062500000000000,  0.312500000000000,  0.031250000000000,  0.531250000000000, -0.250000000000000};

//second batchnorm [channel][mean, weight/sqrt(var), bias]
weight_t batch2_2to0[NHIDDEN][3] = {{ 0.437500000000000,  1.500612616539001,  0.156250000000000}, { 0.687500000000000,  1.169369101524353,  0.125000000000000}, { 0.281250000000000,  2.165514469146729,  0.093750000000000}, { 0.593750000000000,  1.264911055564880,  0.062500000000000}, { 1.937500000000000,  0.857804477214813,  0.093750000000000}, { 0.843750000000000,  0.547497332096100, -0.281250000000000}, { 1.968750000000000,  0.762492895126343,  0.156250000000000}, { 0.156250000000000,  1.530930995941162, -0.062500000000000}};

//2to1 linear layer
weight_t w2_2to0[NHIDDEN*2*NOUT] = {-0.156250000000000,  0.093750000000000, -0.468750000000000, -0.625000000000000, -0.250000000000000,  0.031250000000000, -0.156250000000000,  0.187500000000000,  0.343750000000000,  0.968750000000000, -1.156250000000000,  0.187500000000000,  0.781250000000000,  0.531250000000000, -0.062500000000000, -0.250000000000000,  0.250000000000000,  0.562500000000000,  0.437500000000000, -0.718750000000000, -0.343750000000000,  0.687500000000000,  0.343750000000000,  0.687500000000000, -0.468750000000000,  0.343750000000000,  0.000000000000000,  0.093750000000000,  0.031250000000000,  0.750000000000000,  0.125000000000000, -0.593750000000000,  0.375000000000000,  0.187500000000000, -0.343750000000000,  0.000000000000000,  0.187500000000000, -0.312500000000000, -0.156250000000000, -0.281250000000000,  0.031250000000000,  0.343750000000000,  0.281250000000000,  0.593750000000000, -0.500000000000000, -0.281250000000000, -0.406250000000000,  0.531250000000000,  0.218750000000000, -0.156250000000000,  0.187500000000000, -0.062500000000000, -0.875000000000000, -0.437500000000000,  0.281250000000000,  0.375000000000000,  0.312500000000000,  0.062500000000000,  0.093750000000000,  0.250000000000000,  0.125000000000000, -0.218750000000000,  0.593750000000000,  0.468750000000000,  0.281250000000000,  0.125000000000000,  0.187500000000000, -0.156250000000000, -0.156250000000000,  0.468750000000000,  0.062500000000000,  0.312500000000000, -0.218750000000000, -0.468750000000000,  0.312500000000000, -0.062500000000000, -0.437500000000000,  0.312500000000000,  0.031250000000000,  0.375000000000000};
bias_t b2_2to0[NOUT] = { 0.031250000000000,  0.437500000000000,  0.343750000000000, -0.375000000000000, -0.375000000000000};
