#ifndef REAL
#define REAL float
#endif


__kernel void act_tanh (__global REAL* x,
                     uint offset_x, const uint stride_x) {
    const uint ix = offset_x + get_global_id(0) * stride_x;
    x[ix] = tanh (x[ix]);
}


__kernel void deact_tanh (__global REAL* x,
                    const uint offset_x, const uint stride_x) {
    const uint ix = offset_x + get_global_id(0) * stride_x;
    x[ix] = 1.0 - (x[ix] * x[ix]);
}


__kernel void mul (__global REAL* x,
                    const uint offset_x, const uint stride_x,
                    __global const REAL* y,
                    const uint offset_y, const uint stride_y) {
    const uint ix = offset_x + get_global_id(0) * stride_x;
    const uint iy = offset_y + get_global_id(0) * stride_y;
    x[ix] = x[ix] * y[iy];
}

