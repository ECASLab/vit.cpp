/*
 * Copyright 2022-2024
 * Author: Luis G. Leon-Vega <luis.leon@ieee.org>
 */

#ifndef __MATMUL_H__
#define __MATMUL_H__

#include "../common/config.h"


extern "C" {

void matmul_f32(float *src0, float *src1, float *dst, int ne00,int ne01,int ne02,int ne03,
            int ne11);
}

#endif // __MATMUL_H__
