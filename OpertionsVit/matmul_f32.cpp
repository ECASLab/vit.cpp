#include "matmul_f32.h"
#include "hls_math.h"

static void matmul_gemm(float *src0, float *src1, float *dst, const int ne00,const int ne01,
                        const int ne11,const int ne02,const int ne03, const int nedst){

#pragma HLS INLINE off
    
    int stepRowsrc0=0;
    int rowbatch=0;
gemm_batch:
    for (int batch = 0; batch < ne03; batch++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1 avg=1
gemm_dim:
      for (int dim = 0; dim < ne02; dim++) {
#pragma HLS LOOP_TRIPCOUNT min=ne02 max=ne02 avg=ne02
gemm_rowsrc0:
        for (int rowsrc0 = 0; rowsrc0 < ne01; rowsrc0++) { //rowsrc0
#pragma HLS LOOP_TRIPCOUNT min=ne01 max=ne01 avg=ne01
gemm_rowsrc1:
          int stepRowsrc1=0;
          for (int rowsrc1 = 0; rowsrc1 < ne11; rowsrc1++)  { //rowsrc1
#pragma HLS LOOP_TRIPCOUNT min=ne11 max=ne11 avg=ne11
            float newdata= 0.0f;                      
gemm_elem:
              for (int p = 0; p < ne00; ++p) { // cada valor dentro del paquete
#pragma HLS UNROLL
                float a_val= src0[stepRowsrc0+p];
                float b_val= src1[stepRowsrc1+p+rowbatch];
                newdata += a_val * b_val;
              }
              int index = batch * ne02 * ne01 * ne11 + dim * ne01 * ne11 + rowsrc0 * ne11 + rowsrc1;
              dst[index] = newdata;
              //dst[i]=newdata;
              //i++;
              stepRowsrc1+=ne00;
            }
            stepRowsrc0+=ne00;
        }
        rowbatch+=(ne00*ne11);
      }
    }
}

extern "C" {
 
/**
 * matrix: (rows, cols)
 * a: input (samples, inputs)
 * b: weights (outputs, inputs) assumed transposed
 * c: output (samples, outputs)
 */
void matmul_f32(float *src0, float *src1, float *dst, int ne00,int ne01,int ne02,int ne03,
            int ne11) {

#pragma HLS INTERFACE m_axi offset = slave port = src0 bundle = gmem0 depth=1024
#pragma HLS INTERFACE m_axi offset = slave port = src1 bundle = gmem1 depth=1024
#pragma HLS INTERFACE m_axi offset = slave port = dst bundle = gmem2 depth=1024
#pragma HLS INTERFACE s_axilite register port = ne00
#pragma HLS INTERFACE s_axilite register port = ne01
#pragma HLS INTERFACE s_axilite register port = ne02 
#pragma HLS INTERFACE s_axilite register port = ne03
#pragma HLS INTERFACE s_axilite register port = ne11
#pragma HLS INTERFACE s_axilite register port = return

  /*ne00 = dimensión común
  ne01 = filas de src0
  ne11 = columnas de src1
  ne02 = ne12 = batches
  ne03 = ne13 = batches*/

    const int ne_dst = ne01 * ne11 * ne02 * ne03;
    const int ne0= ne00*ne01*ne02*ne03;
    const int ne1= ne00*ne11*ne02*ne03;

 
  matmul_gemm(src0, src1, dst, ne00 ,ne01, ne11,ne02, ne03,ne_dst);
}
}







