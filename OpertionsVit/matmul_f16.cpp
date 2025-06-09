#include "matmul_f16.h"
#include "hls_math.h"

static half execute_add(const half operand1, const half operand2){
#pragma HLS INLINE off
    return operand1+operand2;
}
static half execute_mult(const half operand1, const half operand2){
#pragma HLS INLINE off
    return operand1*operand2;
}

static float dequantise(const half input_op) {
#pragma HLS INLINE off
    return (float)input_op;
}

/*static half load_data_a(half *src0, int stepRowsrc0, int p){
  half a_val= src0[stepRowsrc0+p];
  return a_val;
}

static half load_data_b(half *src1, int stepRowsrc1, int p,int rowbatch){
  half b_val= src1[stepRowsrc1+p+rowbatch];
  return b_val;
}*/

static void load_data_a_stream(half *src0, int stepRowsrc0, int ne00, hls::stream<half> &a_stream) {
#pragma HLS INLINE off
load_a_loop:
  for (int p = 0; p < ne00; ++p) {
#pragma HLS PIPELINE II=1
    a_stream.write(src0[stepRowsrc0 + p]);
  }
}

static void load_data_b_stream(half *src1, int stepRowsrc1, int rowbatch, int ne00, hls::stream<half> &b_stream) {
#pragma HLS INLINE off
load_b_loop:
  for (int p = 0; p < ne00; ++p) {
#pragma HLS PIPELINE II=1
    b_stream.write(src1[stepRowsrc1 + p + rowbatch]);
  }
}

/*static half compute_streams(hls::stream<half> &a_stream, hls::stream<half> &b_stream, int ne00) {
#pragma HLS INLINE off
  half acc = 0;
compute_loop:
  for (int p = 0; p < ne00; ++p) {
#pragma HLS PIPELINE
    half a_val = a_stream.read();
    half b_val = b_stream.read();
    acc = execute_add(acc, execute_mult(a_val, b_val));
  }
  return acc;
}*/

void produce_products(hls::stream<half> &a_stream, hls::stream<half> &b_stream, hls::stream<half> &out,int ne00) {
#pragma HLS INLINE off
    for (int p = 0; p < ne00; ++p) {
#pragma HLS PIPELINE 
        out.write(a_stream.read() * b_stream.read());
    }
}

void consume_accumulate(hls::stream<half> &in, int ne00, half &result) {
#pragma HLS INLINE off
    result = 0;
    for (int p = 0; p < ne00; ++p) {
#pragma HLS PIPELINE
    	auto val = in.read();
        result = result + val;
    }
}

static void execute_gemm_element(half *src0, half *src1,half &result_out, int stepRowsrc0, int stepRowsrc1, int rowbatch, int ne00){
#pragma HLS INLINE off
  half newdata=0;

#pragma HLS DATAFLOW

  hls::stream<half> a_stream("a_stream");
  hls::stream<half> b_stream("b_stream");
  hls::stream<half> c_stream("c_stream");

#pragma HLS STREAM variable=a_stream 
#pragma HLS STREAM variable=b_stream
#pragma HLS STREAM variable=c_stream
  

  load_data_a_stream(src0, stepRowsrc0, ne00, a_stream);
  load_data_b_stream(src1, stepRowsrc1, rowbatch, ne00, b_stream);
  produce_products(a_stream, b_stream, c_stream,ne00);
  consume_accumulate(c_stream,ne00,result_out);
  //result_out= compute_streams(a_stream, b_stream, ne00);
}

static void matmul_gemm(half *src0, half *src1, float *dst, const int ne00,const int ne01,
                        const int ne11,const int ne02,const int ne03, const int nedst){

#pragma HLS INLINE off

    //int i = 0;
    
    int stepRowsrc0=0;
    int rowbatch=0;
gemm_batch:
    for (int batch = 0; batch < ne03; batch++) {
gemm_dim:
      int index1= batch * ne02 * ne01 * ne11;
      for (int dim = 0; dim < ne02; dim++) {
gemm_rowsrc0:
        int index2= dim * ne01 * ne11;
        for (int rowsrc0 = 0; rowsrc0 < ne01; rowsrc0++) { //rowsrc0
gemm_rowsrc1:
          int stepRowsrc1=0;
          int index3=rowsrc0 * ne11;
gemm_process_row:
      for (int rowsrc1 = 0; rowsrc1 < ne11; rowsrc1++)  { //rowsrc1
#pragma HLS PIPELINE

gemm_elem:
            half newdata=0;
            execute_gemm_element(src0, src1,newdata, stepRowsrc0, stepRowsrc1, rowbatch, ne00);
              
            int index = index1 + index2 + index3 + rowsrc1;
            //float newdatafloat = (float)newdata;
            float result_f32 = dequantise(newdata);
            dst[index] = result_f32;
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
void matmul_f16(half *src0, half *src1, float *dst, int ne00,int ne01,int ne02,int ne03,
            int ne11) {

#pragma HLS INTERFACE m_axi offset = slave port = src0 bundle = gmem0 depth=32
#pragma HLS INTERFACE m_axi offset = slave port = src1 bundle = gmem1 depth=32
#pragma HLS INTERFACE m_axi offset = slave port = dst bundle = gmem2 depth=32
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
    const int ne0=  ne00 * ne01 * ne02 * ne03;
    const int ne1= ne00*ne11*ne02*ne03;

 
  matmul_gemm(src0, src1, dst, ne00 ,ne01, ne11,ne02, ne03,ne_dst);
}
}
