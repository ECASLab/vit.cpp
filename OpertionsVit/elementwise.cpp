#include "elementwise.h"
enum { OP_ADD = 0, OP_MULT = 1 };


static void load_input(RawDataT* in, hls::stream<RawDataT>& inStream, int size) {
    const int size_raw = size / kPackets;
mem_rd:
    for (int i = 0; i < size_raw; ++i) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min = kTotalMaxSize max = kTotalMaxSize avg = kTotalMaxSize
        inStream << in[i];
    } 
}

static void compute(hls::stream<RawDataT>& in1_stream,
                        hls::stream<RawDataT>& in2_stream,
                        hls::stream<RawDataT>& out_stream,
                        int size, int op) {
execute:
    for (int i = 0; i < size; i += kPackets) {
#pragma HLS LOOP_TRIPCOUNT min = kTotalMaxSize max = kTotalMaxSize avg = kTotalMaxSize
#pragma HLS PIPELINE
        RawDataT raw_in1 = in1_stream.read();
        RawDataT raw_in2 = in2_stream.read();
        RawDataT raw_out = 0;
        for (int p = 0; p < kPackets; ++p) {
#pragma HLS UNROLL
            // Offsets 
            const int offlow = p * kDataWidth;
            const int offhigh = offlow + kDataWidth - 1;

            // Data
            //DataT in1, in2, out;
            AccT in1, in2, out;
            in1.i = raw_in1(offhigh, offlow);
            in2.i = raw_in2(offhigh, offlow);

            
            
            // Operation
            switch (op) {
                case OP_ADD:
                    out.f = in1.f + in2.f;
                    //result_f16 = input1_f16 + input2_f16;
                    break;
                case OP_MULT:
                    out.f = in1.f * in2.f;
                    //result_f16 = input1_f16 * input2_f16;
                    break;
                default:
                    break;
            }
            raw_out(offhigh, offlow) = out.i;
        }
        out_stream << raw_out;
    }
}

static void store_result(RawDataT* out, hls::stream<RawDataT>& out_stream, int size) {
    const int size_raw = size / kPackets;
mem_wr:
    for (int i = 0; i < size_raw; ++i) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min = kTotalMaxSize max = kTotalMaxSize avg = kTotalMaxSize
        out[i] = out_stream.read();
    }
}

extern "C" {

    /*
    Vector Addition Kernel

    Arguments:
        in1  (input)  --> Input vector 1
        in2  (input)  --> Input vector 2
        out  (output) --> Output vector
        size (input)  --> Number of elements in vector
        op (input)    --> Operation: 0: add, 1: mult
*/

void elementwise(RawDataT* in1, RawDataT* in2, RawDataT* out, int size, int op) {
#pragma HLS INTERFACE m_axi port = in1 bundle = gmem0
#pragma HLS INTERFACE m_axi port = in2 bundle = gmem1
#pragma HLS INTERFACE m_axi port = out bundle = gmem0

#pragma HLS dataflow

    static hls::stream<RawDataT> in1_stream("input_stream_1");
    static hls::stream<RawDataT> in2_stream("input_stream_2");
    static hls::stream<RawDataT> out_stream("output_stream");

#pragma HLS stream variable = in1_stream depth = 32
#pragma HLS stream variable = in2_stream depth = 32
#pragma HLS stream variable = out_stream depth = 32


    // dataflow pragma instruct compiler to run following three APIs in parallel
    load_input(in1, in1_stream, size);
    load_input(in2, in2_stream, size);
    compute(in1_stream, in2_stream, out_stream, size, op);
    store_result(out, out_stream, size);
}
}
