/*
 * FFT-256 Butterfly -- Sample SystemC/HLS design for AIV-DSE analysis.
 *
 * Radix-2 decimation-in-time FFT with 8 butterfly stages.
 * Fixed-point arithmetic using sc_fixed<16,8>.
 *
 * Optimization opportunities (for code_advisor to find):
 *   - Twiddle arrays lack ARRAY_PARTITION pragma
 *   - Pipeline II=4 on inner loop (could be II=1 with partitioning)
 *   - Bit-reverse inner loop (8 iters) could be fully unrolled
 *   - Output copy loop has no pipeline pragma
 *   - Working buffers lack partitioning
 */

#include "systemc.h"

// Twiddle factors (precomputed, read-only at runtime)
static sc_fixed<16,8> twiddle_re[128];
static sc_fixed<16,8> twiddle_im[128];

// Butterfly operation -- called from inner loop
void butterfly_op(
    sc_fixed<16,8>& a_re, sc_fixed<16,8>& a_im,
    sc_fixed<16,8>& b_re, sc_fixed<16,8>& b_im,
    sc_fixed<16,8> w_re, sc_fixed<16,8> w_im
) {
    sc_fixed<16,8> t_re = b_re * w_re - b_im * w_im;
    sc_fixed<16,8> t_im = b_re * w_im + b_im * w_re;
    b_re = a_re - t_re;
    b_im = a_im - t_im;
    a_re = a_re + t_re;
    a_im = a_im + t_im;
}

// Top-level HLS function
void fft256(
    sc_fixed<16,8> in_re[256],
    sc_fixed<16,8> in_im[256],
    sc_fixed<16,8> out_re[256],
    sc_fixed<16,8> out_im[256]
) {
    // Working buffers
    sc_fixed<16,8> buf_re[256];
    sc_fixed<16,8> buf_im[256];

    // --- Stage 1: Bit-reverse copy ---
    for (int i = 0; i < 256; i++) {
        int rev = 0;
        int tmp = i;
        for (int j = 0; j < 8; j++) {
            rev = (rev << 1) | (tmp & 1);
            tmp >>= 1;
        }
        buf_re[rev] = in_re[i];
        buf_im[rev] = in_im[i];
    }

    // --- Stage 2: 8 butterfly stages ---
    for (int stage = 0; stage < 8; stage++) {
        int half = 1 << stage;
        int span = half << 1;

        for (int k = 0; k < 256; k += span) {
            #pragma HLS PIPELINE II=4
            for (int j = 0; j < half; j++) {
                int idx = k + j;
                int tw_idx = j * (128 >> stage);

                butterfly_op(
                    buf_re[idx], buf_im[idx],
                    buf_re[idx + half], buf_im[idx + half],
                    twiddle_re[tw_idx], twiddle_im[tw_idx]
                );
            }
        }
    }

    // --- Stage 3: Copy output ---
    for (int i = 0; i < 256; i++) {
        out_re[i] = buf_re[i];
        out_im[i] = buf_im[i];
    }
}
