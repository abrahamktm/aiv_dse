/*
 * FIR Filter (32-tap) -- Sample SystemC/HLS design for AIV-DSE analysis.
 *
 * 32-tap symmetric FIR filter with fixed-point arithmetic.
 * Demonstrates a different optimization profile than the FFT sample:
 *   - Tight inner MAC loop (good unroll target)
 *   - Coefficient ROM (good partition target)
 *   - Sequential streaming access pattern (good for II=1 pipelining)
 *
 * Optimization opportunities (for code_advisor to find):
 *   - Coefficient array lacks ARRAY_PARTITION pragma
 *   - Tap delay line lacks partitioning (limits MAC parallelism)
 *   - Inner MAC loop has no pipeline pragma
 *   - Sample shift loop could be unrolled
 *   - No INTERFACE pragma for axis-streaming I/O
 */

#include "systemc.h"

static const int N_TAPS = 32;

// Precomputed FIR coefficients (low-pass, normalized)
static sc_fixed<16,1> coeffs[N_TAPS];

// Top-level HLS function -- streaming sample-by-sample
void fir_filter(
    sc_fixed<16,8> in_sample,
    sc_fixed<16,8>& out_sample
) {
    // Tap delay line (state across calls)
    static sc_fixed<16,8> delay_line[N_TAPS];

    // --- Shift delay line ---
    for (int i = N_TAPS - 1; i > 0; i--) {
        delay_line[i] = delay_line[i - 1];
    }
    delay_line[0] = in_sample;

    // --- MAC accumulation ---
    sc_fixed<32,16> acc = 0;
    for (int t = 0; t < N_TAPS; t++) {
        acc += delay_line[t] * coeffs[t];
    }

    out_sample = (sc_fixed<16,8>)acc;
}
