# My Domain Knowledge

Add your personal optimization heuristics, project-specific notes,
and lessons learned here. This file is chunked and indexed alongside
the HLS documentation for RAG retrieval.

## My Optimization Rules

(Add your personal optimization heuristics here. Examples below.)

<!-- Example:
- For our FFT IPs, always start with unroll=2, pipeline=2
- DPO_AUTO_OPT is safer than DPO_AUTO_ALL for timing-critical paths
- resource_sharing breaks butterfly parallelism — avoid for FFT
-->

## Project-Specific Notes

(Add IP-specific knowledge here. Examples below.)

<!-- Example:
- FFT-256: twiddle arrays need cyclic partition factor=4 minimum
- FIR filter: can handle unroll=16 without area issues
- AES block: bitwidth_reduce is not safe (fixed 128-bit paths)
-->

## Lessons Learned

(Add insights from past synthesis runs here. Examples below.)

<!-- Example:
- DPO_AUTO_ALL caused timing failure on Project X at 200 MHz
- Combining flatten + loop_merge gave 18% latency improvement on FFT
- resource_sharing + unroll > 8 always causes area explosion
-->
