# HLS Directive Reference

## Synthesis Parameters (Knobs)

### unroll_factor (1-64)
- Controls loop unrolling parallelism
- Higher unroll = lower latency, higher area + power
- Diminishing returns above 16 for most designs
- Interaction: conflicts with resource_sharing at high values (>8)

### pipeline_depth (1-16)
- Sets initiation interval (II) target for pipelined loops
- pipeline=1 means new input every clock cycle (maximum throughput)
- Higher values reduce area but increase latency
- Requires arrays to be partitioned for II=1 with memory access

### clock_period_ns (1.0-100.0)
- Target clock frequency: lower period = faster clock
- Faster clock increases power superlinearly (~power^1.2)
- Tool may not meet timing if period is too aggressive
- Check slack_ns in timing report after synthesis

### array_partition_factor (1-32)
- Splits arrays into multiple physical memories
- Enables parallel memory access (required for low-II pipelines)
- Types: cyclic (interleaved), block (contiguous), complete (registers)
- Higher factor = more area (each partition is a separate BRAM/register)

## Clock Slack

### clock_slack_ns (-5.0 to 50.0)
- Positive slack relaxes timing closure, negative tightens
- 1-2 ns slack usually improves timing without major area cost
- Above 3 ns: area grows fast, diminishing returns
- Negative slack: tool works harder on timing, may fail to close

## DPO Modes (Datapath Optimization)

| Mode | Area Reduction | Power Reduction | Risk |
|------|---------------|-----------------|------|
| none | 0% | 0% | None |
| DPO_AUTO_EXPR | ~5% | ~5% | Low - expression-level only |
| DPO_AUTO_OPT | ~12% | ~12% | Medium - may affect timing |
| DPO_AUTO_ALL | ~20% | ~20% | Higher - aggressive, verify timing |

- Use DPO_AUTO_EXPR first when timing is marginal
- DPO_AUTO_ALL is best when area/power are primary concerns
- DPO_AUTO_OPT is a good middle ground for most designs

## Boolean Directives

### flatten
- Removes module hierarchy, enables cross-boundary optimization
- Effect: ~5% latency reduction, ~15% area increase
- Use when: inter-module optimization is important
- Avoid when: area is already over budget

### inline
- Inlines function calls (similar to flatten but at function level)
- Effect: ~7% latency reduction, ~12% area increase
- Use when: small helper functions are called frequently
- Combine with flatten for maximum effect (but watch area)

### loop_merge
- Merges adjacent loops with same bounds into one loop
- Effect: ~10% latency reduction, minimal area impact
- Use when: sequential loops iterate over same range
- Requires loops to have no data dependencies between iterations

### bitwidth_reduce
- Automatically narrows data paths to minimum required width
- Effect: ~15% area reduction, ~12% power reduction
- Use when: design uses fixed-point with generous bit widths
- Safe for most designs; rarely causes functional issues

### resource_sharing
- Shares hardware resources (multipliers, adders) across operations
- Effect: ~25% area reduction, ~5% power increase (muxing overhead)
- Use when: area is primary concern and latency budget allows
- Trade-off: may increase latency due to scheduling constraints
- Interaction: conflicts with high unroll_factor (shared resources can't be unrolled)

## Pragma Syntax Reference

### Pipeline
```
#pragma HLS PIPELINE II=<int>
```
- II=1: maximum throughput, highest area
- II=2+: reduces area, one new input every II cycles
- Place before the loop body

### Unroll
```
#pragma HLS UNROLL factor=<int>
```
- factor=0 or omitted: fully unroll
- factor=N: partially unroll by factor N
- Only for loops with compile-time-determinable bounds

### Array Partition
```
#pragma HLS ARRAY_PARTITION variable=<name> type=<cyclic|block|complete> factor=<int> dim=<int>
```
- cyclic: interleaved access (good for strided patterns)
- block: contiguous chunks (good for sequential access)
- complete: fully partition to registers (small arrays only)
- dim=1 for 1D arrays, dim=0 partitions all dimensions

### Interface
```
#pragma HLS INTERFACE mode=<ap_fifo|ap_memory|axis> port=<name>
```
- ap_memory: standard memory interface (default)
- ap_fifo: streaming FIFO interface
- axis: AXI-Stream (for streaming designs)

## Common Parameter Interactions

- **High unroll + resource_sharing**: Conflict. Sharing reduces parallelism.
- **flatten + inline**: Complementary. Both remove hierarchy barriers.
- **DPO + tight clock**: DPO_AUTO_ALL may prevent timing closure.
- **bitwidth_reduce + fixed-point**: Highly effective. Less useful for float.
- **pipeline II=1 + no array_partition**: Cannot achieve II=1 if arrays have read/write conflicts.
