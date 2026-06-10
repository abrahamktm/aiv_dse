# HLS Optimization Strategies

## Area Reduction Priority Order

When area exceeds the constraint threshold, try these in order:

1. **Enable DPO_AUTO_ALL** — biggest single-knob impact (~20% area reduction)
2. **Enable resource_sharing** — another ~25% area reduction, slight power increase
3. **Enable bitwidth_reduce** — ~15% area reduction, safe for fixed-point
4. **Reduce unroll_factor** — direct area reduction but latency will increase
5. **Increase pipeline II** — reduces replicated logic, increases latency
6. **Reduce array_partition_factor** — fewer memory banks, but may hurt throughput

## Latency Reduction Checklist

When latency exceeds the constraint threshold:

1. **Increase unroll_factor** — most direct path to lower latency
2. **Set pipeline II=1** — requires array partitioning for memory-bound loops
3. **Enable flatten + inline** — removes hierarchy barriers for optimization
4. **Enable loop_merge** — merges sequential loops, reduces total cycles
5. **Partition arrays** — eliminates memory port conflicts
6. **Add clock slack (1-2 ns)** — gives timing optimizer more room

## Power Reduction Strategies

1. **Enable DPO** — reduces both dynamic and static power
2. **Enable bitwidth_reduce** — narrower data paths = less switching
3. **Reduce clock frequency** (increase clock_period_ns) — power scales ~clock^1.2
4. **Reduce unroll_factor** — less parallel hardware = less power
5. **Avoid: resource_sharing** — actually increases power slightly due to muxing

## Common Anti-Patterns in SystemC Code

### Dynamic memory allocation
- **Problem**: `new`, `malloc`, `std::vector` are not synthesizable
- **Fix**: Use fixed-size arrays with compile-time bounds

### Variable loop bounds
- **Problem**: Loops with non-constant bounds cannot be unrolled or pipelined efficiently
- **Fix**: Use `#define` or `const int` for loop bounds

### Pointer aliasing
- **Problem**: Compiler cannot prove pointers don't overlap, prevents optimization
- **Fix**: Use separate arrays or add `#pragma HLS DEPENDENCE`

### Deep function call chains
- **Problem**: Each function boundary limits optimization scope
- **Fix**: Enable flatten/inline, or manually inline critical paths

### Large arrays without partitioning
- **Problem**: Single-port memory limits throughput to 1 read + 1 write per cycle
- **Fix**: Partition arrays to match the parallelism of your unroll/pipeline settings

## FFT-Specific Optimization Tips

### Butterfly operations
- Inner butterfly loop should target II=1 with full pipeline
- Twiddle factor arrays must be partitioned (cyclic, factor >= butterfly parallelism)
- Consider partial unroll of stage loop (8 stages = try factor=2 or 4)

### Bit-reverse permutation
- Can be computed with compile-time index arrays for power-of-2 sizes
- Small inner loop (log2(N) iterations) is a good candidate for full unroll
- Memory access is random — partition the working buffer

### Memory architecture
- Dual-port BRAM allows 2 accesses per cycle (1 read + 1 write, or 2 reads)
- For butterfly: need 2 reads + 2 writes per cycle at II=1
- Partition factor >= 2 on working buffers is essential
- Twiddle tables: read-only, cyclic partition works well

## Constraint Interaction Rules

- **Area + Latency conflict**: Reducing one typically increases the other
- **Resolution strategy**: Find the minimum unroll that meets latency, then use DPO + bitwidth_reduce + resource_sharing to bring area under budget
- **Power usually follows area**: Techniques that reduce area tend to reduce power too (except resource_sharing)
- **Clock period is a meta-knob**: Changing clock affects all three metrics simultaneously
