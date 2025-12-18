# Profiling Guide: Using Apple Tools for CPU and Memory Profiling

This guide shows how to profile the model loading code using Apple's native tools (Instruments) and Python profilers to see call stacks and time spent in each function.

## Overview

We provide multiple profiling methods:

1. **Apple Instruments** (Recommended for macOS) - Native Apple profiler with GUI
2. **py-spy** - Python call stack profiler with flamegraphs
3. **cProfile** - Python's built-in profiler
4. **memory_profiler** - Memory usage profiling

## Prerequisites

### Install Required Tools

```bash
# For Instruments: Install full Xcode.app from App Store (not just Command Line Tools)
# Instruments is part of Xcode.app, not Command Line Tools
# - Open App Store
# - Search for "Xcode"
# - Install (requires ~15GB)

# Install Python profiling tools (work without Xcode)
uv pip install py-spy memory-profiler snakeviz
```

**Note**: Instruments requires the full Xcode.app installation. Command Line Tools alone are not sufficient. If you don't have Xcode installed, use `py-spy` instead (recommended and works without Xcode).

## Method 1: Apple Instruments (Recommended)

Instruments is Apple's native profiling tool that provides:
- **Call stack visualization** with time spent in each function
- **Memory allocations** and leaks
- **CPU usage** per function
- **Thread analysis**

### Quick Start

```bash
# Use the helper script
uv run python profile_with_instruments.py \
    --method instruments \
    --script profile_model_loading.py

# Or manually
instruments -t "Time Profiler" -D trace.trace \
    uv run python profile_model_loading.py --model Qwen/Qwen2.5-7B-Instruct
```

### Step-by-Step with Instruments GUI

1. **Open Instruments**:
   ```bash
   open -a Instruments
   ```

2. **Select Template**:
   - Choose "Time Profiler" for CPU profiling
   - Choose "Allocations" for memory profiling
   - Choose "Leaks" for memory leak detection

3. **Configure Target**:
   - Click the target dropdown
   - Select "Choose Target..." → "Attach to Process"
   - Or select "Choose Target..." → "Choose Executable" and select Python

4. **Start Profiling**:
   - Click the Record button (red circle)
   - Run your script in terminal:
     ```bash
     uv run python profile_model_loading.py --model Qwen/Qwen2.5-7B-Instruct
     ```
   - Stop recording when done

5. **Analyze Results**:
   - **Call Tree**: Shows call stack with time spent
   - **Invert Call Tree**: Shows which functions call which (bottom-up)
   - **Hide System Libraries**: Focus on your code
   - **Flame Graph**: Visual representation of call stacks

### Reading Instruments Output

- **Self Time**: Time spent in the function itself (not including callees)
- **Total Time**: Time including all called functions
- **Call Count**: Number of times function was called
- **Symbol Name**: Function name and file location

### Example: Finding Slow Functions

1. Open Instruments with Time Profiler
2. Run the model loading script
3. In the Call Tree view:
   - Sort by "Total Time"
   - Expand `AutoModelForCausalLM.from_pretrained`
   - Look for functions taking >1 second
   - Check if they're in your code or library code

## Method 2: py-spy (Flamegraph)

py-spy provides low-overhead profiling with beautiful flamegraphs.

### Quick Start

```bash
# Use the helper script
uv run python profile_with_instruments.py \
    --method pyspy \
    --script profile_model_loading.py \
    --duration 120

# Or manually
py-spy record -o flamegraph.svg --duration 120 --subprocesses \
    -- uv run python profile_model_loading.py --model Qwen/Qwen2.5-7B-Instruct
```

### Viewing Results

```bash
# Open flamegraph in browser
open flamegraph.svg
```

### Understanding Flamegraphs

- **Width**: Time spent (wider = more time)
- **Height**: Call stack depth
- **Colors**: Different functions/modules
- **Click**: Zoom into specific functions

### Example Analysis

1. Look for wide bars at the bottom (top-level functions taking time)
2. Follow the stack upward to see what's calling what
3. Identify bottlenecks (very wide bars)

## Method 3: cProfile

Python's built-in profiler with detailed statistics.

### Quick Start

```bash
# Use the helper script
uv run python profile_with_instruments.py \
    --method cprofile \
    --script profile_model_loading.py

# Or manually
python -m cProfile -o profile.prof profile_model_loading.py
```

### Viewing Results

```bash
# Install snakeviz for visualization
uv pip install snakeviz

# View interactive visualization
snakeviz profile.prof
```

### Command Line Stats

```bash
# View top functions
python -m pstats profile.prof
# Then in pstats prompt:
# sort cumulative
# stats 20
```

### Understanding cProfile Output

```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   1       0.000    0.000    120.5    120.5  transformers/modeling_utils.py:1234(load_pretrained_model)
   10      0.001    0.000    45.2     4.52   transformers/modeling_utils.py:567(load_state_dict)
```

- **ncalls**: Number of calls
- **tottime**: Total time in function (excluding subcalls)
- **percall**: Average time per call
- **cumtime**: Cumulative time (including subcalls)
- **filename:lineno(function)**: Location and function name

## Method 4: Memory Profiling

Profile memory usage line-by-line.

### Quick Start

```bash
# Use the helper script
uv run python profile_with_instruments.py \
    --method memory \
    --script profile_model_loading.py

# Or manually
python -m memory_profiler profile_model_loading.py
```

### Understanding Output

```
Line #    Mem usage    Increment  Occurrences   Line Contents
============================================================
    45    125.0 MiB    125.0 MiB           1   model = AutoModelForCausalLM.from_pretrained(...)
```

- **Mem usage**: Memory at this line
- **Increment**: Memory increase from previous line
- **Occurrences**: How many times this line was executed

## Method 5: All Methods Combined

Run all profiling methods at once:

```bash
uv run python profile_with_instruments.py \
    --method all \
    --script profile_model_loading.py \
    --output-dir ./profiles
```

This will generate:
- `instruments.trace` - Instruments trace file
- `flamegraph.svg` - py-spy flamegraph
- `cprofile.prof` - cProfile data
- `cprofile_stats.txt` - Text stats
- `memory_profile.txt` - Memory profile

## Profiling Specific Functions

### Profile Model Loading Only

```bash
uv run python profile_model_loading.py --model Qwen/Qwen2.5-7B-Instruct
```

This script is optimized for profiling and includes markers for each phase.

### Profile Training Loop

To profile the training loop, modify `train_rfai.py` to add profiling:

```python
import cProfile
import pstats

# In the train method
profiler = cProfile.Profile()
profiler.enable()

# ... training code ...

profiler.disable()
profiler.dump_stats('training_profile.prof')
```

## Tips for Effective Profiling

### 1. Focus on Hot Paths

- Profile the actual operation (model loading, training step)
- Don't profile initialization/setup code
- Use `--duration` to limit profiling time

### 2. Compare Before/After

- Profile before optimization
- Make changes
- Profile after optimization
- Compare results

### 3. Use Multiple Tools

- **Instruments**: Best for overall view and system-level analysis
- **py-spy**: Best for Python call stacks and flamegraphs
- **cProfile**: Best for detailed function-level statistics
- **memory_profiler**: Best for memory usage analysis

### 4. Filter Results

In Instruments:
- Hide System Libraries (uncheck in settings)
- Focus on your code paths
- Use search to find specific functions

In py-spy:
- Use `--include` to filter modules
- Use `--exclude` to exclude modules

### 5. Profile in Production-Like Conditions

- Use similar data sizes
- Use similar hardware
- Profile with actual workloads

## Common Bottlenecks to Look For

### Model Loading

1. **Checkpoint Loading**: `load_state_dict` - often the slowest part
2. **Device Mapping**: `device_map="auto"` - can be slow for large models
3. **Quantization**: 4-bit quantization setup - happens during loading
4. **Memory Allocation**: Large tensor allocations

### Training Loop

1. **Forward Pass**: Model forward computation
2. **Backward Pass**: Gradient computation
3. **Optimizer Step**: Weight updates
4. **Data Loading**: DataLoader iteration
5. **Teacher API Calls**: External API latency

## Example: Profiling Model Loading

```bash
# 1. Profile with Instruments
instruments -t "Time Profiler" -D loading.trace \
    uv run python profile_model_loading.py

# 2. Open in Instruments
open loading.trace

# 3. In Instruments:
#    - Sort by "Total Time"
#    - Expand "AutoModelForCausalLM.from_pretrained"
#    - Look for slow functions
#    - Check if they're in transformers or your code
```

## Troubleshooting

### Instruments Not Found

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Verify installation
xcode-select -p
```

### py-spy Permission Error

```bash
# On macOS, you may need to allow py-spy
# It will prompt you to allow it in System Preferences
```

### cProfile Output Too Large

```bash
# Filter to top functions only
python -m pstats profile.prof
# Then: stats 50  # Top 50 functions
```

### Memory Profiler Not Working

```bash
# Make sure you're using @profile decorator or -m memory_profiler
# Check that psutil is installed
uv pip install psutil
```

## Additional Resources

- [Apple Instruments User Guide](https://developer.apple.com/documentation/instruments)
- [py-spy Documentation](https://github.com/benfred/py-spy)
- [Python Profiling Guide](https://docs.python.org/3/library/profile.html)
- [Flamegraph Explanation](http://www.brendangregg.com/flamegraphs.html)

## Quick Reference

```bash
# Instruments (Time Profiler)
instruments -t "Time Profiler" -D trace.trace python script.py

# py-spy (Flamegraph)
py-spy record -o flamegraph.svg -- python script.py

# cProfile
python -m cProfile -o profile.prof script.py
snakeviz profile.prof

# Memory Profiler
python -m memory_profiler script.py

# All at once
python profile_with_instruments.py --method all --script script.py
```

