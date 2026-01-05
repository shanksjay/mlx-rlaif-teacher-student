# Manual Instruments Profiling Guide

This guide shows you how to manually attach Apple Instruments to a running training script. This is useful for profiling long-running processes like `train_rlaif.py` where you want to profile specific phases (e.g., after warmup, during generation, etc.).

## Quick Start

### Step 1: Start Training Script

In **Terminal 1**, start your training:

```bash
uv run python "scripts/training/train_rlaif.py" --config config.yaml
```

Let it run - you can attach Instruments at any time.

### Step 2: Find the Process ID (PID)

In **Terminal 2**, find the process ID:

**Option A - Using pgrep (simplest):**
```bash
pgrep -f train_rlaif.py
```
This prints just the PID(s).

**Option B - Using ps:**
```bash
ps aux | grep train_rlaif.py
```
Look for the process - PID is in the second column.

**Option C - Using Activity Monitor:**
1. Open Activity Monitor.app
2. Search for 'Python' or 'train_rlaif'
3. PID is shown in the PID column

### Step 3: Attach Instruments

#### Method A: Using Instruments GUI (Recommended)

1. **Open Instruments:**
   ```bash
   open /Applications/Xcode.app/Contents/Applications/Instruments.app
   ```

2. **Choose Template:**
   
   **📊 For CPU, Memory, GPU, Bandwidth (RECOMMENDED):**
   - Select **"System Trace"** template
   - Tracks: CPU, Memory, GPU, Network, Disk I/O, Threads
   - Best for comprehensive system monitoring
   
   **⚡ For code profiling (finding bottlenecks):**
   - Select **"Time Profiler"** template
   - Tracks: CPU time per function, call stacks
   - Best for finding slow code
   
   **💾 For memory analysis:**
   - Select **"Allocations"** template
   - Tracks: Memory allocations, growth, leaks
   - Best for memory issues
   
   Click "Choose"

3. **Attach to Process:**
   - Click the "Choose Target" dropdown (top left, next to Record button)
   - Select "Attach to Process"
   - In the process list, find your Python process:
     - Look for 'Python' with `train_rlaif.py` in the path
     - Or search by PID if you know it
   - Click "Choose"

4. **Start Profiling:**
   - Click the Record button (red circle)
   - Let it profile for as long as needed
   - Click Stop when done

5. **Analyze Results:**
   - The trace file opens automatically in Instruments
   - Use the Call Tree view to see where time is spent
   - Filter by your code: Type 'train_rlaif' in the search box
   - Look for hot spots (functions taking most time)

#### Method B: Using Command Line

**For System Trace (CPU, Memory, GPU, Bandwidth - RECOMMENDED):**
```bash
instruments -t "System Trace" -D system_trace.trace -p <PID>
```

**For Time Profiler (Code profiling):**
```bash
instruments -t "Time Profiler" -D time_profile.trace -p <PID>
```

**For Allocations (Memory):**
```bash
instruments -t "Allocations" -D allocations.trace -p <PID>
```

Replace `<PID>` with the actual process ID from Step 2.

The trace file will be saved. Open it with:
```bash
open system_trace.trace
# or
open time_profile.trace
```

## Choosing the Right Template

### 📊 System Trace (RECOMMENDED for Training Scripts)

**Best for:** Comprehensive system monitoring (CPU, Memory, GPU, Bandwidth)

**Tracks:**
- ✅ CPU utilization (per core, per process)
- ✅ Memory usage (RSS, virtual memory, allocations)
- ✅ GPU utilization (Metal/GPU activity)
- ✅ Network bandwidth (I/O, network activity)
- ✅ Disk I/O (read/write operations)
- ✅ Thread activity and context switches

**Command:**
```bash
instruments -t "System Trace" -D system_trace.trace -p <PID>
```

### ⚡ Time Profiler

**Best for:** Finding code bottlenecks, CPU hotspots

**Tracks:**
- ✅ CPU time per function
- ✅ Call stack with time spent
- ✅ Thread activity
- ❌ Limited GPU/memory details

**Command:**
```bash
instruments -t "Time Profiler" -D time_profile.trace -p <PID>
```

### 💾 Allocations

**Best for:** Memory allocations, growth, leaks

**Tracks:**
- ✅ Memory allocations (size, count)
- ✅ Memory growth over time
- ✅ Allocation call stacks
- ✅ Memory categories (heap, stack, etc.)

**Command:**
```bash
instruments -t "Allocations" -D allocations.trace -p <PID>
```

### 🔍 Leaks

**Best for:** Finding memory leaks

**Tracks:**
- ✅ Objects that should have been deallocated
- ✅ Leak call stacks

**Command:**
```bash
instruments -t "Leaks" -D leaks.trace -p <PID>
```

### 💡 Recommendation

For training scripts like `train_rlaif.py`, use **"System Trace"** - it gives you everything (CPU, Memory, GPU, Bandwidth) in one comprehensive view.

## Tips

### When to Profile

- **After warmup**: Start profiling after the first epoch completes
- **Specific phases**: Start/stop during generation vs training phases
- **Multiple profiles**: You can attach/detach multiple times during training
- **Low overhead**: Instruments has minimal impact on performance

### What to Look For

**System Trace:**
1. **CPU Tab:**
   - CPU usage per core, per process
   - CPU utilization over time
   - Thread activity

2. **Memory Tab:**
   - Memory usage (RSS, virtual) over time
   - Memory growth patterns
   - Memory pressure indicators

3. **GPU Tab:**
   - Metal/GPU activity
   - GPU utilization
   - GPU memory usage

4. **Network Tab:**
   - I/O operations
   - Network bandwidth
   - Disk read/write activity

**Time Profiler:**
1. **Call Tree View:**
   - Shows where time is spent in the call stack
   - Sort by "Self Time" to find hot functions
   - Expand to see callers

2. **Heavy Stack Trace:**
   - Shows the most expensive call paths
   - Useful for finding bottlenecks

**Allocations:**
1. **Memory Growth:**
   - See memory growth over time
   - Find allocation call stacks
   - Identify memory leaks

**General:**
- **Filtering:** Type `train_rlaif` in search to focus on your code
- **Hide system libraries:** Uncheck system frameworks to see your code clearly
- **Time Range:** Select specific time ranges to analyze specific phases

### Example Workflow

```bash
# Terminal 1: Start training
uv run python "scripts/training/train_rlaif.py" --config config.yaml

# Terminal 2: Wait for warmup, then find PID
pgrep -f train_rlaif.py
# Output: 12345

# Terminal 2: Attach Instruments (System Trace for comprehensive monitoring)
instruments -t "System Trace" -D training_profile.trace -p 12345

# Let it run for a few epochs, then stop (Ctrl+C)
# Open the trace
open training_profile.trace
```

## Alternative: Get Instructions from Script

You can also get these instructions programmatically:

```bash
# Get manual attachment instructions
uv run python scripts/profiling/profile_with_instruments.py --manual-instruments --script scripts/training/train_rlaif.py

# Get guide for choosing the right template
uv run python scripts/profiling/profile_with_instruments.py --instruments-templates
```

These print detailed step-by-step instructions.

## Troubleshooting

### Instruments Not Found

If Instruments isn't found, you need full Xcode.app (not just Command Line Tools):

1. Install Xcode from App Store
2. Or use alternative profiler: `--method pyspy`

### Can't Find Process

- Make sure training script is actually running
- Try `ps aux | grep python` to see all Python processes
- Check Activity Monitor for the process name

### Permission Issues

- Instruments may need accessibility permissions
- Go to System Settings > Privacy & Security > Accessibility
- Add Instruments.app if prompted

## See Also

- [Profile with Instruments automatically](./scripts/profiling/profile_with_instruments.py) - Automatic profiling
- [Profile with py-spy](./README.md#profiling) - Alternative profiler (no Xcode needed)

