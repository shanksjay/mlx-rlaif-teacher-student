#!/usr/bin/env python3
"""
Profile model loading using Apple Instruments and Python profilers

This script provides multiple profiling options:
1. Instruments (Time Profiler) - Apple's native profiler
2. py-spy - Python call stack profiler
3. cProfile - Python's built-in profiler
4. Memory profiling with memory_profiler
"""

import os
import sys
import time
import argparse
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def find_instruments():
    """Find the instruments executable"""
    # Check common locations
    possible_paths = [
        "instruments",  # In PATH
        "/usr/bin/instruments",  # System location
        "/Applications/Xcode.app/Contents/Developer/usr/bin/instruments",  # Xcode.app
        "/Applications/Xcode-beta.app/Contents/Developer/usr/bin/instruments",  # Xcode beta
    ]
    
    # Also search for Xcode installations
    import glob
    xcode_paths = glob.glob("/Applications/Xcode*.app/Contents/Developer/usr/bin/instruments")
    possible_paths.extend(xcode_paths)
    
    # Search more broadly in Xcode.app
    xcode_broad_search = glob.glob("/Applications/Xcode*.app/**/instruments", recursive=True)
    possible_paths.extend(xcode_broad_search)
    
    for path in possible_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    
    return None


def profile_with_instruments(script_path: str, output_dir: str = "./profiles"):
    """
    Profile using Apple Instruments (Time Profiler)
    
    Instruments provides:
    - Call stack with time spent in each function
    - Memory allocations
    - CPU usage per function
    - Thread analysis
    
    Note: Instruments requires full Xcode.app installation, not just Command Line Tools.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    trace_file = output_dir / "instruments.trace"
    
    logger.info("="*80)
    logger.info("Profiling with Apple Instruments (Time Profiler)")
    logger.info("="*80)
    logger.info(f"Output: {trace_file}")
    logger.info("\nThis will open Instruments.app with detailed profiling data.")
    logger.info("In Instruments, you can:")
    logger.info("  - View call stacks with time spent")
    logger.info("  - See memory allocations")
    logger.info("  - Analyze CPU usage per function")
    logger.info("  - View thread activity")
    logger.info("\nStarting profiling...\n")
    
    # Find instruments executable
    instruments_path = find_instruments()
    
    if not instruments_path:
        logger.error("Instruments not found!")
        logger.info("\nInstruments requires full Xcode.app installation.")
        logger.info("Command Line Tools alone are not sufficient.")
        logger.info("\nOptions:")
        logger.info("1. Install Xcode from the App Store:")
        logger.info("   - Open App Store")
        logger.info("   - Search for 'Xcode'")
        logger.info("   - Install (requires ~15GB)")
        logger.info("\n2. Use alternative profiling methods:")
        logger.info("   - py-spy: --method pyspy (recommended)")
        logger.info("   - cProfile: --method cprofile")
        logger.info("   - Memory profiler: --method memory")
        logger.info("\n3. Use Instruments GUI manually:")
        logger.info("   - Open Instruments.app")
        logger.info("   - Choose 'Time Profiler' template")
        logger.info("   - Attach to your Python process")
        return
    
    logger.info(f"Found Instruments at: {instruments_path}")
    
    # Resolve symlink to actual Python executable (Instruments can't handle symlinks)
    # os.path.realpath() follows all symlinks to the actual file
    python_executable = os.path.realpath(sys.executable)
    
    logger.info(f"Python executable (resolved): {python_executable}")
    
    # Verify the executable exists and is not a symlink
    if not os.path.exists(python_executable):
        logger.error(f"Python executable not found: {python_executable}")
        logger.info("Try using --method pyspy instead (doesn't require Instruments)")
        return
    
    # Double-check it's not still a symlink (shouldn't happen after realpath, but verify)
    if os.path.islink(python_executable):
        logger.warning(f"Python path is still a symlink after realpath resolution")
        logger.warning("This shouldn't happen. Instruments may fail.")
        logger.info("Try using --method pyspy instead (doesn't require Instruments)")
        return
    
    logger.info(f"Using Python executable: {python_executable}")
    
    # Instruments needs the command as a single string or properly quoted
    # The syntax is: instruments -t "Template" -D trace.trace <command> [args...]
    # We need to pass Python + script as separate arguments, not as a shell command
    
    # Instruments has issues with Python executables directly
    # Use a shell wrapper to avoid "cannot open files of this type" error
    script_abs_path = os.path.abspath(script_path)
    
    # Find the shell executable (Instruments needs full path)
    shell_path = "/bin/sh"  # Standard location on macOS
    if not os.path.exists(shell_path):
        # Fallback: try to find sh
        import shutil
        shell_path = shutil.which("sh") or "/bin/sh"
    
    if not os.path.exists(shell_path):
        logger.error(f"Shell executable not found: {shell_path}")
        logger.info("Cannot use Instruments command-line. Use GUI method or py-spy instead.")
        return
    
    # Method 1: Use shell wrapper (recommended)
    # This avoids Instruments trying to open Python as a document
    shell_cmd = f"{python_executable} {script_abs_path}"
    cmd = [
        instruments_path,
        "-t", "Time Profiler",
        "-D", str(trace_file),
        shell_path,  # Use full path to shell executable
        "-c", shell_cmd  # Command to execute
    ]
    
    logger.info(f"Command: {shell_cmd}")
    logger.info(f"Trace file: {trace_file}")
    logger.info("\nNote: Instruments will run the Python script and collect profiling data.")
    logger.info("This may take a while depending on your script...\n")
    
    try:
        # Run Instruments
        result = subprocess.run(cmd, check=False, text=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        
        # Check if trace file was created
        if trace_file.exists():
            logger.info(f"\n✓ Profiling complete!")
            logger.info(f"Trace file: {trace_file}")
            logger.info(f"Opening Instruments with trace file...")
            
            # Open the trace file in Instruments
            subprocess.run(["open", str(trace_file)])
        else:
            logger.warning(f"Trace file not found: {trace_file}")
            if result.returncode != 0:
                logger.error(f"Instruments exited with code: {result.returncode}")
                if result.stderr:
                    error_msg = result.stderr
                    logger.error(f"Error output: {error_msg}")
                    # Check for specific error about file types
                    if "cannot open files" in error_msg.lower() or "cannot open files of this type" in error_msg.lower():
                        logger.error("\n⚠️  Instruments cannot handle this file type (known limitation).")
                        logger.info("\n📋 Solutions:")
                        logger.info("\n1. Use Instruments GUI (most reliable):")
                        logger.info("   • Open: /Applications/Xcode.app/Contents/Applications/Instruments.app")
                        logger.info("   • Choose 'Time Profiler' template")
                        logger.info("   • Click Record button")
                        logger.info(f"   • In terminal, run: {python_executable} {script_abs_path}")
                        logger.info("   • Stop recording when script completes")
                        logger.info("\n2. Use py-spy instead (recommended, no Xcode issues):")
                        logger.info(f"   uv run python profile_with_instruments.py --method pyspy --script {script_path}")
                        logger.info("   This provides flamegraphs and call stacks without Instruments limitations.")
                if result.stdout:
                    logger.info(f"Output: {result.stdout}")
            else:
                logger.info("Instruments may have created it with a different name or location.")
                logger.info("Check the output directory for .trace files.")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running Instruments: {e}")
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        logger.info("\nTroubleshooting:")
        logger.info("1. Make sure Xcode.app is fully installed (not just Command Line Tools)")
        logger.info("2. Try opening Instruments.app manually to verify it works")
        logger.info("3. Use alternative profiling: --method pyspy")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.info("\nTry using alternative profiling methods:")
        logger.info("  --method pyspy (recommended, no Xcode needed)")
        logger.info("  --method cprofile")


def profile_with_pyspy(script_path: str, output_dir: str = "./profiles", duration: int = 60):
    """
    Profile using py-spy (Python call stack profiler)
    
    py-spy provides:
    - Real-time call stack sampling
    - Flamegraph visualization
    - Low overhead profiling
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    flamegraph_svg = output_dir / "flamegraph.svg"
    raw_output = output_dir / "pyspy_raw.txt"
    
    logger.info("="*80)
    logger.info("Profiling with py-spy")
    logger.info("="*80)
    logger.info(f"Duration: {duration} seconds")
    logger.info(f"Output: {flamegraph_svg}")
    logger.info("\nStarting profiling...\n")
    
    # Check if py-spy is installed
    try:
        subprocess.run(["py-spy", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("py-spy not found. Install it with:")
        logger.info("  uv pip install py-spy")
        logger.info("  Or: pip install py-spy")
        return
    
    # Get the process ID of the running script
    # We'll need to run the script and attach to it
    logger.info("Note: py-spy needs to attach to a running process.")
    logger.info("Run your script in one terminal, then in another terminal run:")
    logger.info(f"  py-spy record -o {flamegraph_svg} --pid <PID> --duration {duration}")
    logger.info("\nOr use the record command directly:")
    
    cmd = [
        "py-spy", "record",
        "-o", str(flamegraph_svg),
        "--subprocesses",
        "--", sys.executable, script_path
    ]
    
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info(f"\n✓ Profiling complete!")
        logger.info(f"Flamegraph saved to: {flamegraph_svg}")
        logger.info(f"Open it in a browser to view the call stack visualization.")
        
        # Try to open in browser
        try:
            subprocess.run(["open", str(flamegraph_svg)])
        except:
            pass
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running py-spy: {e}")


def profile_with_cprofile(script_path: str, output_dir: str = "./profiles"):
    """
    Profile using Python's cProfile
    
    cProfile provides:
    - Function-level timing
    - Call counts
    - Cumulative time
    - Can be visualized with snakeviz
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    profile_file = output_dir / "cprofile.prof"
    stats_file = output_dir / "cprofile_stats.txt"
    
    logger.info("="*80)
    logger.info("Profiling with cProfile")
    logger.info("="*80)
    logger.info(f"Output: {profile_file}")
    logger.info("\nStarting profiling...\n")
    
    import cProfile
    import pstats
    from io import StringIO
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Run the script
    start_time = time.time()
    profiler.enable()
    
    try:
        # Execute the script
        with open(script_path, 'r') as f:
            code = compile(f.read(), script_path, 'exec')
            exec(code, {'__name__': '__main__', '__file__': script_path})
    except Exception as e:
        logger.error(f"Error running script: {e}")
        return
    finally:
        profiler.disable()
        elapsed = time.time() - start_time
    
    # Save profile
    profiler.dump_stats(str(profile_file))
    logger.info(f"Profile saved to: {profile_file}")
    
    # Generate stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    with open(stats_file, 'w') as f:
        stats.print_stats(50, file=f)  # Top 50 functions
    
    logger.info(f"Stats saved to: {stats_file}")
    logger.info(f"Total time: {elapsed:.2f} seconds")
    
    # Print top 20 functions
    logger.info("\n" + "="*80)
    logger.info("Top 20 Functions by Cumulative Time:")
    logger.info("="*80)
    stats.print_stats(20)
    
    logger.info("\nTo visualize with snakeviz:")
    logger.info(f"  uv pip install snakeviz")
    logger.info(f"  snakeviz {profile_file}")


def profile_memory(script_path: str, output_dir: str = "./profiles"):
    """
    Profile memory usage with memory_profiler
    
    Provides:
    - Memory usage per line
    - Memory growth over time
    - Peak memory usage
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    mem_profile = output_dir / "memory_profile.txt"
    
    logger.info("="*80)
    logger.info("Profiling Memory Usage")
    logger.info("="*80)
    logger.info(f"Output: {mem_profile}")
    logger.info("\nStarting profiling...\n")
    
    try:
        from memory_profiler import profile
    except ImportError:
        logger.error("memory_profiler not installed. Install it with:")
        logger.info("  uv pip install memory-profiler")
        return
    
    # Run with memory profiler
    cmd = [
        sys.executable,
        "-m", "memory_profiler",
        script_path
    ]
    
    try:
        with open(mem_profile, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        
        logger.info(f"✓ Memory profile saved to: {mem_profile}")
        
        # Print summary
        with open(mem_profile, 'r') as f:
            lines = f.readlines()
            for line in lines[-20:]:  # Last 20 lines usually contain summary
                print(line, end='')
                
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running memory profiler: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Profile Python code with various tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile with Instruments (recommended for macOS)
  python profile_with_instruments.py --method instruments --script preload_model.py
  
  # Profile with py-spy (flamegraph)
  python profile_with_instruments.py --method pyspy --script preload_model.py
  
  # Profile with cProfile
  python profile_with_instruments.py --method cprofile --script preload_model.py
  
  # Profile memory usage
  python profile_with_instruments.py --method memory --script preload_model.py
  
  # Profile with all methods
  python profile_with_instruments.py --method all --script preload_model.py
        """
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['instruments', 'pyspy', 'cprofile', 'memory', 'all'],
        default='all',
        help='Profiling method to use'
    )
    
    parser.add_argument(
        '--script',
        type=str,
        required=True,
        help='Python script to profile'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./profiles',
        help='Output directory for profiling results'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Duration for py-spy profiling (seconds)'
    )
    
    args = parser.parse_args()
    
    # Check if script exists
    if not Path(args.script).exists():
        logger.error(f"Script not found: {args.script}")
        return
    
    methods = {
        'instruments': profile_with_instruments,
        'pyspy': profile_with_pyspy,
        'cprofile': profile_with_cprofile,
        'memory': profile_memory,
    }
    
    if args.method == 'all':
        # Skip instruments if not available, but try others
        available_methods = {}
        for name, func in methods.items():
            if name == 'instruments':
                instruments_path = find_instruments()
                if instruments_path:
                    available_methods[name] = func
                else:
                    logger.warning(f"Skipping {name} (not available - requires full Xcode.app)")
                    logger.info("  Use --method pyspy for similar functionality")
                    continue
            else:
                available_methods[name] = func
        
        for name, func in available_methods.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"Running {name} profiling...")
            logger.info(f"{'='*80}\n")
            try:
                if name == 'pyspy':
                    func(args.script, args.output_dir, args.duration)
                else:
                    func(args.script, args.output_dir)
            except Exception as e:
                logger.error(f"Error with {name}: {e}")
                continue
    else:
        func = methods[args.method]
        if args.method == 'pyspy':
            func(args.script, args.output_dir, args.duration)
        else:
            func(args.script, args.output_dir)


if __name__ == "__main__":
    main()

