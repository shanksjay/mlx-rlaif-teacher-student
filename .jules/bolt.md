## 2024-05-23 - Static Hardware Queries in Loops
**Learning:** `subprocess.run` calls (e.g., `sysctl` to check hardware specs) are relatively expensive (milliseconds) and should never be placed in frequent monitoring or logging loops. Even if the call is fast, the accumulated overhead of forking/spawning checks can be significant (e.g., 60x slowdown compared to cached access).
**Action:** Always cache static hardware information (CPU model, total RAM, peak TFLOPS) on initialization or lazy-load it once, rather than querying it on every tick.
