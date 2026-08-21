"""
Live Solar Filament Training Monitor (Continuous Progress Stream)
=================================================================
Tails the active training log and shows the live progress bar, batch metrics,
and epoch completion status in real-time.

Usage:
    python monitor_training.py
"""

import os
import sys
import time
import glob
import subprocess

def get_latest_task_log():
    """Finds the most recently updated task log file."""
    base_dir = os.path.expanduser(r"~\.gemini\antigravity-ide\brain\8bbea7b0-77c8-409d-a9ef-ad18e46d3bc2\.system_generated\tasks")
    if os.path.exists(base_dir):
        logs = glob.glob(os.path.join(base_dir, "task-*.log"))
        if logs:
            logs.sort(key=os.path.getmtime, reverse=True)
            for l in logs:
                try:
                    with open(l, 'r', encoding='utf-8', errors='ignore') as f:
                        txt = f.read(600)
                        if "patch refiner" in txt.lower() or "mask2former" in txt.lower() or "epoch" in txt.lower():
                            return l
                except Exception:
                    pass
            return logs[0]
    return None

def main():
    log_path = get_latest_task_log()
    if not log_path or not os.path.exists(log_path):
        print("[!] No active task log found. Waiting...")
        time.sleep(2)
        log_path = get_latest_task_log()
        if not log_path:
            print("[!] Could not locate active task log.")
            return

    print("=" * 75)
    print(" [*] LIVE TRAINING PROGRESS STREAM")
    print(f" [*] Watching Log: {os.path.basename(log_path)}")
    print(" [*] Press Ctrl+C at any time to exit (Training runs in background)")
    print("=" * 75)

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        # Seek near end
        f.seek(max(0, os.path.getsize(log_path) - 2000))
        initial_chunk = f.read()
        if initial_chunk:
            sys.stdout.write(initial_chunk)
            sys.stdout.flush()

        while True:
            try:
                line = f.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                else:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\n\n[*] Stopped progress monitor. Training is still running.")
                break

if __name__ == '__main__':
    main()
