"""
Live Training Monitor & Tracker
===============================
Reads live training logs and displays real-time epoch stats, loss, Dice, IoU,
learning rate, and GPU status in your terminal.
"""

import os
import sys
import time
import re
import glob

def find_latest_training_log():
    # Check IDE task logs
    base_dir = os.path.expanduser(r"~\.gemini\antigravity-ide\brain\8bbea7b0-77c8-409d-a9ef-ad18e46d3bc2\.system_generated\tasks")
    if os.path.exists(base_dir):
        logs = glob.glob(os.path.join(base_dir, "task-*.log"))
        if logs:
            # Sort by modification time
            logs.sort(key=os.path.getmtime, reverse=True)
            for l in logs:
                try:
                    with open(l, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(500)
                        if "training" in content.lower() or "epoch" in content.lower() or "mask2former" in content.lower():
                            return l
                except Exception:
                    pass
    return None

def monitor():
    log_path = find_latest_training_log()
    if not log_path or not os.path.exists(log_path):
        print("Waiting for active training task log...")
        time.sleep(2)
        log_path = find_latest_training_log()
        if not log_path:
            print("No active training task log found.")
            return

    print("=" * 70)
    print(" LIVE SOLAR FILAMENT TRAINING TRACKER")
    print(f" Reading from: {os.path.basename(log_path)}")
    print(" Press Ctrl+C to exit tracker anytime (training will continue)")
    print("=" * 70)

    last_pos = 0
    while True:
        try:
            if not os.path.exists(log_path):
                time.sleep(1)
                continue

            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(last_pos)
                new_text = f.read()
                last_pos = f.tell()

                if new_text:
                    # Clean carriage returns for clean output
                    lines = new_text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and any(k in line for k in ["Epoch", "[SAVED]", "Starting", "Building", "DEVICE", "GPU:", "VRAM:", "AMP:"]):
                            print(line)
                        elif "Training:" in line or "Validating:" in line:
                            # Show latest progress bar line cleanly
                            parts = line.split('\r')
                            if parts:
                                sys.stdout.write(f"\r  {parts[-1].strip()[:75]} ")
                                sys.stdout.flush()

            time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nTracker closed. Training is still running in background.")
            break
        except Exception as e:
            time.sleep(1)

if __name__ == '__main__':
    monitor()
