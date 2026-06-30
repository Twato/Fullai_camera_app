#!/usr/bin/env python3
"""
Raspberry Pi system monitor for AI Camera tests.
Run after or during a camera test:
    python3 check_pi_system_status.py
    python3 check_pi_system_status.py --seconds 60 --interval 2

It prints CPU temperature, CPU usage, RAM, disk usage, throttling status,
and top CPU/RAM processes. No external Python packages required.
"""
import argparse
import os
import shutil
import subprocess
import time
from datetime import datetime


def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return f"N/A ({e})"


def get_temp_c():
    # Preferred on Raspberry Pi
    out = run_cmd(["vcgencmd", "measure_temp"])
    if out.startswith("temp="):
        try:
            return float(out.split("=")[1].replace("'C", ""))
        except Exception:
            pass
    # Fallback Linux thermal zone
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return None


def get_throttled():
    out = run_cmd(["vcgencmd", "get_throttled"])
    return out


def read_cpu_times():
    with open("/proc/stat", "r") as f:
        parts = f.readline().split()[1:]
    vals = list(map(int, parts))
    idle = vals[3] + vals[4]
    total = sum(vals)
    return idle, total


def get_cpu_usage(interval=0.4):
    idle1, total1 = read_cpu_times()
    time.sleep(interval)
    idle2, total2 = read_cpu_times()
    idle_delta = idle2 - idle1
    total_delta = total2 - total1
    if total_delta <= 0:
        return None
    return 100.0 * (1.0 - idle_delta / total_delta)


def get_mem_info():
    info = {}
    with open("/proc/meminfo", "r") as f:
        for line in f:
            key, val = line.split(":", 1)
            info[key] = int(val.strip().split()[0])  # kB
    total = info.get("MemTotal", 0)
    available = info.get("MemAvailable", 0)
    used = total - available
    return total / 1024, used / 1024, available / 1024, (used / total * 100 if total else 0)


def get_disk_usage(path="/"):
    total, used, free = shutil.disk_usage(path)
    gb = 1024 ** 3
    pct = used / total * 100 if total else 0
    return total / gb, used / gb, free / gb, pct


def get_load_avg():
    try:
        return os.getloadavg()
    except Exception:
        return None


def print_top_processes():
    print("\nTop CPU processes:")
    print(run_cmd(["bash", "-lc", "ps -eo pid,comm,%cpu,%mem --sort=-%cpu | head -8"]))
    print("\nTop RAM processes:")
    print(run_cmd(["bash", "-lc", "ps -eo pid,comm,%cpu,%mem --sort=-%mem | head -8"]))


def print_status(sample_no=None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    temp = get_temp_c()
    cpu = get_cpu_usage()
    mem_total, mem_used, mem_avail, mem_pct = get_mem_info()
    disk_total, disk_used, disk_free, disk_pct = get_disk_usage("/")
    load = get_load_avg()
    throttled = get_throttled()

    prefix = f"[{sample_no}] " if sample_no is not None else ""
    print("=" * 70)
    print(f"{prefix}{now}")
    print(f"CPU Temp       : {temp:.1f} °C" if temp is not None else "CPU Temp       : N/A")
    print(f"CPU Usage      : {cpu:.1f} %" if cpu is not None else "CPU Usage      : N/A")
    if load:
        print(f"Load Average   : {load[0]:.2f}, {load[1]:.2f}, {load[2]:.2f}")
    print(f"RAM            : {mem_used:.0f} / {mem_total:.0f} MB used ({mem_pct:.1f}%), available {mem_avail:.0f} MB")
    print(f"Disk /         : {disk_used:.1f} / {disk_total:.1f} GB used ({disk_pct:.1f}%), free {disk_free:.1f} GB")
    print(f"Throttle       : {throttled}")

    if temp is not None:
        if temp >= 80:
            print("WARNING        : Temperature is high. Consider fan/cooling or lower workload.")
        elif temp >= 70:
            print("NOTICE         : Temperature is warm. Watch for throttling.")

    if "0x0" not in str(throttled) and "N/A" not in str(throttled):
        print("WARNING        : Throttling/undervoltage flag is not 0x0. Check power/cooling.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=0, help="Monitor duration. 0 = one-shot.")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between samples.")
    parser.add_argument("--no-top", action="store_true", help="Do not print top processes.")
    args = parser.parse_args()

    if args.seconds <= 0:
        print_status()
        if not args.no_top:
            print_top_processes()
        return

    end = time.time() + args.seconds
    i = 1
    while time.time() < end:
        print_status(i)
        i += 1
        time.sleep(max(0.1, args.interval))
    if not args.no_top:
        print_top_processes()


if __name__ == "__main__":
    main()
