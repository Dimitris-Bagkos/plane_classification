#!/usr/bin/env python3
"""
Plane audio cropper — script-relative defaults + verbose logging

Why your previous run likely did nothing: when you call
  python .\audio_cropping\cropper.py
the *current working directory* is your project root (Clap_tryout), but the
old script's defaults were relative to the **CWD**, not the script file.
So ../data/... pointed outside your project and found nothing, exiting quietly.

This version anchors defaults **relative to this file's location** and prints
clear messages (counts, missing dirs). It also adds a --verbose flag.

Defaults (relative to this file):
  INPUT DIRS:  ../data/Data 20 June 2025 , ../data/Data 01 August 2025
  OUTPUT DIR:  ../data/cropped

Usage:
  python cropper.py                # use both default input folders
  python cropper.py --one "../data/Data 20 June 2025"
  python cropper.py --length-sec 10 --verbose

Install once:
  pip install numpy scipy librosa soundfile tqdm
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import librosa

# ------------------ utilities ------------------

def script_root() -> Path:
    return Path(__file__).resolve().parent


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def butter_bandpass(lowcut: float, highcut: float, fs: int, order: int = 4):
    nyq = 0.5 * fs
    low = None if lowcut <= 0 else lowcut / nyq
    high = None if highcut <= 0 or highcut >= nyq else highcut / nyq
    if low and high:
        b, a = butter(order, [low, high], btype="band")
    elif low and not high:
        b, a = butter(order, low, btype="high")
    elif high and not low:
        b, a = butter(order, high, btype="low")
    else:
        return None
    return b, a


def apply_filter(y: np.ndarray, sr: int, lowcut: float, highcut: float) -> np.ndarray:
    ba = butter_bandpass(lowcut, highcut, sr, order=4)
    if ba is None:
        return y
    b, a = ba
    return filtfilt(b, a, y)


def rms_envelope(y: np.ndarray, sr: int, frame_ms: float, hop_ms: float):
    frame_length = max(1, int(sr * frame_ms / 1000.0))
    hop_length = max(1, int(sr * hop_ms / 1000.0))
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    return rms, hop_length


def smooth_envelope(x: np.ndarray, win_len: int) -> np.ndarray:
    if win_len <= 1:
        return x
    win = np.ones(win_len, dtype=float) / win_len
    return np.convolve(x, win, mode="same")


def median_abs_deviation(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def find_regions(mask: np.ndarray):
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts, ends))


def choose_anchor(env, hop, sr, min_sustain_sec, mad_k):
    med = float(np.median(env))
    mad = median_abs_deviation(env)
    thr = med + mad_k * mad
    mask = env > thr
    regions = find_regions(mask)
    frame_dur = hop / sr

    long_regions = []
    for s, e in regions:
        dur = (e - s) * frame_dur
        if dur >= min_sustain_sec:
            long_regions.append((s, e, dur))

    diag = {"threshold": thr, "median": med, "mad": mad, "num_regions": len(regions), "fallback": ""}

    if long_regions:
        s, e, _ = max(long_regions, key=lambda r: r[2])
        idx = s + int(np.argmax(env[s:e]))
        return idx * frame_dur, diag
    if regions:
        s, e = max(regions, key=lambda r: (r[1] - r[0]))
        idx = s + int(np.argmax(env[s:e]))
        diag["fallback"] = "used_short_region"
        return idx * frame_dur, diag
    idx = int(np.argmax(env))
    diag["fallback"] = "used_global_peak"
    return idx * frame_dur, diag


def clamp_window(center: float, length: float, total: float):
    half = 0.5 * length
    start = max(0.0, center - half)
    end = min(total, center + half)
    if end - start < length:
        # pad right then left as needed
        deficit = length - (end - start)
        start = max(0.0, start - deficit)
        end = min(total, start + length)
    return float(start), float(end)

# ------------------ core ------------------

def process_file(in_path: Path, out_root_for_input: Path, in_root: Path, params: Dict, verbose=False) -> Dict:
    rel = in_path.relative_to(in_root)
    out_path = out_root_for_input / rel
    ensure_parent_dir(out_path)

    if out_path.exists() and not params["overwrite"]:
        return {"input_root": str(in_root), "file": str(rel).replace("\\", "/"), "status": "skipped_exists"}

    info = sf.info(str(in_path))
    sr = info.samplerate
    data, sr = sf.read(str(in_path), always_2d=True, dtype="float32")
    dur = data.shape[0] / float(sr)

    if dur <= params["length_sec"] and params["copy_short"]:
        shutil.copy2(str(in_path), str(out_path))
        return {"input_root": str(in_root), "file": str(rel).replace("\\", "/"), "status": "copied_unchanged", "duration_sec": f"{dur:.3f}"}

    mono = data.mean(axis=1)
    if params["proc_sr"] != sr:
        y = librosa.resample(mono, orig_sr=sr, target_sr=params["proc_sr"], res_type="kaiser_best")
        sr_proc = params["proc_sr"]
    else:
        y = mono
        sr_proc = sr

    y = apply_filter(y, sr_proc, params["lowcut"], params["highcut"])  # band-pass

    env, hop = rms_envelope(y, sr_proc, params["rms_frame_ms"], params["rms_hop_ms"])
    frames_per_sec = sr_proc / hop
    env = smooth_envelope(env, max(1, int(round(params["smooth_sec"] * frames_per_sec))))

    anchor_sec, diag = choose_anchor(env, hop, sr_proc, params["min_sustain_sec"], params["mad_k"])
    start_sec, end_sec = clamp_window(anchor_sec, params["length_sec"], dur)
    s = int(round(start_sec * sr))
    e = int(round(end_sec * sr))
    cropped = data[s:e, :]

    sf.write(str(out_path), cropped, sr, subtype=info.subtype or None)

    if verbose:
        print(f"  -> {rel} | dur={dur:.2f}s | anchor={anchor_sec:.2f}s | window=({start_sec:.2f},{end_sec:.2f})")

    return {
        "input_root": str(in_root),
        "file": str(rel).replace("\\", "/"),
        "status": "cropped",
        "duration_sec": f"{dur:.3f}",
        "anchor_sec": f"{anchor_sec:.3f}",
        "start_sec": f"{start_sec:.3f}",
        "end_sec": f"{end_sec:.3f}",
        "threshold": f"{diag['threshold']:.6f}",
        "median_env": f"{diag['median']:.6f}",
        "mad_env": f"{diag['mad']:.6f}",
        "num_regions": str(diag["num_regions"]),
        "fallback": diag["fallback"],
    }

# ------------------ CLI ------------------

def main():
    base = script_root()

    parser = argparse.ArgumentParser(description="Crop plane WAVs to most interesting section (script-relative paths)")
    parser.add_argument("--one", type=Path, default=None, help="Process only this input directory; overrides defaults")
    parser.add_argument("--output-dir", type=Path, default=base.parent / "data" / "cropped", help="Output root directory")
    parser.add_argument("--length-sec", type=float, default=8.0, help="Crop window length (seconds)")
    parser.add_argument("--proc-sr", type=int, default=32000, help="Processing sample rate")
    parser.add_argument("--lowcut", type=float, default=80.0, help="Band-pass lowcut (Hz; 0 disables)")
    parser.add_argument("--highcut", type=float, default=2000.0, help="Band-pass highcut (Hz; 0 disables)")
    parser.add_argument("--rms-frame-ms", type=float, default=50.0)
    parser.add_argument("--rms-hop-ms", type=float, default=10.0)
    parser.add_argument("--smooth-sec", type=float, default=0.3)
    parser.add_argument("--mad-k", type=float, default=1.0)
    parser.add_argument("--min-sustain-sec", type=float, default=2.0)
    parser.add_argument("--no-copy-short", dest="copy_short", action="store_false", help="Do not copy short files unchanged")
    parser.set_defaults(copy_short=True)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-file details")

    args = parser.parse_args()

    # Resolve defaults relative to this file
    default_inputs = [
        base.parent / "data" / "Data 20 June 2025",
        base.parent / "data" / "Data 01 August 2025",
    ]

    input_dirs = [args.one] if args.one else default_inputs

    params = {
        "proc_sr": args.proc_sr,
        "lowcut": args.lowcut,
        "highcut": args.highcut,
        "rms_frame_ms": args.rms_frame_ms,
        "rms_hop_ms": args.rms_hop_ms,
        "smooth_sec": args.smooth_sec,
        "mad_k": args.mad_k,
        "min_sustain_sec": args.min_sustain_sec,
        "length_sec": args.length_sec,
        "copy_short": bool(args.copy_short),
        "overwrite": bool(args.overwrite),
    }

    out_root = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    print("Cropper starting…")
    print(f"  Script dir:  {base}")
    print(f"  CWD:         {Path.cwd()}")
    print(f"  Output dir:  {out_root}")

    total_found = 0
    total_done = 0
    log_rows: List[Dict] = []

    for in_root in input_dirs:
        in_root = in_root.resolve()
        print(f"\nScanning: {in_root}")
        if not in_root.exists():
            print("  !! Missing input directory — skipping")
            log_rows.append({"input_root": str(in_root), "file": "-", "status": "error", "note": "input folder not found"})
            continue

        # Discover wavs
        wavs: List[Path] = []
        for root, _, files in os.walk(in_root):
            for f in files:
                if f.lower().endswith(".wav"):
                    wavs.append(Path(root) / f)
        print(f"  Found {len(wavs)} wav files")
        total_found += len(wavs)

        out_root_for_input = out_root / in_root.name
        out_root_for_input.mkdir(parents=True, exist_ok=True)

        for fpath in tqdm(sorted(wavs), desc=f"Cropping {in_root.name}"):
            try:
                row = process_file(fpath, out_root_for_input, in_root, params, verbose=args.verbose)
                total_done += 1 if row.get("status") in {"cropped", "copied_unchanged", "skipped_exists"} else 0
            except Exception as e:
                row = {"input_root": str(in_root), "file": str(fpath.relative_to(in_root)).replace("\\", "/"), "status": "error", "note": repr(e)}
            log_rows.append(row)

    # Write combined log
    log_path = out_root / "crop_log.csv"
    keys = set()
    for r in log_rows:
        keys.update(r.keys())
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(keys))
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nDone. Found {total_found} wavs across inputs. Processed {total_done}. Log: {log_path}")


if __name__ == "__main__":
    main()
