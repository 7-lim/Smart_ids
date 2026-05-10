"""
Pcap -> CICIDS-style flow features.

This is the bridge between **raw network traffic** and the model. The MLP
and Autoencoder were trained on the 77 CICIDS2017 flow features; this
module turns a pcap (or a directory of pcaps) into rows of those features
that `SmartTIDS_Predictor` can consume directly.

Two extraction backends are provided:

1. `extract_with_cicflowmeter(pcap_path, out_dir)` — preferred.
   Shells out to the official **CICFlowMeter** (Java) tool. Produces the
   full 77-feature schema with the same statistics CICIDS2017 was labelled
   with. Use this for production / serious evaluation. Requires Java + the
   CICFlowMeter jar to be installed; we just call the binary.

2. `extract_with_scapy(pcap_path)` — pure-Python fallback. Computes the
   ~50 most diagnostic features directly from packets. Good for demos and
   when CICFlowMeter is not available. Features it cannot compute
   (advanced bulk-rate stats, retransmission counts, etc.) default to 0 —
   this is safe because `SmartTIDS_Predictor.align_features` already
   tolerates missing keys by zero-filling.

CLI
---
    python -m src.flow_extractor capture.pcap                   # print to stdout
    python -m src.flow_extractor capture.pcap --json out.json   # save JSON
    python -m src.flow_extractor capture.pcap --predict         # also classify
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple


# =========================================================================== #
# Backend 1 -- CICFlowMeter (preferred)
# =========================================================================== #
def extract_with_cicflowmeter(
    pcap_path: str | Path,
    out_dir: str | Path,
    cmd: str = "cfm",
) -> Path:
    """
    Run CICFlowMeter on `pcap_path`. Returns the path of the produced CSV.

    Looks for the binary on PATH as `cfm` by default (the script CIC ships
    with the jar). Override `cmd` if your binary is named differently.
    Raises FileNotFoundError if the tool is not installed.
    """
    pcap_path = Path(pcap_path)
    out_dir   = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which(cmd) is None:
        raise FileNotFoundError(
            f"`{cmd}` not on PATH. Install CICFlowMeter "
            "(https://github.com/ahlashkari/CICFlowMeter) and ensure the "
            "wrapper script is callable, or use extract_with_scapy()."
        )

    subprocess.run([cmd, str(pcap_path), str(out_dir)], check=True)

    # CICFlowMeter writes <pcap_basename>_Flow.csv
    expected = out_dir / f"{pcap_path.stem}_Flow.csv"
    if not expected.exists():
        # Fall back to whatever .csv was just produced.
        candidates = sorted(out_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise RuntimeError(f"CICFlowMeter produced no CSV in {out_dir}")
        expected = candidates[-1]
    return expected


# =========================================================================== #
# Backend 2 -- scapy (pure Python fallback)
# =========================================================================== #
class _Flow:
    """Bidirectional flow accumulator keyed by (src, dst, sport, dport, proto)."""

    __slots__ = (
        "key", "first_ts", "last_ts",
        "fwd_lens", "bwd_lens", "fwd_times", "bwd_times",
        "fin", "syn", "rst", "psh", "ack", "urg", "ece", "cwe",
        "fwd_header_len", "bwd_header_len",
        "init_win_fwd", "init_win_bwd", "act_data_pkt_fwd", "min_seg_size_fwd",
    )

    def __init__(self, key):
        self.key = key
        self.first_ts: Optional[float] = None
        self.last_ts:  Optional[float] = None
        self.fwd_lens:  List[int]   = []
        self.bwd_lens:  List[int]   = []
        self.fwd_times: List[float] = []
        self.bwd_times: List[float] = []
        self.fin = self.syn = self.rst = self.psh = 0
        self.ack = self.urg = self.ece = self.cwe = 0
        self.fwd_header_len = 0
        self.bwd_header_len = 0
        self.init_win_fwd: Optional[int] = None
        self.init_win_bwd: Optional[int] = None
        self.act_data_pkt_fwd = 0
        self.min_seg_size_fwd: Optional[int] = None

    def add(self, ts, pkt_len, header_len, direction, flags=None,
            window=None, payload_len=0):
        if self.first_ts is None:
            self.first_ts = ts
        self.last_ts = ts
        if direction == "fwd":
            self.fwd_lens.append(pkt_len)
            self.fwd_times.append(ts)
            self.fwd_header_len += header_len
            if self.init_win_fwd is None and window is not None:
                self.init_win_fwd = window
            if payload_len > 0:
                self.act_data_pkt_fwd += 1
            if self.min_seg_size_fwd is None or header_len < self.min_seg_size_fwd:
                self.min_seg_size_fwd = header_len
        else:
            self.bwd_lens.append(pkt_len)
            self.bwd_times.append(ts)
            self.bwd_header_len += header_len
            if self.init_win_bwd is None and window is not None:
                self.init_win_bwd = window
        if flags:
            f = flags
            self.fin += "F" in f
            self.syn += "S" in f
            self.rst += "R" in f
            self.psh += "P" in f
            self.ack += "A" in f
            self.urg += "U" in f
            self.ece += "E" in f
            self.cwe += "C" in f

    def to_features(self) -> Dict[str, float]:
        """Return a 77-key feature dict (zero-filled where uncomputable)."""
        all_lens  = self.fwd_lens + self.bwd_lens
        all_times = sorted(self.fwd_times + self.bwd_times)

        duration = max((self.last_ts or 0) - (self.first_ts or 0), 0)  # seconds
        duration_us = duration * 1e6                                   # microseconds

        flow_iats = [b - a for a, b in zip(all_times[:-1], all_times[1:])]
        fwd_iats  = [b - a for a, b in zip(self.fwd_times[:-1], self.fwd_times[1:])]
        bwd_iats  = [b - a for a, b in zip(self.bwd_times[:-1], self.bwd_times[1:])]

        n_fwd, n_bwd = len(self.fwd_lens), len(self.bwd_lens)
        total_fwd_bytes = sum(self.fwd_lens)
        total_bwd_bytes = sum(self.bwd_lens)
        total_bytes     = total_fwd_bytes + total_bwd_bytes

        f = {name: 0.0 for name in CICIDS_FEATURES}                 # zero-fill
        proto = self.key[4]
        f.update({
            "Protocol":                        float(proto),
            "Flow Duration":                   duration_us,
            "Total Fwd Packets":               float(n_fwd),
            "Total Backward Packets":          float(n_bwd),
            "Total Length of Fwd Packets":     float(total_fwd_bytes),
            "Total Length of Bwd Packets":     float(total_bwd_bytes),
            "Fwd Packet Length Max":           float(max(self.fwd_lens, default=0)),
            "Fwd Packet Length Min":           float(min(self.fwd_lens, default=0)),
            "Fwd Packet Length Mean":          float(mean(self.fwd_lens) if self.fwd_lens else 0),
            "Fwd Packet Length Std":           float(pstdev(self.fwd_lens) if len(self.fwd_lens) > 1 else 0),
            "Bwd Packet Length Max":           float(max(self.bwd_lens, default=0)),
            "Bwd Packet Length Min":           float(min(self.bwd_lens, default=0)),
            "Bwd Packet Length Mean":          float(mean(self.bwd_lens) if self.bwd_lens else 0),
            "Bwd Packet Length Std":           float(pstdev(self.bwd_lens) if len(self.bwd_lens) > 1 else 0),
            "Flow Bytes/s":                    total_bytes  / duration if duration > 0 else 0,
            "Flow Packets/s":                  (n_fwd+n_bwd)/ duration if duration > 0 else 0,
            "Flow IAT Mean":                   float(mean(flow_iats)*1e6 if flow_iats else 0),
            "Flow IAT Std":                    float(pstdev(flow_iats)*1e6 if len(flow_iats) > 1 else 0),
            "Flow IAT Max":                    float(max(flow_iats)*1e6 if flow_iats else 0),
            "Flow IAT Min":                    float(min(flow_iats)*1e6 if flow_iats else 0),
            "Fwd IAT Total":                   float(sum(fwd_iats)*1e6),
            "Fwd IAT Mean":                    float(mean(fwd_iats)*1e6 if fwd_iats else 0),
            "Fwd IAT Std":                     float(pstdev(fwd_iats)*1e6 if len(fwd_iats) > 1 else 0),
            "Fwd IAT Max":                     float(max(fwd_iats)*1e6 if fwd_iats else 0),
            "Fwd IAT Min":                     float(min(fwd_iats)*1e6 if fwd_iats else 0),
            "Bwd IAT Total":                   float(sum(bwd_iats)*1e6),
            "Bwd IAT Mean":                    float(mean(bwd_iats)*1e6 if bwd_iats else 0),
            "Bwd IAT Std":                     float(pstdev(bwd_iats)*1e6 if len(bwd_iats) > 1 else 0),
            "Bwd IAT Max":                     float(max(bwd_iats)*1e6 if bwd_iats else 0),
            "Bwd IAT Min":                     float(min(bwd_iats)*1e6 if bwd_iats else 0),
            "Fwd Header Length":               float(self.fwd_header_len),
            "Bwd Header Length":               float(self.bwd_header_len),
            "Fwd Packets/s":                   n_fwd / duration if duration > 0 else 0,
            "Bwd Packets/s":                   n_bwd / duration if duration > 0 else 0,
            "Min Packet Length":               float(min(all_lens, default=0)),
            "Max Packet Length":               float(max(all_lens, default=0)),
            "Packet Length Mean":              float(mean(all_lens) if all_lens else 0),
            "Packet Length Std":               float(pstdev(all_lens) if len(all_lens) > 1 else 0),
            "Packet Length Variance":          float(pstdev(all_lens)**2 if len(all_lens) > 1 else 0),
            "FIN Flag Count":                  float(self.fin),
            "SYN Flag Count":                  float(self.syn),
            "RST Flag Count":                  float(self.rst),
            "PSH Flag Count":                  float(self.psh),
            "ACK Flag Count":                  float(self.ack),
            "URG Flag Count":                  float(self.urg),
            "CWE Flag Count":                  float(self.cwe),
            "ECE Flag Count":                  float(self.ece),
            "Down/Up Ratio":                   (n_bwd / n_fwd) if n_fwd else 0,
            "Average Packet Size":             float(total_bytes / max(n_fwd+n_bwd, 1)),
            "Avg Fwd Segment Size":            float(mean(self.fwd_lens) if self.fwd_lens else 0),
            "Avg Bwd Segment Size":            float(mean(self.bwd_lens) if self.bwd_lens else 0),
            "Subflow Fwd Packets":             float(n_fwd),
            "Subflow Fwd Bytes":               float(total_fwd_bytes),
            "Subflow Bwd Packets":             float(n_bwd),
            "Subflow Bwd Bytes":               float(total_bwd_bytes),
            "Init_Win_bytes_forward":          float(self.init_win_fwd if self.init_win_fwd is not None else -1),
            "Init_Win_bytes_backward":         float(self.init_win_bwd if self.init_win_bwd is not None else -1),
            "act_data_pkt_fwd":                float(self.act_data_pkt_fwd),
            "min_seg_size_forward":            float(self.min_seg_size_fwd or 0),
        })
        return f


def extract_with_scapy(
    pcap_path: str | Path,
    max_packets: int = 1_000_000,
) -> List[Dict[str, float]]:
    """
    Read `pcap_path` and return one feature dict per bidirectional flow.

    Lazy import of scapy so importing this module is cheap when only the
    CICFlowMeter backend is used.
    """
    try:
        from scapy.all import PcapReader, IP, IPv6, TCP, UDP    # type: ignore
    except ImportError as exc:
        raise ImportError(
            "scapy is required. Install with: pip install scapy"
        ) from exc

    flows: Dict[Tuple, _Flow] = {}

    with PcapReader(str(pcap_path)) as reader:
        for i, pkt in enumerate(reader):
            if i >= max_packets:
                break
            if pkt.haslayer(IP):
                ip = pkt[IP]; src, dst = ip.src, ip.dst
            elif pkt.haslayer(IPv6):
                ip = pkt[IPv6]; src, dst = ip.src, ip.dst
            else:
                continue

            if pkt.haslayer(TCP):
                t = pkt[TCP]
                sport, dport, proto = int(t.sport), int(t.dport), 6
                header_len = int(t.dataofs) * 4 if t.dataofs else 20
                window = int(t.window)
                payload_len = max(len(pkt) - (len(ip) - len(t)), 0)
                flag_str = str(t.flags)
            elif pkt.haslayer(UDP):
                u = pkt[UDP]
                sport, dport, proto = int(u.sport), int(u.dport), 17
                header_len = 8
                window = None
                payload_len = max(int(u.len) - 8, 0)
                flag_str = ""
            else:
                continue

            ts = float(pkt.time)
            pkt_len = len(pkt)
            fwd_key = (src, dst, sport, dport, proto)
            bwd_key = (dst, src, dport, sport, proto)

            if fwd_key in flows:
                flows[fwd_key].add(ts, pkt_len, header_len, "fwd",
                                   flag_str, window, payload_len)
            elif bwd_key in flows:
                flows[bwd_key].add(ts, pkt_len, header_len, "bwd",
                                   flag_str, window, payload_len)
            else:
                flows[fwd_key] = _Flow(fwd_key)
                flows[fwd_key].add(ts, pkt_len, header_len, "fwd",
                                   flag_str, window, payload_len)

    return [fl.to_features() for fl in flows.values()]


# =========================================================================== #
# Feature catalogue (matches data/cicids2017/features/feature_names.json)
# =========================================================================== #
CICIDS_FEATURES = [
    "Protocol", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]


# =========================================================================== #
# CLI
# =========================================================================== #
def _main():                                                   # pragma: no cover
    p = argparse.ArgumentParser(description="Extract CICIDS-style flow features from a pcap.")
    p.add_argument("pcap", help="Path to the .pcap or .pcapng file")
    p.add_argument("--json", help="Write features as JSON to this path")
    p.add_argument("--predict", action="store_true",
                   help="Also feed flows to SmartTIDS_Predictor and emit predictions")
    p.add_argument("--max-packets", type=int, default=1_000_000)
    args = p.parse_args()

    flows = extract_with_scapy(args.pcap, max_packets=args.max_packets)
    print(f"Extracted {len(flows)} flow(s) from {args.pcap}", file=sys.stderr)

    if args.predict:
        from src.inference import SmartTIDS_Predictor
        predictor = SmartTIDS_Predictor()
        results = predictor.predict_batch(flows)
        out = [{"flow_idx": i, **r} for i, r in enumerate(results)]
    else:
        out = flows

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}", file=sys.stderr)
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":                                     # pragma: no cover
    _main()
