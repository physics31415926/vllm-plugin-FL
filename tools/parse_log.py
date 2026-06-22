#!/usr/bin/env python3
"""Parse baseline.log: normalize \r, extract error lines with context."""
import sys

log_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/adapt-logs/baseline.log"
mode = sys.argv[2] if len(sys.argv) > 2 else "errors"

with open(log_path, "rb") as f:
    raw = f.read()

text = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n").decode("utf-8", errors="replace")
lines = text.split("\n")

if mode == "errors":
    keywords = ["Error", "error", "Traceback", "Exception", "cause", "Failed",
                "failed", "ImportError", "AttributeError", "RuntimeError",
                "TypeError", "ValueError", "raise ", "assert "]
    context = 3
    printed = set()
    for i, line in enumerate(lines):
        if any(k in line for k in keywords):
            start = max(0, i - context)
            end = min(len(lines), i + context + 1)
            for j in range(start, end):
                if j not in printed:
                    print(f"{j+1:5d}: {lines[j]}")
                    printed.add(j)
            print("---")
elif mode == "tail":
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    for i, line in enumerate(lines[-n:], len(lines) - n):
        print(f"{i+1:5d}: {line}")
elif mode == "full":
    for i, line in enumerate(lines):
        print(f"{i+1:5d}: {line}")
