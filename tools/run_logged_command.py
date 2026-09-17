#!/usr/bin/env python3
"""Run a command while atomically preserving its combined console stream."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("a command is required after --")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.log.with_suffix(args.log.suffix + ".incomplete")
    with temporary.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        return_code = process.wait()
    os.replace(temporary, args.log)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
