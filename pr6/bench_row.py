#!/usr/bin/env python3
"""fixed_bench.py + DiT capture memory logs + saving the last timed video to $PR6_SAVE (PR 6)."""
import os
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
_check_output = subprocess.check_output


def check_output(cmd, *args, **kwargs):
    try:
        return _check_output(cmd, *args, **kwargs)
    except FileNotFoundError:
        if cmd and cmd[0] == "git":  # no git inside the distrobox; launch.sh records the SHA on the host
            return os.environ.get("PR6_SHA", "unknown") if "rev-parse" in cmd else ""
        raise


subprocess.check_output = check_output
import torch  # noqa: E402

from pr6 import hooks  # noqa: E402

hooks.install()
from evaluation.bench import fixed_bench as fb  # noqa: E402

_compare = fb.compare


def compare(video, reference, *args, **kwargs):
    if os.environ.get("PR6_SAVE"):
        torch.save(video.clone(), os.environ["PR6_SAVE"])
    return _compare(video, reference, *args, **kwargs)


fb.compare = compare

if __name__ == "__main__":
    try:
        sys.exit(fb.main())
    except BaseException as error:
        if not isinstance(error, SystemExit):
            traceback.print_exc()
            hooks.mem("at exception")
            print(torch.cuda.memory_summary(abbreviated=True), flush=True)
        raise
