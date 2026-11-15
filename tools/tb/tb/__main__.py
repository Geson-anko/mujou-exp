"""``python -m tb export --runs DIR... | --batch B... [--root runs] --out LOGDIR``
and ``python -m tb view LOGDIR [--port 6006]``."""

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from tb.export import export


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tb", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="write the events of run directories")
    e.add_argument("--runs", type=Path, nargs="*", default=[], help="run or batch dirs")
    e.add_argument("--batch", nargs="*", default=[], help="batches under --root")
    e.add_argument("--root", type=Path, default=Path("runs"), help="of --batch")
    e.add_argument("--out", type=Path, default=Path("tb"), help="the logdir")
    v = sub.add_parser("view", help="serve a logdir with TensorBoard")
    v.add_argument("logdir", type=Path, nargs="?", default=Path("tb"))
    v.add_argument("--port", type=int, default=6006)
    a = ap.parse_args(argv)
    if a.cmd == "export":
        paths = [*a.runs, *(a.root / b for b in a.batch)]
        missing = [p for p in paths if not p.is_dir()]
        if not paths or missing:
            ap.error(f"no run directories{f': {missing}' if missing else ''}")
        for target in export(paths, a.out):
            print(target)
        return 0
    argv_tb = ["tensorboard", "--logdir", str(a.logdir), "--port", str(a.port)]
    os.execvp(argv_tb[0], argv_tb)


if __name__ == "__main__":
    sys.exit(main())
