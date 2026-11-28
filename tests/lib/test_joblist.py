"""``exp.lib.joblist``: a job list run in parallel.

The oracles: every line runs once as ``<cmd> <name> <gpu> <args>`` (its
arguments through, a blank and a comment line skipped), its output in its
own log and its exit code in the ``.done`` file; ``parallel`` jobs run at
once (each waits until all have started: it ends only if they did); two
jobs of one name are refused; the module imports the standard library only
(it runs as a file by the host's ``python3``).
"""

import ast
import sys
from pathlib import Path

import pytest

from exp.lib import joblist
from exp.lib.joblist import main, parse_list, run_list

JOB = r"""
import pathlib, sys, time
logs, name, gpu, *args = sys.argv[1:]
d = pathlib.Path(logs) / "started"
d.mkdir(exist_ok=True)
(d / name).touch()
need = int(args[0]) if args and args[0].isdigit() else 0
end = time.time() + 10
while len(list(d.iterdir())) < need and time.time() < end:
    time.sleep(0.01)
if len(list(d.iterdir())) < need:
    sys.exit(9)  # the others never started alongside
print(name, repr(gpu), *args)
sys.exit(3 if "fail" in args else 0)
"""


def test_the_list_runs_in_parallel(tmp_path: Path):
    lst = tmp_path / "std-00.list"
    lst.write_text("a 3 x\n\n# a comment\nb 3 'y z'\nc 3 fail\n")
    assert parse_list(lst.read_text()) == [("a", ["3", "x"]), ("b", ["3", "y z"]),
                                           ("c", ["3", "fail"])]  # fmt: skip
    logs = tmp_path / "logs"
    cmd = [sys.executable, "-c", JOB, str(logs)]
    rc = main(["--list", str(lst), "--logs", str(logs), "--parallel", "3", "--gpu",
               "0", "--", *cmd])  # fmt: skip
    assert rc == 1
    assert (logs / "a.log").read_text() == "a '0' 3 x\n"
    assert (logs / "b.log").read_text() == "b '0' 3 y z\n"
    done = (logs / "std-00.done").read_text().splitlines()
    assert sorted(line.split()[1:3] for line in done[:-1]) == [
        ["a", "rc=0"], ["b", "rc=0"], ["c", "rc=3"]]  # fmt: skip
    assert done[-1].split()[1] == "ALL"


def test_one_at_a_time_and_no_gpu(tmp_path: Path):
    logs = tmp_path / "logs"
    cmd = [sys.executable, "-c", JOB, str(logs)]
    codes = run_list(cmd, [("a", ["1"]), ("b", ["1"])], logs, parallel=1)
    assert codes == {"a": 0, "b": 0}
    assert (logs / "a.log").read_text() == "a '' 1\n"


def test_a_name_twice_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="twice"):
        run_list(["true"], [("a", []), ("a", [])], tmp_path, parallel=2)


def test_the_standard_library_only():
    tree = ast.parse(Path(joblist.__file__).read_text())
    names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)
             for a in n.names}  # fmt: skip
    names |= {n.module.split(".")[0] for n in ast.walk(tree)
              if isinstance(n, ast.ImportFrom) and n.module}  # fmt: skip
    assert names <= sys.stdlib_module_names
