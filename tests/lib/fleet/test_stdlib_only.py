import importlib.util
import json
import subprocess
import sys

from tests.lib.fleet.fleetkit import REPO

FLEET = [
    "exp.lib.fleet.cli",
    "exp.lib.fleet.fake",
    "exp.lib.fleet.worker",
    "exp.lib.fleet.provision",
    "exp.lib.fleet.snapshot",
    "exp.lib.fleet.transport",
]
_PROBE = """
import importlib, json, sys
for m in sys.argv[1:]:
    importlib.import_module(m)
print(json.dumps(sorted(sys.modules)))
"""


class TestStdlibOnly:
    def test_the_fleet_imports_no_torch_hydra_or_other_lib_modules(self):
        mods = list(FLEET)
        if importlib.util.find_spec("exp.lib.runs") is not None:
            mods.append("exp.lib.runs")
        proc = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                f"import sys; sys.path.insert(0, {str(REPO)!r})\n" + _PROBE,
                *mods,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        loaded = set(json.loads(proc.stdout))
        assert not {"torch", "hydra", "omegaconf", "numpy", "mujou"} & loaded
        ours = {m for m in loaded if m == "exp" or m.startswith("exp.")}
        allowed = {"exp", "exp.lib", "exp.lib.fleet", "exp.lib.runs"}
        assert {m for m in ours if not m.startswith("exp.lib.fleet.")} <= allowed
