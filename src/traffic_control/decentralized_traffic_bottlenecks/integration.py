# integration.py  (overwrite the top and wrapper call)

from pathlib import Path

import sys                                   #  ← NEW

pkg_dir = Path(__file__).resolve().parent    #  ← NEW
if str(pkg_dir) not in sys.path:             #  ← NEW
    sys.path.insert(0, str(pkg_dir))         #  ← NEW

from .runner import run as tree_run                   # ← correct path & function
from .classes.run_config import RunConfig             # ← RunConfig definition lives here

from .enums import AlgoType, CostType


def run_tree_method(
    net_file: str,
    route_file: str,
    sumo_cfg: str,
    *,
    sumo_binary: str = "sumo-gui",
    extra_sumo_flags: list[str] | None = None,
) -> None:
    """
    Launch SUMO-GUI and let the Tree-method logic control every TLS.
    """
    extra_sumo_flags = extra_sumo_flags or []

    # --- build RunConfig with defaults ---
    # rc = RunConfig()          # upstream class has sensible defaults

    # --- derive paths expected by tree_run() ---
    work_dir   = Path(sumo_cfg).parent.as_posix()     # 'path'       arg
    output_dir = work_dir                             # same folder is fine

    rc = RunConfig(
        is_actuated=False,           # ← keep the original paper’s open-loop mode
        output_directory=output_dir,
        cost_type=CostType.QUEUE,    # ← same default used in the repo
        algo_type=AlgoType.TREE,     # ← the decentralized tree algorithm
        # any other **optional** params can be omitted—they already have defaults
    )


    # run() will create <work_dir>/simulation.sumocfg internally,
    # so we just pass the folder, NOT the file.
    tree_run(
        sumo_binary,
        work_dir,
        output_dir,
        net_file,
        rc,
    )
