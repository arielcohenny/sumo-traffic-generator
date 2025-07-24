# src/traffic_control/decentralized_traffic_bottlenecks/integration.py

import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from types import SimpleNamespace

from .enums import CostType, AlgoType
from .classes.run_config import RunConfig


def load_tree(
    net_file: str,
    *,
    cost_type: CostType = CostType.TREE_CURRENT,
    algo_type: AlgoType = AlgoType.BABY_STEPS,
    sumo_cfg: str | None = None
) -> tuple[SimpleNamespace, RunConfig]:
    """
    Instead of Tree Method’s JSON-based loader, we:
      1. (Optionally) copy the SUMO config to simulation.sumocfg
      2. Parse net_file for <tlLogic> entries
      3. Build a trivial tree_data with just the list of TLS IDs
    """
    # 1) Handle configuration file placement
    if sumo_cfg:
        work_dir = Path(sumo_cfg).parent
        target = work_dir / "simulation.sumocfg"
        if not target.exists():
            shutil.copyfile(sumo_cfg, target)
    else:
        work_dir = Path(net_file).parent

    # 2) Parse the network XML for traffic-light logic
    xml = ET.parse(net_file)
    root = xml.getroot()
    tls_elements = root.findall("tlLogic")
    tls_ids = [tl.get("id") for tl in tls_elements]

    if not tls_ids:
        raise RuntimeError(f"No <tlLogic> entries found in {net_file}")

    # 3) Build a minimal tree_data object
    tree_data = SimpleNamespace(tls_ids=tls_ids)

    # 4) Build the RunConfig exactly as before
    rc = RunConfig(
        is_actuated=False,
        output_directory=str(work_dir),
        cost_type=cost_type,
        algo_type=algo_type,
    )

    return tree_data, rc


def compute_phases(
    tree_data: SimpleNamespace,
    sim_time: int,
    run_config: RunConfig
) -> dict[str, str]:
    """
    A trivial 2-phase static policy:
      - All lights show "GGrr" (green for north-south, red for east-west).
      - Replace or extend this with your dynamic logic later.
    """
    return {tls_id: "GGrr" for tls_id in tree_data.tls_ids}
