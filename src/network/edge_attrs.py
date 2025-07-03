import xml.etree.ElementTree as ET
import numpy as np
from src.config import CONFIG


def assign_edge_attractiveness(seed: int) -> None:
    """
    Adds 'depart_attractiveness' and 'arrive_attractiveness' attributes to each edge in the .net.xml file,
    with values sampled from a Poisson distribution.

    Parameters
    ----------
    seed : random seed for reproducibility
    """
    np.random.seed(seed)

    net_file = CONFIG.network_file
    tree = ET.parse(net_file)
    root = tree.getroot()

    for edge in root.findall("edge"):
        depart_attr = np.random.poisson(lam=CONFIG.LAMBDA_DEPART)
        arrive_attr = np.random.poisson(lam=CONFIG.LAMBDA_ARRIVE)
        edge.set("depart_attractiveness", str(depart_attr))
        edge.set("arrive_attractiveness", str(arrive_attr))

    tree.write(net_file, encoding="utf-8")
