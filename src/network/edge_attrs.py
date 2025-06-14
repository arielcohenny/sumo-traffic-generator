import xml.etree.ElementTree as ET
import random
import numpy as np
from pathlib import Path

def assign_edge_attractiveness(
    seed: int,
    net_file_in: str | Path,
    net_file_out: str | Path = None,
    lambda_depart: float = 2.0,
    lambda_arrive: float = 2.0,
) -> None:
    """
    Adds 'depart_attractiveness' and 'arrive_attractiveness' attributes to each edge in a .net.xml file,
    with values sampled from a Poisson distribution.

    Parameters
    ----------
    net_file_in : path to input .net.xml
    net_file_out : path to output modified .net.xml (if None, input file is overwritten)
    lambda_depart : lambda for Poisson distribution of depart attractiveness
    lambda_arrive : lambda for Poisson distribution of arrive attractiveness
    seed : random seed for reproducibility
    """
    np.random.seed(seed)

    if net_file_out is None:
        net_file_out = net_file_in

    tree = ET.parse(net_file_in)
    root = tree.getroot()

    for edge in root.findall("edge"):
        depart_attr = np.random.poisson(lam=lambda_depart)
        arrive_attr = np.random.poisson(lam=lambda_arrive)
        edge.set("depart_attractiveness", str(depart_attr))
        edge.set("arrive_attractiveness", str(arrive_attr))

    tree.write(net_file_out, encoding="utf-8")
