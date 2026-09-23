from ..enums import CostType, AlgoType


class RunConfig:
    def __init__(self, is_actuated, output_directory, cost_type: CostType, algo_type: AlgoType, cost_iter_limit=0):
        self.is_actuated: bool = is_actuated
        self.name = output_directory
        self.cost_type = cost_type.name
        self.algo_type = algo_type.name
        self.cost_iter_limit = cost_iter_limit
