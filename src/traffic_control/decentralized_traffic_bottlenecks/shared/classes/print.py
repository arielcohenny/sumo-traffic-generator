
class PrintData:
    def __init__(self, links, nodes, tl_node_ids, output_path):
        self.iterations = None
        self.links = links
        self.nodes = nodes
        self.tl_node_ids = tl_node_ids
        self.vehicle_total_time = None
        self.ended_vehicles_count = None
        self.started_vehicles_count = None
        self.costs = []
        self.driving_time_distribution = open(
            output_path + '/driving_time_distribution.txt', 'w')
        self.tree_cost_distribution = open(
            output_path + '/tree_cost_distribution.txt', 'w')
