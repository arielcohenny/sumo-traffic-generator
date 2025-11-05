
from ..config import IS_DIVIDE_COST


class Phase:
    def __init__(self, phase_id, duration, heads, links):
        self.phase_id = phase_id
        self.duration = duration
        self.heads = heads
        self.link_ids = links
        self.cost = 0

        # DEBUG: Track object ID and initial duration
        self._object_id = id(self)
        self._initial_duration = duration

        # DEBUG: Track creation time to detect object reuse
        import time
        self._creation_timestamp = time.time()

    def calc_my_cost(self, all_heads, heads_cost, iteration):
        self.cost = 0
        for head_name in self.heads:
            if head_name in heads_cost:
                if IS_DIVIDE_COST:
                    self.cost += heads_cost[head_name] / \
                        all_heads[head_name].phase_count
                else:
                    self.cost += heads_cost[head_name]

    def define_duration(self, duration, iteration):
        # DEBUG: Log duration changes at iteration 0
        if iteration == 0 and duration > 0:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"🔧 PHASE {self.phase_id} (obj={self._object_id}): duration {self.duration} → {duration}")

        if duration > 0:
            self.duration = duration
