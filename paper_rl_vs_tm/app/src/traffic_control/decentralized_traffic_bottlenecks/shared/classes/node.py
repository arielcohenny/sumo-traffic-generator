from math import ceil

import traci
import random

from .phase import Phase
from ..config import MIN_PHASE_TIME
from ..enums import AlgoType
from .algo_config import CostToStepSize


def define_tl_program(j_key, inx, duration):
    traci.trafficlight.setPhase(j_key, inx)
    traci.trafficlight.setPhaseDuration(j_key, duration)


class JunctionNode:
    def __init__(self, node_id, node_name, links_in, links_out, is_traffic_light, tl_name, steps):
        self.node_id = node_id
        self.links_from_me = links_out
        self.links_to_me = links_in
        self.is_traffic_light = is_traffic_light
        self.name = node_name
        self.phases = []
        self.tl = tl_name
        self.min_phase_time = MIN_PHASE_TIME
        self.phase_key_per_sec = [None] * steps
        self.phases_breakdown_per_iter = []

    def add_my_phases(self, junctions_dict, link_names, heads_to_tails, all_heads):
        # FIXED by Ariel on 2025-10-19
        # Initial phase durations from network file may violate MIN_PHASE_TIME constraint
        # and may sum to wrong cycle length (e.g., 90s from netgenerate but running with 60s interval).
        # Fix: Enforce MIN_PHASE_TIME and normalize to current cycle length during initialization.

        for inx, ph in enumerate(junctions_dict[self.name]["phases"]):
            phase_links = set()
            for head_name in ph['heads']:
                l_name = heads_to_tails[head_name]
                phase_links.add(link_names[l_name])
                all_heads[head_name].add_me_to_phase()

            # Clamp initial duration to MIN_PHASE_TIME
            initial_duration = max(ph['duration'], self.min_phase_time)
            self.phases.append(
                Phase(inx, initial_duration, ph['heads'], phase_links))

    def calc_phases_cost(self, all_heads, heads_costs, iteration):
        for phase in self.phases:
            phase.calc_my_cost(all_heads, heads_costs, iteration)

    def calc_wanted_program(self, seconds_in_cycle, current_trees, cost_type, all_links, all_heads, iteration, algo_type):
        heads_costs = {}
        for link_id in self.links_to_me:
            all_links[link_id].calc_heads_costs(
                heads_costs, current_trees, cost_type, all_heads)
        self.calc_phases_cost(all_heads, heads_costs, iteration)

        # DEBUG: Log phase costs and durations
        import logging
        logger = logging.getLogger(__name__)
        phase_costs = [phase.cost for phase in self.phases]
        phase_durations_before = [phase.duration for phase in self.phases]

        sum_cost = sum(phase.cost for phase in self.phases)
        if algo_type == AlgoType.NAIVE.name:
            time_to_play_with = seconds_in_cycle - \
                len(self.phases) * self.min_phase_time
            for phase_inx, phase in enumerate(self.phases):
                phase.define_duration(round(
                    phase.cost / sum_cost * time_to_play_with + self.min_phase_time), iteration)
        elif algo_type == AlgoType.BABY_STEPS.name:
            # FIXED by Ariel on 2025-10-19
            # The original implementation had two bugs:
            # 1. When left_over is negative (durations exceed seconds_in_cycle), the redistribution
            #    can push phase durations below MIN_PHASE_TIME
            # 2. Final durations don't always sum exactly to seconds_in_cycle
            # Fix: Enforce MIN_PHASE_TIME constraint AFTER redistribution and ensure exact sum

            cts = CostToStepSize()
            duration_step = cts.calc_duration_step(sum_cost)
            time_to_play_with = 0
            for phase in self.phases:
                time_to_play_with += phase.duration - \
                    max(phase.duration - duration_step, self.min_phase_time)
            new_phases_duration = []
            for phase_inx, phase in enumerate(self.phases):
                new_phases_duration.append(min(round(phase.cost / max(sum_cost, 1) * time_to_play_with +
                                                     max(phase.duration - duration_step, self.min_phase_time)),
                                               phase.duration + duration_step))
            left_over = seconds_in_cycle - sum(new_phases_duration)
            left_oer_base = left_over // len(self.phases)
            left_over_extra_count = left_over - \
                left_oer_base * len(self.phases)

            # Apply redistribution and enforce MIN_PHASE_TIME constraint
            final_durations = []
            for phase_inx, phase in enumerate(self.phases):
                phase_left_over = left_oer_base
                if phase_inx < left_over_extra_count:
                    phase_left_over += 1
                adjusted_duration = new_phases_duration[phase_inx] + phase_left_over
                # Clamp to minimum phase time
                final_durations.append(max(adjusted_duration, self.min_phase_time))

            # Ensure final durations sum exactly to seconds_in_cycle
            # This may require adjusting phases that are above minimum
            deficit = seconds_in_cycle - sum(final_durations)

            while deficit != 0:
                # Find phases that have room to adjust (above minimum if deficit<0, or any phase if deficit>0)
                adjustable_phases = []
                for i in range(len(final_durations)):
                    if deficit > 0:
                        # Need to add time - all phases can receive
                        adjustable_phases.append(i)
                    elif deficit < 0 and final_durations[i] > self.min_phase_time:
                        # Need to subtract time - only phases above minimum
                        adjustable_phases.append(i)

                if not adjustable_phases:
                    # Cannot adjust further without violating MIN_PHASE_TIME
                    # This shouldn't happen in practice, but log if it does
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Cannot enforce sum={seconds_in_cycle} without violating MIN_PHASE_TIME. "
                                 f"Final sum={sum(final_durations)}, deficit={deficit}")
                    break

                # Distribute deficit evenly across adjustable phases
                adjustment_per_phase = deficit // len(adjustable_phases)
                adjustment_extra_count = abs(deficit) - (abs(adjustment_per_phase) * len(adjustable_phases))

                for i, phase_idx in enumerate(adjustable_phases):
                    adjustment = adjustment_per_phase
                    if i < adjustment_extra_count:
                        adjustment += 1 if deficit > 0 else -1
                    final_durations[phase_idx] += adjustment
                    # Enforce minimum
                    final_durations[phase_idx] = max(final_durations[phase_idx], self.min_phase_time)

                # Recalculate deficit after adjustments
                new_deficit = seconds_in_cycle - sum(final_durations)
                if new_deficit == deficit:
                    # No progress made, avoid infinite loop
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Stuck adjusting deficit. sum={sum(final_durations)}, target={seconds_in_cycle}")
                    break
                deficit = new_deficit

            # Assign final durations to phases
            for phase_inx, phase in enumerate(self.phases):
                phase.define_duration(final_durations[phase_inx], iteration)
        elif algo_type == AlgoType.PLANNED or algo_type == AlgoType.RANDOM.name:
            for phase in self.phases:
                phase.define_duration(-1, iteration)
        elif algo_type == AlgoType.UNIFORM.name:
            dur = ceil(seconds_in_cycle / len(self.phases))
            for phase in self.phases:
                phase.define_duration(dur, iteration)

    def save_phase(self, sec):
        self.phase_key_per_sec[sec] = traci.trafficlight.getPhase(self.tl)

    def aggregate_phases_per_iter(self, iteration, seconds_in_cycle):
        data = self.phase_key_per_sec[iteration *
                                      seconds_in_cycle: (iteration + 1) * seconds_in_cycle:]
        phases_breakdown = {'switch_count': 0}
        prev_phase = data[0]
        for phase_key in data:
            if phase_key not in phases_breakdown:
                phases_breakdown[phase_key] = 0
            phases_breakdown[phase_key] += 1
            if phase_key != prev_phase:
                phases_breakdown['switch_count'] += 1
            prev_phase = phase_key
        self.phases_breakdown_per_iter.append(phases_breakdown)

    def update_traffic_light(self, inner_sec, algo_type):
        if algo_type == AlgoType.RANDOM.name:
            if inner_sec % MIN_PHASE_TIME == 0:
                phase_inx = random.randint(0, len(self.phases) - 1)
                define_tl_program(self.tl, phase_inx, MIN_PHASE_TIME)
            return
        secs_sum = 0
        for inx, phase in enumerate(self.phases):
            if inner_sec is secs_sum:
                define_tl_program(self.tl, inx, phase.duration)
                break
            secs_sum += phase.duration
