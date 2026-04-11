class TrafficSignalAgent:
    """Deterministic heuristic controller for a two-phase traffic light."""

    def __init__(self):
        self.last_action = 0
        self.steps_since_switch = 0

    def reset(self):
        """Clear controller memory at the beginning of each task."""

        self.last_action = 0
        self.steps_since_switch = 0

    def choose_action(self, state):
        """Return 0 to keep the current signal or 1 to switch it.

        The policy is intentionally deterministic and lightweight. It improves
        reward by serving the busier axis, while avoiding rapid flickering that
        wastes green time and creates switch penalties.
        """

        ns_traffic = int(state.north) + int(state.south)
        ew_traffic = int(state.east) + int(state.west)
        total_traffic = ns_traffic + ew_traffic
        current_signal = getattr(state, "signal", "NS")
        phase_age = int(getattr(state, "phase_age", self.steps_since_switch))

        if total_traffic <= 0:
            self._remember(0, switched=False)
            return 0

        active_queue = ns_traffic if current_signal == "NS" else ew_traffic
        inactive_queue = ew_traffic if current_signal == "NS" else ns_traffic
        difference = abs(ns_traffic - ew_traffic)

        target_signal = "NS" if ns_traffic >= ew_traffic else "EW"
        target_is_current = target_signal == current_signal

        # Low traffic is best handled calmly: keep the light stable unless the
        # inactive approach is clearly waiting and the current approach is empty.
        if total_traffic <= 6:
            should_switch = inactive_queue >= active_queue + 3 and active_queue == 0
            return self._finish(1 if should_switch else 0)

        # A deadband treats small differences as equal, preventing oscillation
        # when both roads have similar demand.
        weak_difference = max(2, int(total_traffic * 0.15))
        if difference <= weak_difference:
            return self._finish(0)

        # Minimum green time preserves throughput. Switching too early loses a
        # service opportunity in the environment, so only urgent congestion can
        # override this guard.
        min_green_steps = 2
        strong_difference = max(5, int(total_traffic * 0.35))
        just_switched = self.last_action == 1 and self.steps_since_switch < min_green_steps
        if not target_is_current and (phase_age < min_green_steps or just_switched) and difference < strong_difference:
            return self._finish(0)

        # If the current green is already serving the busier axis, keep it.
        if target_is_current:
            return self._finish(0)

        # Congestion-based prioritization: switch when the inactive road has a
        # materially larger queue, especially after the current phase has aged.
        aged_phase = phase_age >= 3
        inactive_is_heavy = inactive_queue >= active_queue + weak_difference
        inactive_is_urgent = inactive_queue >= active_queue + strong_difference

        if inactive_is_urgent or (aged_phase and inactive_is_heavy):
            return self._finish(1)

        return self._finish(0)

    def _finish(self, action):
        self._remember(action, switched=action == 1)
        return action

    def _remember(self, action, switched):
        self.last_action = action
        if switched:
            self.steps_since_switch = 0
        else:
            self.steps_since_switch += 1


_agent = TrafficSignalAgent()


def reset_agent():
    _agent.reset()


def choose_action(state):
    return _agent.choose_action(state)
