def choose_action(state):
    """
    Rule-based traffic signal controller.

    Strategy:
    - Compute total vehicles in North-South (NS) and East-West (EW).
    - Prefer the direction with higher congestion.
    - Avoid unnecessary switching when traffic is balanced.

    Actions:
    - 0: keep current signal
    - 1: switch signal

    Args:
        state: An object with attributes:
               - north (int)
               - south (int)
               - east (int)
               - west (int)

    Returns:
        int: 0 or 1
    """

    # Aggregate traffic by axis
    ns_traffic = state.north + state.south
    ew_traffic = state.east + state.west

    # Decision logic
    if ns_traffic > ew_traffic:
        # Keep NS priority if it has more vehicles
        return 0
    elif ew_traffic > ns_traffic:
        # Switch to EW if it has more vehicles
        return 1
    else:
        # If equal, keep current signal to avoid oscillation
        return 0