from qos.error_mitigator.run import compute_gv_cost, compute_wc_cost
from qos.types.types import Qernel


def evolved_cost_search(self, q: Qernel, size_to_reach: int, budget: int):
    # OE_BEGIN
    gv_timeout_sec = 0
    wc_timeout_sec = 0
    gv_cost, _gv_timed_out = compute_gv_cost(q, size_to_reach, timeout_sec=gv_timeout_sec)
    wc_cost, _wc_timed_out = compute_wc_cost(q, size_to_reach, timeout_sec=wc_timeout_sec)
    while (gv_cost <= budget or wc_cost <= budget) and size_to_reach > 2:
        size_to_reach = size_to_reach - 1
        gv_cost, _gv_timed_out = compute_gv_cost(
            q, size_to_reach, timeout_sec=gv_timeout_sec
        )
        wc_cost, _wc_timed_out = compute_wc_cost(
            q, size_to_reach, timeout_sec=wc_timeout_sec
        )

    while gv_cost > budget and wc_cost > budget:
        size_to_reach = size_to_reach + 1
        gv_cost, _gv_timed_out = compute_gv_cost(
            q, size_to_reach, timeout_sec=gv_timeout_sec
        )
        wc_cost, _wc_timed_out = compute_wc_cost(
            q, size_to_reach, timeout_sec=wc_timeout_sec
        )

    method = "GV" if gv_cost <= wc_cost else "WC"
    # OE_END
    return size_to_reach, method
