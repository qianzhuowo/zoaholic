import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.request_helpers import build_attempt_provider_list


def test_weight_slots_do_not_expand_attempt_list_for_single_provider():
    """单渠道即使被权重展开成多个槽位，也只应尝试一次该渠道。"""
    providers = [{"provider": "p1"}] * 100

    for start_index in range(5):
        ordered = build_attempt_provider_list(providers, start_index=start_index)
        assert [p["provider"] for p in ordered] == ["p1"]



def test_weight_slots_preserve_weighted_first_provider_distribution():
    """固定优先级 + 权重槽位时，通过移动起点保持首选渠道的权重占比。"""
    providers = [
        {"provider": "A"},
        {"provider": "B"},
        {"provider": "A"},
        {"provider": "A"},
    ]

    first_providers = []
    for start_index in range(8):
        ordered = build_attempt_provider_list(providers, start_index=start_index)
        first_providers.append(ordered[0]["provider"])

    assert first_providers.count("A") == 6
    assert first_providers.count("B") == 2



def test_round_robin_keeps_rotating_on_unique_providers():
    """非 fixed_priority 下，唯一渠道列表保持轮转。"""
    providers = [
        {"provider": "A"},
        {"provider": "B"},
        {"provider": "C"},
    ]

    first_providers = []
    for start_index in range(6):
        ordered = build_attempt_provider_list(providers, start_index=start_index)
        first_providers.append(ordered[0]["provider"])

    assert first_providers == ["A", "B", "C", "A", "B", "C"]
