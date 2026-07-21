"""Safety bounds for canonical generated seed inputs."""

from interpreter.utils.gen_input import _apply_seed_variant


def test_random_seed_scalars_are_deterministic_and_trace_bounded():
    entries = [
        {"var": "$i0", "private": False, "value": 0},
        {"var": "$i1", "private": False, "value": 0},
    ]
    repeated = [dict(entry) for entry in entries]

    _apply_seed_variant(entries, "random")
    _apply_seed_variant(repeated, "random")

    values = [entry["value"] for entry in entries]
    assert values == [entry["value"] for entry in repeated]
    assert all(0 <= value <= 256 for value in values)
