from interpreter.coverage_gen.driver import _result_path_features


def test_result_path_features_unions_sequence_and_compact_edges():
    sequence_edges = _result_path_features({
        "block_sequence": ("entry", "body", "exit"),
        "covered_edges": (),
    })
    combined_edges = _result_path_features({
        "block_sequence": ("entry", "body", "exit"),
        "covered_edges": (("entry", "rare"),),
    })

    assert sequence_edges["edges"] == (
        ("body", "exit"),
        ("entry", "body"),
    )
    assert set(combined_edges["edges"]) == {
        ("entry", "body"),
        ("body", "exit"),
        ("entry", "rare"),
    }


def test_compact_edge_presence_disables_sorted_block_fallback():
    compact = _result_path_features(
        {"block_sequence": (), "covered_edges": ()},
        covered={"entry", "body", "exit"},
    )
    legacy = _result_path_features(
        {"block_sequence": ()},
        covered={"entry", "body", "exit"},
    )

    assert compact["edges"] == ()
    assert legacy["edges"] == (("body", "entry"), ("entry", "exit"))
