from scripts.spike_vision import VIEW_FRAMES_TOOL, extract_timestamps, positional_accuracy


def test_timestamp_scoring_is_positional() -> None:
    actual = extract_timestamps("00:00, 00:30, 01:00")
    score = positional_accuracy(actual, ["00:00", "00:30", "01:00", "01:30"])

    assert actual == ["00:00", "00:30", "01:00"]
    assert score["correct"] == 3
    assert score["total"] == 4
    assert score["accuracy"] == 0.75


def test_view_frames_tool_has_required_m2_signature() -> None:
    function = VIEW_FRAMES_TOOL["function"]
    parameters = function["parameters"]

    assert function["name"] == "view_frames"
    assert parameters["required"] == ["start_s", "end_s", "fps", "n"]
    assert parameters["properties"]["fps"]["exclusiveMinimum"] == 0
    assert parameters["properties"]["n"]["minimum"] == 1
