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
    assert parameters["required"] == ["start_s", "end_s", "fps", "resolution"]
    assert parameters["properties"]["fps"]["enum"] == [0.1, 0.2, 0.5, 1, 2]
    assert parameters["properties"]["resolution"]["enum"] == [256, 512, 768]
