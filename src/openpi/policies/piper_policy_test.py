import numpy as np
import pytest

from openpi.policies import piper_policy


def test_inputs_convert_grippers_and_cameras_without_mutating_source():
    state = np.arange(14, dtype=np.float32)
    state[[6, 13]] = [0.0, 0.105]
    actions = np.stack([state, state], axis=0)
    actions[:, [6, 13]] = [[0.0525, 0.105], [0.0, 0.2]]
    source_state = state.copy()
    source_actions = actions.copy()
    data = {
        "state": state,
        "actions": actions,
        "images": {
            "cam_high": np.ones((3, 4, 5), dtype=np.float32),
            "cam_left_wrist": np.full((4, 5, 3), 7, dtype=np.uint8),
            "cam_right_wrist": np.zeros((3, 4, 5), dtype=np.uint8),
        },
        "prompt": "put the mango on the plate",
    }

    result = piper_policy.PiperInputs()(data)

    np.testing.assert_allclose(result["state"][[6, 13]], [1.0, 0.0])
    np.testing.assert_allclose(result["actions"][:, [6, 13]], [[0.5, 0.0], [1.0, 0.0]])
    np.testing.assert_array_equal(result["state"][:6], state[:6])
    np.testing.assert_array_equal(result["actions"][:, :6], actions[:, :6])
    assert result["image"]["base_0_rgb"].shape == (4, 5, 3)
    assert result["image"]["base_0_rgb"].dtype == np.uint8
    assert np.all(result["image"]["base_0_rgb"] == 255)
    assert all(result["image_mask"].values())
    assert result["prompt"] == data["prompt"]
    np.testing.assert_array_equal(state, source_state)
    np.testing.assert_array_equal(actions, source_actions)


def test_inputs_fill_missing_wrist_cameras():
    result = piper_policy.PiperInputs()(
        {
            "state": np.zeros(14, dtype=np.float32),
            "images": {"cam_high": np.ones((3, 2, 3), dtype=np.uint8)},
        }
    )

    assert result["image_mask"]["base_0_rgb"]
    assert not result["image_mask"]["left_wrist_0_rgb"]
    assert not result["image_mask"]["right_wrist_0_rgb"]
    assert not np.any(result["image"]["left_wrist_0_rgb"])
    assert not np.any(result["image"]["right_wrist_0_rgb"])


def test_outputs_convert_and_clip_grippers():
    actions = np.zeros((2, 32), dtype=np.float32)
    actions[:, [6, 13]] = [[0.0, 0.5], [-1.0, 2.0]]

    result = piper_policy.PiperOutputs()({"actions": actions})["actions"]

    assert result.shape == (2, 14)
    np.testing.assert_allclose(result[:, [6, 13]], [[0.105, 0.0525], [0.105, 0.0]])


def test_gripper_conversion_round_trip():
    raw_actions = np.zeros((1, 14), dtype=np.float32)
    raw_actions[0, [6, 13]] = [0.02, 0.08]
    encoded = piper_policy.PiperInputs()(
        {
            "state": raw_actions[0],
            "actions": raw_actions,
            "images": {"cam_high": np.zeros((3, 2, 2), dtype=np.uint8)},
        }
    )["actions"]

    decoded = piper_policy.PiperOutputs()({"actions": encoded})["actions"]

    np.testing.assert_allclose(decoded, raw_actions, atol=1e-7)


def test_inputs_reject_invalid_shape_and_camera():
    transform = piper_policy.PiperInputs()
    with pytest.raises(ValueError, match="state must have shape"):
        transform({"state": np.zeros(13), "images": {"cam_high": np.zeros((3, 2, 2))}})
    with pytest.raises(ValueError, match="unexpected cameras"):
        transform(
            {
                "state": np.zeros(14),
                "images": {
                    "cam_high": np.zeros((3, 2, 2)),
                    "cam_unknown": np.zeros((3, 2, 2)),
                },
            }
        )
