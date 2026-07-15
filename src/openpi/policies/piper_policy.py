import dataclasses
from typing import ClassVar

import numpy as np

from openpi import transforms

_GRIPPER_INDICES = (6, 13)


def make_piper_example() -> dict:
    """Create an input example for a bimanual Piper policy."""
    return {
        "state": np.ones((14,), dtype=np.float32),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        },
        "prompt": "put the mango on the plate",
    }


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """Map bimanual Piper observations and actions into the pi0 convention.

    Piper gripper positions use zero for closed and ``gripper_open_position``
    for open. pi0 uses zero for open and one for closed.
    """

    gripper_open_position: float = 0.105

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    )

    def __post_init__(self) -> None:
        if not np.isfinite(self.gripper_open_position) or self.gripper_open_position <= 0:
            raise ValueError("gripper_open_position must be finite and greater than zero")

    def __call__(self, data: dict) -> dict:
        state = _require_vector(data["state"], name="state").copy()
        state[list(_GRIPPER_INDICES)] = _gripper_to_pi(state[list(_GRIPPER_INDICES)], self.gripper_open_position)

        input_images = data["images"]
        unexpected_cameras = set(input_images) - set(self.EXPECTED_CAMERAS)
        if unexpected_cameras:
            raise ValueError(
                f"Expected images to contain only {self.EXPECTED_CAMERAS}, got unexpected cameras "
                f"{tuple(sorted(unexpected_cameras))}"
            )
        if "cam_high" not in input_images:
            raise ValueError("images must contain cam_high")

        base_image = _convert_image(input_images["cam_high"])
        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}
        for destination, source in (
            ("left_wrist_0_rgb", "cam_left_wrist"),
            ("right_wrist_0_rgb", "cam_right_wrist"),
        ):
            if source in input_images:
                images[destination] = _convert_image(input_images[source])
                image_masks[destination] = np.True_
            else:
                images[destination] = np.zeros_like(base_image)
                image_masks[destination] = np.False_

        result = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }
        if "actions" in data:
            actions = _require_actions(data["actions"]).copy()
            actions[..., list(_GRIPPER_INDICES)] = _gripper_to_pi(
                actions[..., list(_GRIPPER_INDICES)], self.gripper_open_position
            )
            result["actions"] = actions
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    """Map pi0 actions back to raw bimanual Piper joint targets."""

    gripper_open_position: float = 0.105

    def __post_init__(self) -> None:
        if not np.isfinite(self.gripper_open_position) or self.gripper_open_position <= 0:
            raise ValueError("gripper_open_position must be finite and greater than zero")

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] < 14:
            raise ValueError(f"actions must have shape [horizon, >=14], got {actions.shape}")
        actions = actions[:, :14].copy()
        actions[:, list(_GRIPPER_INDICES)] = _gripper_from_pi(
            actions[:, list(_GRIPPER_INDICES)], self.gripper_open_position
        )
        return {"actions": actions}


def _require_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (14,):
        raise ValueError(f"{name} must have shape [14], got {array.shape}")
    return array


def _require_actions(value: np.ndarray) -> np.ndarray:
    actions = np.asarray(value)
    if actions.ndim != 2 or actions.shape[-1] != 14:
        raise ValueError(f"actions must have shape [horizon, 14], got {actions.shape}")
    return actions


def _convert_image(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"images must have three dimensions, got {image.shape}")
    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    elif image.shape[-1] != 3:
        raise ValueError(f"images must be CHW or HWC RGB arrays, got {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _gripper_to_pi(value: np.ndarray, open_position: float) -> np.ndarray:
    return 1.0 - np.clip(value / open_position, 0.0, 1.0)


def _gripper_from_pi(value: np.ndarray, open_position: float) -> np.ndarray:
    return open_position * (1.0 - np.clip(value, 0.0, 1.0))
