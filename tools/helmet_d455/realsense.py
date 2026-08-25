"""Optional pyrealsense2 device access isolated from offline geometry code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _rs() -> Any:
    try:
        import pyrealsense2 as rs
    except ImportError as error:
        raise RuntimeError(
            "pyrealsense2 is required only for a connected D455; offline tests do not require it"
        ) from error
    return rs


def _device_info(device: Any, rs: Any) -> dict[str, Any]:
    fields = {
        "name": rs.camera_info.name,
        "serial": rs.camera_info.serial_number,
        "firmware_version": rs.camera_info.firmware_version,
        "recommended_firmware_version": rs.camera_info.recommended_firmware_version,
        "usb_mode": rs.camera_info.usb_type_descriptor,
        "product_line": rs.camera_info.product_line,
        "product_id": rs.camera_info.product_id,
        "physical_port": rs.camera_info.physical_port,
    }
    return {
        key: str(device.get_info(identifier)) if device.supports(identifier) else "NOT_SUPPORTED"
        for key, identifier in fields.items()
    }


def enumerate_devices() -> list[dict[str, Any]]:
    rs = _rs()
    context = rs.context()
    result = []
    for device in context.query_devices():
        row = _device_info(device, rs)
        profiles = []
        for sensor in device.query_sensors():
            sensor_name = str(sensor.get_info(rs.camera_info.name)) if sensor.supports(rs.camera_info.name) else "UNKNOWN"
            for profile in sensor.get_stream_profiles():
                item: dict[str, Any] = {
                    "sensor": sensor_name,
                    "stream": str(profile.stream_type()),
                    "format": str(profile.format()),
                    "fps": int(profile.fps()),
                    "index": int(profile.stream_index()),
                }
                try:
                    video = profile.as_video_stream_profile()
                    item.update(width_px=int(video.width()), height_px=int(video.height()))
                except RuntimeError:
                    pass
                profiles.append(item)
        row["stream_profiles"] = profiles
        result.append(row)
    return result


def intrinsics_record(value: Any) -> dict[str, Any]:
    return {
        "width_px": int(value.width),
        "height_px": int(value.height),
        "fx_px": float(value.fx),
        "fy_px": float(value.fy),
        "cx_px": float(value.ppx),
        "cy_px": float(value.ppy),
        "matrix": [
            [float(value.fx), 0.0, float(value.ppx)],
            [0.0, float(value.fy), float(value.ppy)],
            [0.0, 0.0, 1.0],
        ],
        "distortion_model": str(value.model).split(".")[-1].lower(),
        "distortion_coefficients": [float(item) for item in value.coeffs],
    }


def extrinsics_record(value: Any) -> dict[str, Any]:
    rotation = np.asarray(value.rotation, dtype=np.float64).reshape(3, 3)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(value.translation, dtype=np.float64)
    return {
        "rotation": rotation.tolist(),
        "translation_m": [float(item) for item in value.translation],
        "target_from_source_matrix": result.tolist(),
    }


def _option_record(sensor: Any, option: Any) -> dict[str, Any]:
    if not sensor.supports(option):
        return {"supported": False}
    value = float(sensor.get_option(option))
    range_value = sensor.get_option_range(option)
    return {
        "supported": True,
        "value": value,
        "range": {
            "minimum": float(range_value.min),
            "maximum": float(range_value.max),
            "step": float(range_value.step),
            "default": float(range_value.default),
        },
        "read_only": bool(sensor.is_option_read_only(option)),
        "description": str(sensor.get_option_description(option)),
    }


def sensor_settings(sensor: Any) -> dict[str, Any]:
    rs = _rs()
    names = {
        "enable_auto_exposure": rs.option.enable_auto_exposure,
        "exposure": rs.option.exposure,
        "gain": rs.option.gain,
        "brightness": rs.option.brightness,
        "contrast": rs.option.contrast,
        "saturation": rs.option.saturation,
        "sharpness": rs.option.sharpness,
        "white_balance": rs.option.white_balance,
        "enable_auto_white_balance": rs.option.enable_auto_white_balance,
        "laser_power": rs.option.laser_power,
        "emitter_enabled": rs.option.emitter_enabled,
    }
    return {key: _option_record(sensor, option) for key, option in names.items()}


def set_manual_exposure(sensor: Any, exposure: float | None, gain: float | None) -> None:
    rs = _rs()
    if exposure is not None:
        if not sensor.supports(rs.option.exposure):
            raise RuntimeError("requested exposure is unsupported by this sensor")
        if sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.exposure, float(exposure))
    if gain is not None:
        if not sensor.supports(rs.option.gain):
            raise RuntimeError("requested gain is unsupported by this sensor")
        sensor.set_option(rs.option.gain, float(gain))


@dataclass
class D455Pipeline:
    pipeline: Any
    profile: Any
    device: Any
    color_sensor: Any
    depth_sensor: Any
    color_profile: Any
    depth_profile: Any
    rs: Any

    def stop(self) -> None:
        self.pipeline.stop()


def start_pipeline(
    *,
    serial: str | None,
    color: tuple[int, int, int],
    depth: tuple[int, int, int],
    color_exposure: float | None = None,
    color_gain: float | None = None,
    depth_exposure: float | None = None,
    depth_gain: float | None = None,
) -> D455Pipeline:
    rs = _rs()
    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.color, color[0], color[1], rs.format.rgb8, color[2])
    config.enable_stream(rs.stream.depth, depth[0], depth[1], rs.format.z16, depth[2])
    profile = pipeline.start(config)
    device = profile.get_device()
    actual_serial = str(device.get_info(rs.camera_info.serial_number))
    if serial and actual_serial != serial:
        pipeline.stop()
        raise RuntimeError(f"selected D455 serial mismatch: {actual_serial} != {serial}")
    product = str(device.get_info(rs.camera_info.name))
    if "D455" not in product.upper():
        pipeline.stop()
        raise RuntimeError(f"selected RealSense device is not a D455: {product}")
    color_sensor = device.first_color_sensor()
    depth_sensor = device.first_depth_sensor()
    set_manual_exposure(color_sensor, color_exposure, color_gain)
    set_manual_exposure(depth_sensor, depth_exposure, depth_gain)
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    return D455Pipeline(
        pipeline,
        profile,
        device,
        color_sensor,
        depth_sensor,
        color_profile,
        depth_profile,
        rs,
    )


def device_record(active: D455Pipeline) -> dict[str, Any]:
    return _device_info(active.device, active.rs)


__all__ = [
    "D455Pipeline",
    "device_record",
    "enumerate_devices",
    "extrinsics_record",
    "intrinsics_record",
    "sensor_settings",
    "set_manual_exposure",
    "start_pipeline",
]
