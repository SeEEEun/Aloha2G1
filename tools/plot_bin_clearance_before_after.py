#!/usr/bin/env python3
"""Plot the persisted original-bin and terminal bounded-bin contact audits."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(root: Path) -> dict[str, np.ndarray]:
    with np.load(root / "event_log.npz", allow_pickle=False) as event:
        physics_step = event["physics_step"].copy()
        time_s = event["timestamp_s"].copy()
        speed = np.linalg.norm(event["object_linear_velocity_m_s"], axis=1)
    with np.load(root / "robot_bin_contacts.npz", allow_pickle=False) as contacts:
        force_by_step: dict[int, float] = {}
        for step, force in zip(contacts["physics_step"], contacts["force_n"]):
            if force > 1.0e-6:
                key = int(step)
                force_by_step[key] = max(force_by_step.get(key, 0.0), float(force))
        maximum_penetration_m = float(contacts["penetration_m"].max(initial=0.0))
        links = sorted(set(contacts["robot_link"].astype(str)))
    force = np.asarray([force_by_step.get(int(step), 0.0) for step in physics_step])
    return {
        "time_s": time_s,
        "speed": speed,
        "force": force,
        "maximum_penetration_m": np.asarray(maximum_penetration_m),
        "links": np.asarray(links),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--bounded", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = load(args.original)
    bounded = load(args.bounded)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})
    figure, axes = plt.subplots(2, 1, figsize=(11.2, 7.2), sharex=False)
    cases = [
        (
            axes[0],
            original,
            "Original bin: 190 mm; confirmed right-elbow rim collision",
            "#b2182b",
        ),
        (
            axes[1],
            bounded,
            "Bounded fallback: 100 mm + 3 mm symmetric rim bevel; still invalid",
            "#2166ac",
        ),
    ]
    for axis, data, title, color in cases:
        axis.plot(data["time_s"], data["speed"], color=color, lw=1.1, label="Doll speed")
        contact = data["force"] > 0.0
        if np.any(contact):
            axis.fill_between(
                data["time_s"],
                0.0,
                1.0,
                where=contact,
                transform=axis.get_xaxis_transform(),
                color="#fdae61",
                alpha=0.24,
                label="Robot–bin contact",
            )
        peak = int(np.argmax(data["speed"]))
        axis.scatter(
            [data["time_s"][peak]], [data["speed"][peak]], color=color, s=32, zorder=4
        )
        axis.annotate(
            f"peak {data['speed'][peak]:.3f} m/s",
            (data["time_s"][peak], data["speed"][peak]),
            xytext=(8, 8),
            textcoords="offset points",
        )
        axis.axhline(1.0, color="black", lw=0.9, ls="--", label="1.0 m/s gate")
        axis.set_title(title, loc="left", weight="bold")
        axis.set_ylabel("Doll speed (m/s)")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper left", ncol=3, frameon=False)
        link_text = ", ".join(str(value) for value in data["links"])
        axis.text(
            0.995,
            0.96,
            f"contact links: {link_text}\nmax penetration: "
            f"{float(data['maximum_penetration_m']) * 1e3:.3f} mm",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8},
        )
    axes[1].set_xlabel("Simulation time (s)")
    figure.suptitle(
        "Bin collision audit: speed improved, but the legal lower-bound geometry still snags",
        fontsize=14,
        weight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
