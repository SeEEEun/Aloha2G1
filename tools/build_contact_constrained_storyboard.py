#!/usr/bin/env python3
"""Compose the measured-state scripted-physics keyframes into a storyboard."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/home/jbnu/aloha_g1_dataset")
FIGURES = ROOT / "outputs/final_contact_constrained_eval/06_paper_figures"
SOURCE = FIGURES / "scripted_physical_keyframes"
OUTPUT = FIGURES / "PHYSICAL_TASK_STORYBOARD.png"
CONTACT = FIGURES / "SCRIPTED_PHYSICAL_CONTACT_SHEET.png"
MANIFEST = FIGURES / "PHYSICAL_TASK_STORYBOARD_MANIFEST.json"
EVENT = ROOT / "outputs/final_contact_constrained_eval/02_scripted_validation/run_01/event_log.npz"

PANELS = [
    ("01_left_grasp.png", "1  LEFT GRASP", 80),
    ("02_left_lift.png", "2  LEFT LIFT", 160),
    ("03_handoff.png", "3  HANDOFF APPROACH", 400),
    ("04_right_acquisition.png", "4  RIGHT ACQUISITION", 800),
    ("05_left_release.png", "5  LEFT RELEASE", 1648),
    ("06_right_ownership.png", "6  RIGHT OWNERSHIP", 1744),
    ("07_right_transport.png", "7  RIGHT TRANSPORT", 2256),
    ("08_bin_approach.png", "8  BIN APPROACH", 2512),
    ("09_release.png", "9  RELEASE", 2800),
    ("10_doll_settled.png", "10  DOLL SETTLED", 3264),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def main() -> None:
    images = []
    for filename, label, frame in PANELS:
        path = SOURCE / filename
        if not path.is_file():
            raise RuntimeError(f"missing captured physical keyframe: {path}")
        image = Image.open(path).convert("RGB")
        images.append((path, image, label, frame))

    tile_w, tile_h = 640, 360
    caption_h, title_h, footer_h = 54, 92, 60
    canvas = Image.new("RGB", (tile_w * 5, title_h + 2 * (tile_h + caption_h) + footer_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (canvas.width // 2, 20),
        "COMMON SCRIPTED PHYSICAL VALIDATION",
        anchor="ma",
        font=font(34, True),
        fill="#202124",
    )
    draw.text(
        (canvas.width // 2, 60),
        "Frozen 150 mm-bin contact-constrained PhysX trace — measured robot and doll states",
        anchor="ma",
        font=font(20),
        fill="#555555",
    )
    for index, (_path, image, label, frame) in enumerate(images):
        row, col = divmod(index, 5)
        x = col * tile_w
        y = title_h + row * (tile_h + caption_h)
        image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        canvas.paste(image, (x + (tile_w - image.width) // 2, y + (tile_h - image.height) // 2))
        draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline="#D0D0D0", width=2)
        draw.text((x + 16, y + tile_h + 8), label, font=font(19, True), fill="#222222")
        draw.text((x + tile_w - 16, y + tile_h + 8), f"frame {frame}", anchor="ra", font=font(17), fill="#666666")
    draw.text(
        (canvas.width // 2, canvas.height - 24),
        "Environment-validity evidence only; this storyboard is not an ACT-A/B rollout.",
        anchor="mm",
        font=font(18),
        fill="#7A2E2E",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, optimize=True)
    canvas.save(CONTACT, optimize=True)

    manifest = {
        "schema_version": "contact_constrained_scripted_storyboard_v1",
        "label": "COMMON SCRIPTED PHYSICAL VALIDATION",
        "not_act_policy_output": True,
        "event_log": str(EVENT),
        "event_log_sha256": sha256(EVENT),
        "source_trace_execution": "CONTACT_CONSTRAINED_PHYSICS",
        "rendering": "read-only measured-state viewport capture; no new physics outcome",
        "panels": [
            {"path": str(path), "sha256": sha256(path), "label": label, "control_frame": frame}
            for path, _image, label, frame in images
        ],
        "outputs": [str(OUTPUT), str(CONTACT)],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
