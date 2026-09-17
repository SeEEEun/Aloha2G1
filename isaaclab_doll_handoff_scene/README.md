# Doll-Handoff Isaac Lab Scene v2

This directory is a clean Doll-Handoff scene. Static previews never read a demonstration. The table and black workspace rails are geometry-checked copies of the protected, authoritative table asset; legacy task metadata is removed from the local copy.

Build and perform static validation from anywhere:

```bash
cd /home/jbnu/aloha_g1_dataset
./isaaclab_doll_handoff_scene/build_scene.sh
```

Run the PhysX sanity/drop test:

```bash
cd /home/jbnu/aloha_g1_dataset
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6
~/IsaacLab-3-beta/isaaclab.sh -p \
  isaaclab_doll_handoff_scene/validate_physics.py \
  --viz none
```

Static ALOHA preview:

```bash
cd /home/jbnu/aloha_g1_dataset/isaaclab_doll_handoff_scene
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6
~/IsaacLab-3-beta/isaaclab.sh -p \
  preview_doll_handoff_aloha.py \
  --viz kit \
  --camera overview
```

Static G1 preview:

```bash
cd /home/jbnu/aloha_g1_dataset/isaaclab_doll_handoff_scene
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6
~/IsaacLab-3-beta/isaaclab.sh -p \
  preview_doll_handoff_g1.py \
  --viz kit \
  --camera overview
```

Both previews also accept `--camera top`. The prior oblique overview is retained as
`--camera legacy_overview`; `overview` is the front-high task-review view that keeps
the Doll on the left and the bin on the right. To save a frame, add
`--screenshot /absolute/output.png`.

The Doll and trash-bin dimensions in `scene_layout.json` are explicitly **PROVISIONAL**. No retargeting or policy-training workflow is included here.
