# ALOHA / G1 Doll-Handoff Isaac Lab Scene Package

이 패키지는 기존 `isaaclab_magsafe_fixed_scene`을 **복사**한 뒤, 그 복사본에 새 태스크 물체를 넣습니다. 기존 MagSafe 환경은 건드리지 않습니다.

## 이번 배치

- 기존 테이블/검정 프레임/카메라/조명 유지
- 기존 테이블 크기: **0.835 × 0.720 m**
- 검정 frame 안쪽 기준 왼쪽 하단: 초록색 공 모양 인형
- 검정 frame 안쪽 기준 오른쪽 하단: 열린 사각 쓰레기통
- G1: 기존 20 cm 계열 root/forward offset이 현재 preview code에서 고유하게 식별되면 **15 cm**로 자동 patch
- ALOHA 위치는 기존 환경 그대로

## 모델 치수

다이소 상품 페이지는 자동 접근 시 403으로 막혀 있어서, 현재 모델은 사용자가 보낸 사진 + 기존 0.835 m 테이블 scale을 기준으로 한 provisional 모델입니다.

- 인형/공: 지름 75 mm, 55 g
- 쓰레기통: 상단 190×165 mm, 하단 160×135 mm, 높이 190 mm
- 쓰레기통은 실제로 물체가 들어갈 수 있게 **윗면 collider가 없는 open container**입니다.

실물 줄자로 치수를 재면 `scene_layout_doll_handoff.json` 숫자만 바꿔 재빌드할 수 있습니다.

## 설치

압축을 푼 뒤:

```bash
cd isaaclab_doll_handoff_scene_package
bash install.sh /home/jbnu/aloha_g1_dataset
```

생성 위치:

```bash
/home/jbnu/aloha_g1_dataset/isaaclab_doll_handoff_scene
```

## ALOHA에서 보기

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6
cd /home/jbnu/aloha_g1_dataset/isaaclab_doll_handoff_scene

~/IsaacLab-3-beta/isaaclab.sh -p   preview_doll_handoff_aloha.py   --pose episode_frame0   --viz kit   --camera overview
```

## G1에서 보기

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6
cd /home/jbnu/aloha_g1_dataset/isaaclab_doll_handoff_scene

~/IsaacLab-3-beta/isaaclab.sh -p   preview_doll_handoff_g1.py   --viz kit   --camera overview
```

## 배치 좌표

기존 scene 좌표계는 table front-left가 `(0,0)`이고 +X가 오른쪽, +Y가 테이블 안쪽입니다.
검정 rail 폭 25 mm를 제외한 안쪽 왼쪽 아래를 task origin으로 사용합니다.

- task origin world = `(0.025, 0.025, 0.795)`
- doll center world XY = `(0.165, 0.090)`
- bin center world XY = `(0.675, 0.155)`
- G1 pelvis/table-front requested gap = `0.150 m`

## G1 15 cm patch가 실패하면

기존 G1 preview 소스의 root 위치 표현을 제가 직접 볼 수 없는 상태라, installer는 **0.20 m root/forward literal을 문맥으로 식별할 때만** 0.15 m로 바꿉니다. 애매하면 절대 임의 수정하지 않고:

```bash
g1_pelvis_gap_patch_report.json
```

에 후보 줄을 남깁니다. 이 경우 그 report와 preview 한 장만 주면 다음 수정은 바로 특정할 수 있습니다.

## 주의

이건 현재 **scene/modeling + visual review**용입니다. 50-episode A/B retargeting이나 Policy A/B 학습은 하지 않습니다.
