# Isaac Lab MagSafe Fixed Scene

로봇 없이 **테이블 + 세워진 스마트폰 + 다이소 맥세이프 팝핑거 + 맥세이프 충전 스탠드**만 고정 배치하는 Isaac Lab용 장면입니다.

## 좌표계

이 패키지는 사용자가 준 평면 좌표를 다음처럼 해석합니다.

- 월드 원점: 테이블 상판 **앞-왼쪽 모서리 바로 아래 바닥점**
- 테이블 상판 원점: `(0, 0, 0.795)` m
- `+X`: 테이블 왼쪽 → 오른쪽, 전체 0.835 m
- `+Y`: 작업자/로봇 쪽 앞면 → 충전기 쪽 뒷면, 전체 0.720 m
- `+Z`: 위쪽

즉 사용자가 말한 테이블의 왼쪽 아래점 `(0, 0)`은 위에서 보았을 때 **앞-왼쪽 상판 모서리**입니다.

## 고정 배치

- 스마트폰 하단 모서리 측정점: `(0.450, 0.255)` m → `(0.600, 0.255)` m
- 스마트폰 실제 크기: `149.6 × 7.95 × 71.5 mm`로 landscape/upright 배치
- 스마트폰 화면: `-Y` 방향, 뒷면과 팝핑거: `+Y` 방향
- 충전기 베이스 중심: `(0.420, 0.520)` m
- 충전기: 105 × 105 mm 베이스, 24 mm 베이스 높이, 총 160 mm 높이, 59 mm 패드
- 팝핑거: 다이소 품번 `1062886`, 메인 링 외경 55 mm/내경 45 mm

## 생성되는 파일

`generated/` 아래에 모듈형 USDA를 생성합니다.

- `table_optical.usda`
- `phone_landscape.usda`
- `magsafe_poppinger_1062886.usda`
- `charger_stand.usda`
- `magsafe_fixed_scene.usda` — 위 자산을 고정 배치한 composite scene

테이블/충전기는 나중에도 고정 환경으로 사용할 수 있고, 스마트폰과 액세서리는 별도 USD라 이후 Isaac Lab `RigidObjectCfg` 대상으로 분리해 사용할 수 있습니다.

## 실행

Isaac Lab 저장소 루트에서 절대경로로 실행합니다.

```bash
cd ~/IsaacLab

./isaaclab.sh -p \
  /home/jbnu/isaaclab_magsafe_fixed_scene/preview_magsafe_scene.py \
  --rebuild
```

패키지를 다른 위치에 풀었다면 스크립트 절대경로만 바꾸면 됩니다.

USD만 생성하고 GUI를 열지 않으려면:

```bash
cd ~/IsaacLab

./isaaclab.sh -p \
  /home/jbnu/isaaclab_magsafe_fixed_scene/preview_magsafe_scene.py \
  --rebuild \
  --export-only \
  --headless
```

다시 볼 때는 `--rebuild`를 빼면 됩니다.

```bash
./isaaclab.sh -p \
  /home/jbnu/isaaclab_magsafe_fixed_scene/preview_magsafe_scene.py
```

## 다른 Isaac Lab 코드에서 장면 불러오기

```python
from pathlib import Path
import isaaclab.sim as sim_utils

scene_usd = Path("/home/jbnu/isaaclab_magsafe_fixed_scene/generated/magsafe_fixed_scene.usda")
cfg = sim_utils.UsdFileCfg(usd_path=str(scene_usd))
cfg.func("/World/MagSafeScene", cfg)
```

로봇은 나중에 `/World/Robot` 또는 `/World/envs/env_0/Robot`에 별도로 spawn하면 됩니다.

## RealityScan 자산으로 교체

현재 parametric visual은 단순 box/cylinder가 아니라 다음을 포함합니다.

- 광학 테이블 홀 패턴과 검은 프레임
- 둥근 모서리 스마트폰, 화면, 후면 카메라
- C형 메인 링 + 펼쳐진 지지 링
- 원형 베이스, 이중 지지대, 기울어진 59 mm 충전 패드, LED

RealityScan으로 만든 실제 visual USD가 준비되면 `scene_layout.json`의 다음 항목에 절대경로를 넣습니다.

```json
"asset_overrides": {
  "phone_usd": "/absolute/path/phone_visual.usd",
  "accessory_usd": "/absolute/path/accessory_visual.usd",
  "charger_usd": "/absolute/path/charger_visual.usd"
}
```

그 뒤 다시 `--rebuild`를 실행합니다. 고정 좌표와 collision proxy는 유지되고 visual만 추가됩니다. 스캔 visual의 원점·축·스케일은 Blender에서 정리한 뒤 사용하는 것이 안전합니다.

## 현재 추정값

사진만으로 확정할 수 없어 설정 파일로 분리해 둔 값입니다.

- 테이블 상판 두께, 검은 프레임 폭, 다리 규격, 홀 pitch
- 팝핑거 두께와 작은 지지 링 치수
- 충전 패드의 15° 기울기
- 사진의 검은 직사각형 받침판 크기

실물 측정값이 생기면 `scene_layout.json`만 바꾸고 `--rebuild`하면 됩니다.

## 다음 단계

장면 배치가 화면에서 맞는지 확인한 뒤:

1. G1을 별도 spawn
2. `g1_dynamic_bimanual_full_trajectory.npz` 재생기 연결
3. phone/accessory를 static에서 kinematic 또는 dynamic rigid object로 분리
4. grasp/attach/detach keyframe 정의
5. imitation/RL 환경용 reset 및 randomization 추가

첫 디버깅 버전은 요청대로 좌표를 고정하며, randomization은 넣지 않았습니다.
