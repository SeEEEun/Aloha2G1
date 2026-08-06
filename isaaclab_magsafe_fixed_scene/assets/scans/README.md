# RealityScan / 외부 Mesh 교체 슬롯

RealityScan 결과 OBJ/GLB 또는 외부 FBX를 Isaac Sim Asset Converter로 USD로 바꾼 뒤 아래처럼 보관하는 것을 권장합니다.

```text
assets/scans/
  phone/phone_visual.usd
  accessory/accessory_visual.usd
  charger/charger_visual.usd
```

Blender 전처리 기준:

- 단위: meter
- origin: 물체 중심 또는 README에 정의된 기준 frame
- +Z: 위
- phone: landscape 상태에서 +X가 긴 방향, +Y가 뒷면 방향
- 불필요한 배경 mesh 제거
- visual mesh와 collision mesh 분리
- texture 경로를 USD와 함께 보존

그 후 `scene_layout.json`의 `asset_overrides`에 절대경로를 기록합니다.
