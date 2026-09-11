

|                         |     |
| ----------------------- | --- |
| [[#### App 設定對照表（dev）]] |     |
|                         |     |
|                         |     |
|                         |     |
|                         |     |

#### App 設定對照表（dev）
```
# App 設定對照表（dev）
```

分析基準

- Repository：`D:\Provenance Laboratories projects\ImagingLibWatch`
- Branch / commit：`dev` / `0585ad185a86`
- App 入口：`App/main.py`（SHA-256 前 12 碼 `b1dadd07808f`）
- Internal number registry：`config/internalnum_config.yaml`（schema `4`；SHA-256 前 12 碼 `8178e5f1f42e`）
- System routing：`config/system_config.yaml`（SHA-256 前 12 碼 `325450dc0f4f`）
- 注意：`system_config.yaml` 被 `.gitignore` 排除，不屬於 dev commit；本報告採用目前工作區實際檔案。`App/main.py` 與 `internalnum_config.yaml` 在目前工作區相對 HEAD 無修改。
- Join 規則：`services.<service>.internalnums[].(internalnum1, internalnum2)` 對應 registry capture pair；`server` 取 `services.<service>.script` 的檔名；`service_name` 取 `tasks.<service>.service_name`。
- Point-only registry（7xxx material、8xxx external measurements）沒有 capture-level `internalnum2`，表中以 `—` 表示，並歸類為無 service route。

## 結果摘要

| 項目 | 數量 |
|---|---:|
| 非 backup registry 項目 | 435 |
| 有 service 對應的 registry 項目（unique pair） | 24 |
| service route 數（含一個 pair 對多 service） | 31 |
| internalnum_config 有、system_config 無 service | 411 |
| system_config 有 service、但無 internalnum route | 9 |
| 一個 internalnum pair 對多 service | 7 |
| system_config 指到不存在的 internalnum pair | 0 |
| 已排除 backup 項目 | 32 |

## 一對多 service 對應

| internalnum1 | internalnum2 | displayname | server | service_name |
|---|---|---|---|---|
| 0004 | 0001 | Upper text | server_front_stitch.py; server_ocr.py | front_stitch_service; ocr_service |
| 0005 | 0001 | Lower text | server_front_stitch.py; server_ocr.py | front_stitch_service; ocr_service |
| 0006 | 0001 | Crown | server_front_stitch.py; server_micro_analysis.py | front_stitch_service; side_crown_service |
| 0022 | 0001 | Lume of the 1 hour marker | server_micro_analysis.py | diallume_shape_service; diallume_texture_service |
| 0023 | 0001 | Hour hand lume - near lower text for mercedes hand | server_micro_analysis.py | lume_hour_shape_service; lume_hour_texture_service |
| 0024 | 0001 | Minute hand lume - at end futher from center | server_micro_analysis.py | lume_hand_shape_service; lume_hand_texture_service |
| 0025 | 0001 | Second hand lume | server_micro_analysis.py | lume_hand_shape_service; lume_hand_texture_service |

## system_config 有 service，但無 internalnum 對應

| service_name | server | 設定位置 | 原因 |
|---|---|---|---|
| bracelet_service | server_bracelet.py | tasks + services | internalnums 為空 |
| crown_service | — | tasks only | services 區塊無同名 server/route |
| doctr_service | server_doctr.py | tasks + services | internalnums 為空 |
| links_service | server_links.py | tasks + services | internalnums 為空 |
| lume_service | — | tasks only | services 區塊無同名 server/route |
| movement1_service | server_movement1.py | tasks + services | internalnums 為空 |
| movement2_service | server_movement2.py | tasks + services | internalnums 為空 |
| openbacktop_service | server_openbacktop.py | tasks + services | internalnums 為空 |
| stitched_band2_service | — | tasks only | services 區塊無同名 server/route |

## system_config 指到不存在的 internalnum pair

| service_name | server | target | 原因 |
|---|---|---|---|
| — | — | — | 無（31 條 route 全部可解析） |

## 完整非 backup 對照表

| internalnum1 | internalnum2 | displayname | server | service_name | 狀態 |
|---|---|---|---|---|---|
| 0001 | 0001 | Top view | — | — | internalnum_config only |
| 0002 | 0001 | Top Side | server_sidepoint.py | sidepoint_service | 已對應 |
| 0003 | 0001 | Bottom Side | server_sidepoint.py | sidepoint_service | 已對應 |
| 0004 | 0001 | Upper text | server_front_stitch.py; server_ocr.py | front_stitch_service; ocr_service | 一對多：2 services |
| 0005 | 0001 | Lower text | server_front_stitch.py; server_ocr.py | front_stitch_service; ocr_service | 一對多：2 services |
| 0006 | 0001 | Crown | server_front_stitch.py; server_micro_analysis.py | front_stitch_service; side_crown_service | 一對多：2 services |
| 0006 | 0002 | Crown | — | — | internalnum_config only |
| 0007 | 0001 | Bottom right lug | server_front_stitch.py | front_stitch_service | 已對應 |
| 0008 | 0001 | Dial Left | server_front_stitch.py | front_stitch_service | 已對應 |
| 0008 | 0002 | Dial Left | — | — | internalnum_config only |
| 0009 | 0001 | Dial Right | — | — | internalnum_config only |
| 0009 | 0002 | Dial Right | — | — | internalnum_config only |
| 0010 | 0001 | Dial Lower | — | — | internalnum_config only |
| 0010 | 0002 | Dial Lower | — | — | internalnum_config only |
| 0011 | 0001 | Y in Officially | server_micro_analysis.py | features_letter_service | 已對應 |
| 0011 | 0002 | Y in Officially hdr image | — | — | internalnum_config only |
| 0012 | 0001 | Logo on Dial | server_micro_analysis.py | features_crown_service | 已對應 |
| 0012 | 0002 | Inside of crown logo bottom football shape hdr image | — | — | internalnum_config only |
| 0013 | 0001 | M in Made at near 6 | server_micro_analysis.py | features_letter_service | 已對應 |
| 0013 | 0002 | M in Made at near 6 hdr image | — | — | internalnum_config only |
| 0014 | 0001 | 60 minute marker | server_micro_analysis.py | features_marker_service | 已對應 |
| 0014 | 0002 | 60 minute marker hdr image | — | — | internalnum_config only |
| 0015 | 0001 | 1 minute marker | server_micro_analysis.py | features_marker_service | 已對應 |
| 0015 | 0002 | 1 minute marker hdr image | — | — | internalnum_config only |
| 0016 | 0001 | Bezel at 30 (middle of 3 in 30) | server_micro_analysis.py | bezel_marker_service | 已對應 |
| 0016 | 0002 | Bezel at 30 (middle of 3 in 30) hdr image | — | — | internalnum_config only |
| 0017 | 0001 | Bezel at 45 mark (centered) | server_micro_analysis.py | bezel_marker_service | 已對應 |
| 0017 | 0002 | Bezel at 45 mark (centered) hdr image | — | — | internalnum_config only |
| 0018 | 0001 | Bezel at 12  (middle of 2 in 12) | — | — | internalnum_config only |
| 0019 | 0001 | Bezel at 18 (middle of 8 in 18) | — | — | internalnum_config only |
| 0020 | 0001 | Bezel at 120  (middle of 2 in 120) | — | — | internalnum_config only |
| 0021 | 0001 | Bezel at 240 (middle of 4 in 240) | — | — | internalnum_config only |
| 0022 | 0001 | Lume of the 1 hour marker | server_micro_analysis.py | diallume_shape_service; diallume_texture_service | 一對多：2 services |
| 0022 | 0002 | Lume of 1 marker hdr image | — | — | internalnum_config only |
| 0023 | 0001 | Hour hand lume - near lower text for mercedes hand | server_micro_analysis.py | lume_hour_shape_service; lume_hour_texture_service | 一對多：2 services |
| 0023 | 0002 | Hour hand lume - near lower text for mercedes hand hdr image | — | — | internalnum_config only |
| 0024 | 0001 | Minute hand lume - at end futher from center | server_micro_analysis.py | lume_hand_shape_service; lume_hand_texture_service | 一對多：2 services |
| 0024 | 0002 | Minute hand lume - at end futher from center hdr image | — | — | internalnum_config only |
| 0025 | 0001 | Second hand lume | server_micro_analysis.py | lume_hand_shape_service; lume_hand_texture_service | 一對多：2 services |
| 0025 | 0002 | Second hand lume hdr image | — | — | internalnum_config only |
| 0026 | 0001 | Dial area with no text just inside of 1 marker | server_micro_analysis.py | texture_service | 已對應 |
| 0026 | 0002 | Dial area with no text just inside of 1 marker hdr image | — | — | internalnum_config only |
| 0027 | 0001 | bottom subdial: 20: 0 in 20 | server_micro_analysis.py | bezel_marker_service | 已對應 |
| 0028 | 0001 | Hour to dial height | — | — | internalnum_config only |
| 0029 | 0001 | minute to dial height | — | — | internalnum_config only |
| 0030 | 0001 | 1 hour marker lume to dial height | — | — | internalnum_config only |
| 0031 | 0001 | thickness of the glass | — | — | internalnum_config only |
| 0032 | 0001 | Crown on sapphire crystal | server_crown2.py | crown2_service | 已對應 |
| 0033 | 0001 | Side refer num E in ROLEX | — | — | internalnum_config only |
| 0034 | 0001 | Side refer num R in ROLEX | — | — | internalnum_config only |
| 0035 | 0001 | Side serial num E in STAINLESS | — | — | internalnum_config only |
| 0036 | 0001 | Side serial num S in STEEL | — | — | internalnum_config only |
| 0037 | 0001 | Side crown center | — | — | internalnum_config only |
| 0038 | 0001 | Side crown 3 star | — | — | internalnum_config only |
| 0039 | 0001 | Rehaut at 12 o'clock | — | — | internalnum_config only |
| 0040 | 0001 | Rehaut at 1 o'clock | — | — | internalnum_config only |
| 0041 | 0001 | Rehaut at 2 o'clock | — | — | internalnum_config only |
| 0042 | 0001 | Rehaut at 3 o'clock | — | — | internalnum_config only |
| 0043 | 0001 | Rehaut at 4 o'clock | — | — | internalnum_config only |
| 0044 | 0001 | Rehaut at 5 o'clock | — | — | internalnum_config only |
| 0045 | 0001 | Rehaut at 6 o'clock | — | — | internalnum_config only |
| 0046 | 0001 | Rehaut at 7 o'clock | — | — | internalnum_config only |
| 0047 | 0001 | Rehaut at 8 o'clock | — | — | internalnum_config only |
| 0048 | 0001 | Rehaut at 9 o'clock | — | — | internalnum_config only |
| 0049 | 0001 | Rehaut at 10 o'clock | — | — | internalnum_config only |
| 0050 | 0001 | Rehaut at 11 o'clock | — | — | internalnum_config only |
| 0055 | 0001 | Hour hand lumeless - near lower text for mercedes hand | — | — | internalnum_config only |
| 0056 | 0001 | Minute hand lumeless - at end futher from center | — | — | internalnum_config only |
| 0057 | 0001 | Second hand lumeless | — | — | internalnum_config only |
| 0058 | 0001 | fluted bezel field | — | — | internalnum_config only |
| 0058 | 0002 | fluted bezel field hdr image | — | — | internalnum_config only |
| 1001 | 0001 | Top view | — | — | internalnum_config only |
| 1002 | 0001 | Bottom Left Lug (top left in the image) | — | — | internalnum_config only |
| 1003 | 0001 | Caseback near top of main circle | — | — | internalnum_config only |
| 1004 | 0001 | Caseback Center | — | — | internalnum_config only |
| 2001 | 0001 | Top view | — | — | internalnum_config only |
| 2002 | 0001 | Text below rotor that is exposed | — | — | internalnum_config only |
| 2003 | 0001 | Serial number | — | — | internalnum_config only |
| 2004 | 0001 | Ratchet wheel | — | — | internalnum_config only |
| 2005 | 0001 | A letter or number in the serial | server_isolation.py | isolation_service | 已對應 |
| 3001 | 0001 | Top view | — | — | internalnum_config only |
| 3002 | 0001 | Movement caliber | — | — | internalnum_config only |
| 3003 | 0001 | Balance Wheel and Bridge | — | — | internalnum_config only |
| 3003 | 0002 | Balance Wheel and Bridge -1 | — | — | internalnum_config only |
| 3004 | 0001 | Text below rotor that is exposed | — | — | internalnum_config only |
| 3005 | 0001 | Movement serial number | — | — | internalnum_config only |
| 3006 | 0001 | Rotor | — | — | internalnum_config only |
| 3006 | 0002 | Rotor | — | — | internalnum_config only |
| 3007 | 0001 | First number in movement caliber | — | — | internalnum_config only |
| 3008 | 0001 | Bridge | — | — | internalnum_config only |
| 3009 | 0001 | Rachet wheel (Big yellow gear) | server_isolation.py | isolation_service | 已對應 |
| 3010 | 0001 | A letter or number in the serial | server_isolation.py | isolation_service | 已對應 |
| 4001 | 0001 | underside link1 | — | — | internalnum_config only |
| 4002 | 0001 | underside link2 | — | — | internalnum_config only |
| 4003 | 0001 | underside link3 | — | — | internalnum_config only |
| 4004 | 0001 | underside link4 | — | — | internalnum_config only |
| 4005 | 0001 | underside link5 | — | — | internalnum_config only |
| 4006 | 0001 | underside link6 | — | — | internalnum_config only |
| 4007 | 0001 | underside link7 | — | — | internalnum_config only |
| 4008 | 0001 | 3clock side link1 | — | — | internalnum_config only |
| 4009 | 0001 | 3clock side link2 | — | — | internalnum_config only |
| 4010 | 0001 | 3clock side link3 | — | — | internalnum_config only |
| 4011 | 0001 | 3clock side link4 | — | — | internalnum_config only |
| 4012 | 0001 | 3clock side link5 | — | — | internalnum_config only |
| 4013 | 0001 | 3clock side link6 | — | — | internalnum_config only |
| 4014 | 0001 | 3clock side link7 | — | — | internalnum_config only |
| 4015 | 0001 | Outer surface side1 | — | — | internalnum_config only |
| 4016 | 0001 | Outer surface side2 | — | — | internalnum_config only |
| 4017 | 0001 | Outer surface side3 | — | — | internalnum_config only |
| 4018 | 0001 | Outer surface side4 | — | — | internalnum_config only |
| 4019 | 0001 | Outer surface side5 | — | — | internalnum_config only |
| 4020 | 0001 | Outer surface side6 | — | — | internalnum_config only |
| 4021 | 0001 | Outer surface side7 | — | — | internalnum_config only |
| 4022 | 0001 | 9clock side link1 | — | — | internalnum_config only |
| 4023 | 0001 | 9clock side link2 | — | — | internalnum_config only |
| 4024 | 0001 | 9clock side link3 | — | — | internalnum_config only |
| 4025 | 0001 | 9clock side link4 | — | — | internalnum_config only |
| 4026 | 0001 | 9clock side link5 | — | — | internalnum_config only |
| 4027 | 0001 | 9clock side link6 | — | — | internalnum_config only |
| 4028 | 0001 | 9clock side link7 | — | — | internalnum_config only |
| 4029 | 0001 | 12 clock end link code | — | — | internalnum_config only |
| 4030 | 0001 | 6 clock end link code | — | — | internalnum_config only |
| 4034 | 0001 | Fast macro_cam_1 front tile 4034 | — | — | internalnum_config only |
| 4035 | 0001 | Fast macro_cam_1 front tile 4035 | — | — | internalnum_config only |
| 4036 | 0001 | Fast macro_cam_1 front tile 4036 | — | — | internalnum_config only |
| 4037 | 0001 | Fast macro_cam_1 front tile 4037 | — | — | internalnum_config only |
| 4038 | 0001 | Fast macro_cam_1 front tile 4038 | — | — | internalnum_config only |
| 4039 | 0001 | Fast macro_cam_1 front tile 4039 | — | — | internalnum_config only |
| 4040 | 0001 | Fast macro_cam_1 front tile 4040 | — | — | internalnum_config only |
| 4041 | 0001 | Fast macro_cam_1 front tile 4041 | — | — | internalnum_config only |
| 4042 | 0001 | Fast macro_cam_1 front tile 4042 | — | — | internalnum_config only |
| 4043 | 0001 | Fast macro_cam_1 front tile 4043 | — | — | internalnum_config only |
| 4044 | 0001 | Fast macro_cam_1 front tile 4044 | — | — | internalnum_config only |
| 4045 | 0001 | Fast macro_cam_1 side tile 4045 | — | — | internalnum_config only |
| 4046 | 0001 | Fast macro_cam_1 side tile 4046 | — | — | internalnum_config only |
| 4047 | 0001 | Fast macro_cam_1 side tile 4047 | — | — | internalnum_config only |
| 4048 | 0001 | Fast macro_cam_1 side tile 4048 | — | — | internalnum_config only |
| 4049 | 0001 | Fast macro_cam_1 side tile 4049 | — | — | internalnum_config only |
| 4050 | 0001 | Fast macro_cam_1 side tile 4050 | — | — | internalnum_config only |
| 4051 | 0001 | Fast macro_cam_1 side tile 4051 | — | — | internalnum_config only |
| 4052 | 0001 | Fast macro_cam_1 side tile 4052 | — | — | internalnum_config only |
| 4053 | 0001 | Fast macro_cam_1 side tile 4053 | — | — | internalnum_config only |
| 4054 | 0001 | Fast macro_cam_1 side tile 4054 | — | — | internalnum_config only |
| 4055 | 0001 | Fast macro_cam_1 side tile 4055 | — | — | internalnum_config only |
| 4056 | 0001 | Fast macro_cam_1 back tile 4056 | — | — | internalnum_config only |
| 4057 | 0001 | Fast macro_cam_1 back tile 4057 | — | — | internalnum_config only |
| 4058 | 0001 | Fast macro_cam_1 back tile 4058 | — | — | internalnum_config only |
| 4059 | 0001 | Fast macro_cam_1 back tile 4059 | — | — | internalnum_config only |
| 4060 | 0001 | Fast macro_cam_1 back tile 4060 | — | — | internalnum_config only |
| 4061 | 0001 | Fast macro_cam_1 back tile 4061 | — | — | internalnum_config only |
| 4062 | 0001 | Fast macro_cam_1 back tile 4062 | — | — | internalnum_config only |
| 4063 | 0001 | Fast macro_cam_1 back tile 4063 | — | — | internalnum_config only |
| 4064 | 0001 | Fast macro_cam_1 back tile 4064 | — | — | internalnum_config only |
| 4065 | 0001 | Fast macro_cam_1 back tile 4065 | — | — | internalnum_config only |
| 4066 | 0001 | Fast macro_cam_1 back tile 4066 | — | — | internalnum_config only |
| 4067 | 0001 | Fast macro_cam_1 9-clock tile 4067 | — | — | internalnum_config only |
| 4068 | 0001 | Fast macro_cam_1 9-clock tile 4068 | — | — | internalnum_config only |
| 4069 | 0001 | Fast macro_cam_1 9-clock tile 4069 | — | — | internalnum_config only |
| 4070 | 0001 | Fast macro_cam_1 9-clock tile 4070 | — | — | internalnum_config only |
| 4071 | 0001 | Fast macro_cam_1 9-clock tile 4071 | — | — | internalnum_config only |
| 4072 | 0001 | Fast macro_cam_1 9-clock tile 4072 | — | — | internalnum_config only |
| 4073 | 0001 | Fast macro_cam_1 9-clock tile 4073 | — | — | internalnum_config only |
| 4074 | 0001 | Fast macro_cam_1 9-clock tile 4074 | — | — | internalnum_config only |
| 4075 | 0001 | Fast macro_cam_1 9-clock tile 4075 | — | — | internalnum_config only |
| 4076 | 0001 | Fast macro_cam_1 9-clock tile 4076 | — | — | internalnum_config only |
| 4077 | 0001 | Fast macro_cam_1 9-clock tile 4077 | — | — | internalnum_config only |
| 4101 | 0001 | End link of 12 clock Bracelet section - underside | — | — | internalnum_config only |
| 4102 | 0001 | 12 clock Bracelet section link 1 - underside | — | — | internalnum_config only |
| 4103 | 0001 | 12 clock Bracelet section link 2 - underside | — | — | internalnum_config only |
| 4104 | 0001 | 12 clock Bracelet section link 3 - underside | — | — | internalnum_config only |
| 4105 | 0001 | 12 clock Bracelet section link 4 - underside | — | — | internalnum_config only |
| 4106 | 0001 | 12 clock Bracelet section link 5 - underside | — | — | internalnum_config only |
| 4107 | 0001 | 12 clock Bracelet section link 6 - underside | — | — | internalnum_config only |
| 4108 | 0001 | 12 clock Bracelet section link 7 - underside | — | — | internalnum_config only |
| 4109 | 0001 | 12 clock Bracelet section link 8 - underside | — | — | internalnum_config only |
| 4110 | 0001 | 12 clock Bracelet section link 9 - underside | — | — | internalnum_config only |
| 4111 | 0001 | 12 clock Bracelet section link 10 - underside | — | — | internalnum_config only |
| 4112 | 0001 | 12 clock Bracelet section link 11 - underside | — | — | internalnum_config only |
| 4113 | 0001 | 12 clock Bracelet section link 12 - underside | — | — | internalnum_config only |
| 4114 | 0001 | 12 clock Bracelet section link 13 - underside | — | — | internalnum_config only |
| 4115 | 0001 | 12 clock Bracelet section link 14 - underside | — | — | internalnum_config only |
| 4116 | 0001 | 12 clock Bracelet section link 15 - underside | — | — | internalnum_config only |
| 4117 | 0001 | 12 clock Bracelet section link 16 - underside | — | — | internalnum_config only |
| 4118 | 0001 | 12 clock Bracelet section link 17 - underside | — | — | internalnum_config only |
| 4119 | 0001 | 12 clock Bracelet section link 18 - underside | — | — | internalnum_config only |
| 4120 | 0001 | 12 clock Bracelet section link 19 - underside | — | — | internalnum_config only |
| 4201 | 0001 | End link of 6 clock Bracelet section - underside | — | — | internalnum_config only |
| 4202 | 0001 | 6 clock Bracelet section link 1 - underside | — | — | internalnum_config only |
| 4203 | 0001 | 6 clock Bracelet section link 2 - underside | — | — | internalnum_config only |
| 4204 | 0001 | 6 clock Bracelet section link 3 - underside | — | — | internalnum_config only |
| 4205 | 0001 | 6 clock Bracelet section link 4 - underside | — | — | internalnum_config only |
| 4206 | 0001 | 6 clock Bracelet section link 5 - underside | — | — | internalnum_config only |
| 4207 | 0001 | 6 clock Bracelet section link 6 - underside | — | — | internalnum_config only |
| 4208 | 0001 | 6 clock Bracelet section link 7 - underside | — | — | internalnum_config only |
| 4209 | 0001 | 6 clock Bracelet section link 8 - underside | — | — | internalnum_config only |
| 4210 | 0001 | 6 clock Bracelet section link 9 - underside | — | — | internalnum_config only |
| 4211 | 0001 | 6 clock Bracelet section link 10 - underside | — | — | internalnum_config only |
| 4212 | 0001 | 6 clock Bracelet section link 11 - underside | — | — | internalnum_config only |
| 4213 | 0001 | 6 clock Bracelet section link 12 - underside | — | — | internalnum_config only |
| 4214 | 0001 | 6 clock Bracelet section link 13 - underside | — | — | internalnum_config only |
| 4215 | 0001 | 6 clock Bracelet section link 14 - underside | — | — | internalnum_config only |
| 4216 | 0001 | 6 clock Bracelet section link 15 - underside | — | — | internalnum_config only |
| 4217 | 0001 | 6 clock Bracelet section link 16 - underside | — | — | internalnum_config only |
| 4218 | 0001 | 6 clock Bracelet section link 17 - underside | — | — | internalnum_config only |
| 4219 | 0001 | 6 clock Bracelet section link 18 - underside | — | — | internalnum_config only |
| 4220 | 0001 | 6 clock Bracelet section link 19 - underside | — | — | internalnum_config only |
| 4301 | 0001 | End link of 12 clock Bracelet section - Outer surface | — | — | internalnum_config only |
| 4302 | 0001 | 12 clock Bracelet section link 1 - Outer surface | — | — | internalnum_config only |
| 4303 | 0001 | 12 clock Bracelet section link 2 - Outer surface | — | — | internalnum_config only |
| 4304 | 0001 | 12 clock Bracelet section link 3 - Outer surface | — | — | internalnum_config only |
| 4305 | 0001 | 12 clock Bracelet section link 4 - Outer surface | — | — | internalnum_config only |
| 4306 | 0001 | 12 clock Bracelet section link 5 - Outer surface | — | — | internalnum_config only |
| 4307 | 0001 | 12 clock Bracelet section link 6 - Outer surface | — | — | internalnum_config only |
| 4308 | 0001 | 12 clock Bracelet section link 7 - Outer surface | — | — | internalnum_config only |
| 4309 | 0001 | 12 clock Bracelet section link 8 - Outer surface | — | — | internalnum_config only |
| 4310 | 0001 | 12 clock Bracelet section link 9 - Outer surface | — | — | internalnum_config only |
| 4311 | 0001 | 12 clock Bracelet section link 10 - Outer surface | — | — | internalnum_config only |
| 4312 | 0001 | 12 clock Bracelet section link 11 - Outer surface | — | — | internalnum_config only |
| 4313 | 0001 | 12 clock Bracelet section link 12 - Outer surface | — | — | internalnum_config only |
| 4314 | 0001 | 12 clock Bracelet section link 13 - Outer surface | — | — | internalnum_config only |
| 4315 | 0001 | 12 clock Bracelet section link 14 - Outer surface | — | — | internalnum_config only |
| 4316 | 0001 | 12 clock Bracelet section link 15 - Outer surface | — | — | internalnum_config only |
| 4317 | 0001 | 12 clock Bracelet section link 16 - Outer surface | — | — | internalnum_config only |
| 4318 | 0001 | 12 clock Bracelet section link 17 - Outer surface | — | — | internalnum_config only |
| 4319 | 0001 | 12 clock Bracelet section link 18 - Outer surface | — | — | internalnum_config only |
| 4320 | 0001 | 12 clock Bracelet section link 19 - Outer surface | — | — | internalnum_config only |
| 4401 | 0001 | End link of 6 clock Bracelet section - Outer surface | — | — | internalnum_config only |
| 4402 | 0001 | 6 clock Bracelet section link 1 - Outer surface | — | — | internalnum_config only |
| 4403 | 0001 | 6 clock Bracelet section link 2 - Outer surface | — | — | internalnum_config only |
| 4404 | 0001 | 6 clock Bracelet section link 3 - Outer surface | — | — | internalnum_config only |
| 4405 | 0001 | 6 clock Bracelet section link 4 - Outer surface | — | — | internalnum_config only |
| 4406 | 0001 | 6 clock Bracelet section link 5 - Outer surface | — | — | internalnum_config only |
| 4407 | 0001 | 6 clock Bracelet section link 6 - Outer surface | — | — | internalnum_config only |
| 4408 | 0001 | 6 clock Bracelet section link 7 - Outer surface | — | — | internalnum_config only |
| 4409 | 0001 | 6 clock Bracelet section link 8 - Outer surface | — | — | internalnum_config only |
| 4410 | 0001 | 6 clock Bracelet section link 9 - Outer surface | — | — | internalnum_config only |
| 4411 | 0001 | 6 clock Bracelet section link 10 - Outer surface | — | — | internalnum_config only |
| 4412 | 0001 | 6 clock Bracelet section link 11 - Outer surface | — | — | internalnum_config only |
| 4413 | 0001 | 6 clock Bracelet section link 12 - Outer surface | — | — | internalnum_config only |
| 4414 | 0001 | 6 clock Bracelet section link 13 - Outer surface | — | — | internalnum_config only |
| 4415 | 0001 | 6 clock Bracelet section link 14 - Outer surface | — | — | internalnum_config only |
| 4416 | 0001 | 6 clock Bracelet section link 15 - Outer surface | — | — | internalnum_config only |
| 4417 | 0001 | 6 clock Bracelet section link 16 - Outer surface | — | — | internalnum_config only |
| 4418 | 0001 | 6 clock Bracelet section link 17 - Outer surface | — | — | internalnum_config only |
| 4419 | 0001 | 6 clock Bracelet section link 18 - Outer surface | — | — | internalnum_config only |
| 4420 | 0001 | 6 clock Bracelet section link 19 - Outer surface | — | — | internalnum_config only |
| 4501 | 0001 | End link of 12 clock Bracelet section - 3clock side | — | — | internalnum_config only |
| 4502 | 0001 | 12 clock Bracelet section link 1 - 3clock side | — | — | internalnum_config only |
| 4503 | 0001 | 12 clock Bracelet section link 2 - 3clock side | — | — | internalnum_config only |
| 4504 | 0001 | 12 clock Bracelet section link 3 - 3clock side | — | — | internalnum_config only |
| 4505 | 0001 | 12 clock Bracelet section link 4 - 3clock side | — | — | internalnum_config only |
| 4506 | 0001 | 12 clock Bracelet section link 5 - 3clock side | — | — | internalnum_config only |
| 4507 | 0001 | 12 clock Bracelet section link 6 - 3clock side | — | — | internalnum_config only |
| 4508 | 0001 | 12 clock Bracelet section link 7 - 3clock side | — | — | internalnum_config only |
| 4509 | 0001 | 12 clock Bracelet section link 8 - 3clock side | — | — | internalnum_config only |
| 4510 | 0001 | 12 clock Bracelet section link 9 - 3clock side | — | — | internalnum_config only |
| 4511 | 0001 | 12 clock Bracelet section link 10 - 3clock side | — | — | internalnum_config only |
| 4512 | 0001 | 12 clock Bracelet section link 11 - 3clock side | — | — | internalnum_config only |
| 4513 | 0001 | 12 clock Bracelet section link 12 - 3clock side | — | — | internalnum_config only |
| 4514 | 0001 | 12 clock Bracelet section link 13 - 3clock side | — | — | internalnum_config only |
| 4515 | 0001 | 12 clock Bracelet section link 14 - 3clock side | — | — | internalnum_config only |
| 4516 | 0001 | 12 clock Bracelet section link 15 - 3clock side | — | — | internalnum_config only |
| 4517 | 0001 | 12 clock Bracelet section link 16 - 3clock side | — | — | internalnum_config only |
| 4518 | 0001 | 12 clock Bracelet section link 17 - 3clock side | — | — | internalnum_config only |
| 4519 | 0001 | 12 clock Bracelet section link 18 - 3clock side | — | — | internalnum_config only |
| 4520 | 0001 | 12 clock Bracelet section link 19 - 3clock side | — | — | internalnum_config only |
| 4601 | 0001 | End link of 6 clock Bracelet section - 3clock side | — | — | internalnum_config only |
| 4602 | 0001 | 6 clock Bracelet section link 1 - 3clock side | — | — | internalnum_config only |
| 4603 | 0001 | 6 clock Bracelet section link 2 - 3clock side | — | — | internalnum_config only |
| 4604 | 0001 | 6 clock Bracelet section link 3 - 3clock side | — | — | internalnum_config only |
| 4605 | 0001 | 6 clock Bracelet section link 4 - 3clock side | — | — | internalnum_config only |
| 4606 | 0001 | 6 clock Bracelet section link 5 - 3clock side | — | — | internalnum_config only |
| 4607 | 0001 | 6 clock Bracelet section link 6 - 3clock side | — | — | internalnum_config only |
| 4608 | 0001 | 6 clock Bracelet section link 7 - 3clock side | — | — | internalnum_config only |
| 4609 | 0001 | 6 clock Bracelet section link 8 - 3clock side | — | — | internalnum_config only |
| 4610 | 0001 | 6 clock Bracelet section link 9 - 3clock side | — | — | internalnum_config only |
| 4611 | 0001 | 6 clock Bracelet section link 10 - 3clock side | — | — | internalnum_config only |
| 4612 | 0001 | 6 clock Bracelet section link 11 - 3clock side | — | — | internalnum_config only |
| 4613 | 0001 | 6 clock Bracelet section link 12 - 3clock side | — | — | internalnum_config only |
| 4614 | 0001 | 6 clock Bracelet section link 13 - 3clock side | — | — | internalnum_config only |
| 4615 | 0001 | 6 clock Bracelet section link 14 - 3clock side | — | — | internalnum_config only |
| 4616 | 0001 | 6 clock Bracelet section link 15 - 3clock side | — | — | internalnum_config only |
| 4617 | 0001 | 6 clock Bracelet section link 16 - 3clock side | — | — | internalnum_config only |
| 4618 | 0001 | 6 clock Bracelet section link 17 - 3clock side | — | — | internalnum_config only |
| 4619 | 0001 | 6 clock Bracelet section link 18 - 3clock side | — | — | internalnum_config only |
| 4620 | 0001 | 6 clock Bracelet section link 19 - 3clock side | — | — | internalnum_config only |
| 4701 | 0001 | End link of 12 clock Bracelet section - 9clock side | — | — | internalnum_config only |
| 4702 | 0001 | 12 clock Bracelet section link 1 - 9clock side | — | — | internalnum_config only |
| 4703 | 0001 | 12 clock Bracelet section link 2 - 9clock side | — | — | internalnum_config only |
| 4704 | 0001 | 12 clock Bracelet section link 3 - 9clock side | — | — | internalnum_config only |
| 4705 | 0001 | 12 clock Bracelet section link 4 - 9clock side | — | — | internalnum_config only |
| 4706 | 0001 | 12 clock Bracelet section link 5 - 9clock side | — | — | internalnum_config only |
| 4707 | 0001 | 12 clock Bracelet section link 6 - 9clock side | — | — | internalnum_config only |
| 4708 | 0001 | 12 clock Bracelet section link 7 - 9clock side | — | — | internalnum_config only |
| 4709 | 0001 | 12 clock Bracelet section link 8 - 9clock side | — | — | internalnum_config only |
| 4710 | 0001 | 12 clock Bracelet section link 9 - 9clock side | — | — | internalnum_config only |
| 4711 | 0001 | 12 clock Bracelet section link 10 - 9clock side | — | — | internalnum_config only |
| 4712 | 0001 | 12 clock Bracelet section link 11 - 9clock side | — | — | internalnum_config only |
| 4713 | 0001 | 12 clock Bracelet section link 12 - 9clock side | — | — | internalnum_config only |
| 4714 | 0001 | 12 clock Bracelet section link 13 - 9clock side | — | — | internalnum_config only |
| 4715 | 0001 | 12 clock Bracelet section link 14 - 9clock side | — | — | internalnum_config only |
| 4716 | 0001 | 12 clock Bracelet section link 15 - 9clock side | — | — | internalnum_config only |
| 4717 | 0001 | 12 clock Bracelet section link 16 - 9clock side | — | — | internalnum_config only |
| 4718 | 0001 | 12 clock Bracelet section link 17 - 9clock side | — | — | internalnum_config only |
| 4719 | 0001 | 12 clock Bracelet section link 18 - 9clock side | — | — | internalnum_config only |
| 4720 | 0001 | 12 clock Bracelet section link 19 - 9clock side | — | — | internalnum_config only |
| 4801 | 0001 | End link of 6 clock Bracelet section - 9clock side | — | — | internalnum_config only |
| 4802 | 0001 | 6 clock Bracelet section link 1 - 9clock side | — | — | internalnum_config only |
| 4803 | 0001 | 6 clock Bracelet section link 2 - 9clock side | — | — | internalnum_config only |
| 4804 | 0001 | 6 clock Bracelet section link 3 - 9clock side | — | — | internalnum_config only |
| 4805 | 0001 | 6 clock Bracelet section link 4 - 9clock side | — | — | internalnum_config only |
| 4806 | 0001 | 6 clock Bracelet section link 5 - 9clock side | — | — | internalnum_config only |
| 4807 | 0001 | 6 clock Bracelet section link 6 - 9clock side | — | — | internalnum_config only |
| 4808 | 0001 | 6 clock Bracelet section link 7 - 9clock side | — | — | internalnum_config only |
| 4809 | 0001 | 6 clock Bracelet section link 8 - 9clock side | — | — | internalnum_config only |
| 4810 | 0001 | 6 clock Bracelet section link 9 - 9clock side | — | — | internalnum_config only |
| 4811 | 0001 | 6 clock Bracelet section link 10 - 9clock side | — | — | internalnum_config only |
| 4812 | 0001 | 6 clock Bracelet section link 11 - 9clock side | — | — | internalnum_config only |
| 4813 | 0001 | 6 clock Bracelet section link 12 - 9clock side | — | — | internalnum_config only |
| 4814 | 0001 | 6 clock Bracelet section link 13 - 9clock side | — | — | internalnum_config only |
| 4815 | 0001 | 6 clock Bracelet section link 14 - 9clock side | — | — | internalnum_config only |
| 4816 | 0001 | 6 clock Bracelet section link 15 - 9clock side | — | — | internalnum_config only |
| 4817 | 0001 | 6 clock Bracelet section link 16 - 9clock side | — | — | internalnum_config only |
| 4818 | 0001 | 6 clock Bracelet section link 17 - 9clock side | — | — | internalnum_config only |
| 4819 | 0001 | 6 clock Bracelet section link 18 - 9clock side | — | — | internalnum_config only |
| 4820 | 0001 | 6 clock Bracelet section link 19 - 9clock side | — | — | internalnum_config only |
| 5001 | 0001 | Reserved unused screw slot | — | — | internalnum_config only |
| 5002 | 0001 | 12 clock Bracelet section link 1 - 3clock side screw | — | — | internalnum_config only |
| 5003 | 0001 | 12 clock Bracelet section link 2 - 3clock side screw | — | — | internalnum_config only |
| 5004 | 0001 | 12 clock Bracelet section link 3 - 3clock side screw | — | — | internalnum_config only |
| 5005 | 0001 | 12 clock Bracelet section link 4 - 3clock side screw | — | — | internalnum_config only |
| 5006 | 0001 | 12 clock Bracelet section link 5 - 3clock side screw | — | — | internalnum_config only |
| 5007 | 0001 | 12 clock Bracelet section link 6 - 3clock side screw | — | — | internalnum_config only |
| 5008 | 0001 | 12 clock Bracelet section link 7 - 3clock side screw | — | — | internalnum_config only |
| 5009 | 0001 | 12 clock Bracelet section link 8 - 3clock side screw | — | — | internalnum_config only |
| 5010 | 0001 | 12 clock Bracelet section link 9 - 3clock side screw | — | — | internalnum_config only |
| 5011 | 0001 | 12 clock Bracelet section link 10 - 3clock side screw | — | — | internalnum_config only |
| 5012 | 0001 | 12 clock Bracelet section link 11 - 3clock side screw | — | — | internalnum_config only |
| 5013 | 0001 | 12 clock Bracelet section link 12 - 3clock side screw | — | — | internalnum_config only |
| 5014 | 0001 | 12 clock Bracelet section link 13 - 3clock side screw | — | — | internalnum_config only |
| 5015 | 0001 | 12 clock Bracelet section link 14 - 3clock side screw | — | — | internalnum_config only |
| 5016 | 0001 | 12 clock Bracelet section link 15 - 3clock side screw | — | — | internalnum_config only |
| 5017 | 0001 | 12 clock Bracelet section link 16 - 3clock side screw | — | — | internalnum_config only |
| 5018 | 0001 | 12 clock Bracelet section link 17 - 3clock side screw | — | — | internalnum_config only |
| 5019 | 0001 | 12 clock Bracelet section link 18 - 3clock side screw | — | — | internalnum_config only |
| 5020 | 0001 | 12 clock Bracelet section link 19 - 3clock side screw | — | — | internalnum_config only |
| 5101 | 0001 | Reserved unused screw slot | — | — | internalnum_config only |
| 5102 | 0001 | 6 clock Bracelet section link 1 - 3clock side screw | — | — | internalnum_config only |
| 5103 | 0001 | 6 clock Bracelet section link 2 - 3clock side screw | — | — | internalnum_config only |
| 5104 | 0001 | 6 clock Bracelet section link 3 - 3clock side screw | — | — | internalnum_config only |
| 5105 | 0001 | 6 clock Bracelet section link 4 - 3clock side screw | — | — | internalnum_config only |
| 5106 | 0001 | 6 clock Bracelet section link 5 - 3clock side screw | — | — | internalnum_config only |
| 5107 | 0001 | 6 clock Bracelet section link 6 - 3clock side screw | — | — | internalnum_config only |
| 5108 | 0001 | 6 clock Bracelet section link 7 - 3clock side screw | — | — | internalnum_config only |
| 5109 | 0001 | 6 clock Bracelet section link 8 - 3clock side screw | — | — | internalnum_config only |
| 5110 | 0001 | 6 clock Bracelet section link 9 - 3clock side screw | — | — | internalnum_config only |
| 5111 | 0001 | 6 clock Bracelet section link 10 - 3clock side screw | — | — | internalnum_config only |
| 5112 | 0001 | 6 clock Bracelet section link 11 - 3clock side screw | — | — | internalnum_config only |
| 5113 | 0001 | 6 clock Bracelet section link 12 - 3clock side screw | — | — | internalnum_config only |
| 5114 | 0001 | 6 clock Bracelet section link 13 - 3clock side screw | — | — | internalnum_config only |
| 5115 | 0001 | 6 clock Bracelet section link 14 - 3clock side screw | — | — | internalnum_config only |
| 5116 | 0001 | 6 clock Bracelet section link 15 - 3clock side screw | — | — | internalnum_config only |
| 5117 | 0001 | 6 clock Bracelet section link 16 - 3clock side screw | — | — | internalnum_config only |
| 5118 | 0001 | 6 clock Bracelet section link 17 - 3clock side screw | — | — | internalnum_config only |
| 5119 | 0001 | 6 clock Bracelet section link 18 - 3clock side screw | — | — | internalnum_config only |
| 5120 | 0001 | 6 clock Bracelet section link 19 - 3clock side screw | — | — | internalnum_config only |
| 5201 | 0001 | Reserved unused screw slot | — | — | internalnum_config only |
| 5202 | 0001 | 12 clock Bracelet section link 1 - 9clock side screw | — | — | internalnum_config only |
| 5203 | 0001 | 12 clock Bracelet section link 2 - 9clock side screw | — | — | internalnum_config only |
| 5204 | 0001 | 12 clock Bracelet section link 3 - 9clock side screw | — | — | internalnum_config only |
| 5205 | 0001 | 12 clock Bracelet section link 4 - 9clock side screw | — | — | internalnum_config only |
| 5206 | 0001 | 12 clock Bracelet section link 5 - 9clock side screw | — | — | internalnum_config only |
| 5207 | 0001 | 12 clock Bracelet section link 6 - 9clock side screw | — | — | internalnum_config only |
| 5208 | 0001 | 12 clock Bracelet section link 7 - 9clock side screw | — | — | internalnum_config only |
| 5209 | 0001 | 12 clock Bracelet section link 8 - 9clock side screw | — | — | internalnum_config only |
| 5210 | 0001 | 12 clock Bracelet section link 9 - 9clock side screw | — | — | internalnum_config only |
| 5211 | 0001 | 12 clock Bracelet section link 10 - 9clock side screw | — | — | internalnum_config only |
| 5212 | 0001 | 12 clock Bracelet section link 11 - 9clock side screw | — | — | internalnum_config only |
| 5213 | 0001 | 12 clock Bracelet section link 12 - 9clock side screw | — | — | internalnum_config only |
| 5214 | 0001 | 12 clock Bracelet section link 13 - 9clock side screw | — | — | internalnum_config only |
| 5215 | 0001 | 12 clock Bracelet section link 14 - 9clock side screw | — | — | internalnum_config only |
| 5216 | 0001 | 12 clock Bracelet section link 15 - 9clock side screw | — | — | internalnum_config only |
| 5217 | 0001 | 12 clock Bracelet section link 16 - 9clock side screw | — | — | internalnum_config only |
| 5218 | 0001 | 12 clock Bracelet section link 17 - 9clock side screw | — | — | internalnum_config only |
| 5219 | 0001 | 12 clock Bracelet section link 18 - 9clock side screw | — | — | internalnum_config only |
| 5220 | 0001 | 12 clock Bracelet section link 19 - 9clock side screw | — | — | internalnum_config only |
| 5301 | 0001 | Reserved unused screw slot | — | — | internalnum_config only |
| 5302 | 0001 | 6 clock Bracelet section link 1 - 9clock side screw | — | — | internalnum_config only |
| 5303 | 0001 | 6 clock Bracelet section link 2 - 9clock side screw | — | — | internalnum_config only |
| 5304 | 0001 | 6 clock Bracelet section link 3 - 9clock side screw | — | — | internalnum_config only |
| 5305 | 0001 | 6 clock Bracelet section link 4 - 9clock side screw | — | — | internalnum_config only |
| 5306 | 0001 | 6 clock Bracelet section link 5 - 9clock side screw | — | — | internalnum_config only |
| 5307 | 0001 | 6 clock Bracelet section link 6 - 9clock side screw | — | — | internalnum_config only |
| 5308 | 0001 | 6 clock Bracelet section link 7 - 9clock side screw | — | — | internalnum_config only |
| 5309 | 0001 | 6 clock Bracelet section link 8 - 9clock side screw | — | — | internalnum_config only |
| 5310 | 0001 | 6 clock Bracelet section link 9 - 9clock side screw | — | — | internalnum_config only |
| 5311 | 0001 | 6 clock Bracelet section link 10 - 9clock side screw | — | — | internalnum_config only |
| 5312 | 0001 | 6 clock Bracelet section link 11 - 9clock side screw | — | — | internalnum_config only |
| 5313 | 0001 | 6 clock Bracelet section link 12 - 9clock side screw | — | — | internalnum_config only |
| 5314 | 0001 | 6 clock Bracelet section link 13 - 9clock side screw | — | — | internalnum_config only |
| 5315 | 0001 | 6 clock Bracelet section link 14 - 9clock side screw | — | — | internalnum_config only |
| 5316 | 0001 | 6 clock Bracelet section link 15 - 9clock side screw | — | — | internalnum_config only |
| 5317 | 0001 | 6 clock Bracelet section link 16 - 9clock side screw | — | — | internalnum_config only |
| 5318 | 0001 | 6 clock Bracelet section link 17 - 9clock side screw | — | — | internalnum_config only |
| 5319 | 0001 | 6 clock Bracelet section link 18 - 9clock side screw | — | — | internalnum_config only |
| 5320 | 0001 | 6 clock Bracelet section link 19 - 9clock side screw | — | — | internalnum_config only |
| 6001 | 0001 | Box macro view1 | — | — | internalnum_config only |
| 6002 | 0001 | Box macro view2 | — | — | internalnum_config only |
| 6003 | 0001 | Box macro view3 | — | — | internalnum_config only |
| 6004 | 0001 | Box macro view4 | — | — | internalnum_config only |
| 6005 | 0001 | Box macro view5 | — | — | internalnum_config only |
| 6006 | 0001 | Box micro view1 | — | — | internalnum_config only |
| 6007 | 0001 | Box micro view2 | — | — | internalnum_config only |
| 6008 | 0001 | Box micro view3 | — | — | internalnum_config only |
| 6009 | 0001 | Box micro view4 | — | — | internalnum_config only |
| 6010 | 0001 | Box micro view5 | — | — | internalnum_config only |
| 6011 | 0001 | toppoint1 | — | — | internalnum_config only |
| 7001 | — | Case-Body | — | — | internalnum_config only |
| 7002 | — | Case-Caseback | — | — | internalnum_config only |
| 7003 | — | Crown | — | — | internalnum_config only |
| 7004 | — | Bracelet-Endlink-Center | — | — | internalnum_config only |
| 7005 | — | Bracelet-Endlink-Side | — | — | internalnum_config only |
| 7006 | — | Bracelet-Link-Center | — | — | internalnum_config only |
| 7007 | — | Bracelet-Link-Side | — | — | internalnum_config only |
| 7008 | — | Bracelet-Clasp-Outer-Center | — | — | internalnum_config only |
| 7009 | — | Bracelet-Clasp-Outer-Side | — | — | internalnum_config only |
| 7010 | — | Bracelet-Clasp-Inner | — | — | internalnum_config only |
| 7011 | — | Bezel | — | — | internalnum_config only |
| 7012 | — | Box-Lid-Front | — | — | internalnum_config only |
| 7013 | — | Box-Lid-Crown | — | — | internalnum_config only |
| 8001 | — | watch_body_weight | — | — | internalnum_config only |
| 8002 | — | timing | — | — | internalnum_config only |
| 8003 | — | pressure | — | — | internalnum_config only |
| 8004 | — | strap_weight | — | — | internalnum_config only |
| 8005 | — | amplitude | — | — | internalnum_config only |

## 已排除的 backup

| internalnum1 | internalnum2 | displayname | 類型 |
|---|---|---|---|
| 0051 | 0001 | backup | capture |
| 0052 | 0001 | backup | capture |
| 0053 | 0001 | backup | capture |
| 0054 | 0001 | backup | capture |
| 1005 | 0001 | backup | capture |
| 1006 | 0001 | backup | capture |
| 1007 | 0001 | backup | capture |
| 1008 | 0001 | backup | capture |
| 1009 | 0001 | backup | capture |
| 1010 | 0001 | backup | capture |
| 1011 | 0001 | backup | capture |
| 2006 | 0001 | backup | capture |
| 2007 | 0001 | backup | capture |
| 2008 | 0001 | backup | capture |
| 2009 | 0001 | backup | capture |
| 2010 | 0001 | backup | capture |
| 2011 | 0001 | backup | capture |
| 2012 | 0001 | backup | capture |
| 3011 | 0001 | backup | capture |
| 3012 | 0001 | backup | capture |
| 3013 | 0001 | backup | capture |
| 3014 | 0001 | backup | capture |
| 3015 | 0001 | backup | capture |
| 3016 | 0001 | backup | capture |
| 3017 | 0001 | backup | capture |
| 4031 | 0001 | backup | capture |
| 4032 | 0001 | backup | capture |
| 8006 | — | backup | point-only |
| 8007 | — | backup | point-only |
| 8008 | — | backup | point-only |
| 8009 | — | backup | point-only |
| 8010 | — | backup | point-only |




整理tasks設定對照表
```
請幫我整理有列在config/system_config的tasks service的算法內容, 以及這些tasks service是作用在哪個internalnum1, internalnum2
```
已整理完成。目前 `system_config.yaml` 的 `tasks` 區塊共有 26 個 service：

- 17 個有明確 internalnum route
- 9 個沒有 internalnum route
- 共 31 條 service route，涉及 24 組唯一 internalnum pair

注意：internalnum 並不直接寫在 `tasks`，而是透過 `services.<service_name>.internalnums` 建立對應。

### 有 internalnum route

|tasks service|算法內容|internalnum1 / internalnum2|
|---|---|---|
|`features_letter_service`|U-Net 文字分割、筆畫寬度、骨架、投影與 texture|`0011/0001`、`0013/0001`|
|`features_crown_service`|Crown Logo 輪廓、圓度、凸缺陷與 tine 分析|`0012/0001`|
|`features_marker_service`|分鐘刻度輪廓與 rounded-corner 分析|`0014/0001`、`0015/0001`|
|`crown2_service`|Top-hat/Otsu 亮點分割、位置偏差與形狀分析|`0032/0001`|
|`ocr_service`|DocTR/Tesseract 文字偵測、SAM 字元分割、筆畫特徵|`0004/0001`、`0005/0001`|
|`isolation_service`|U-Net 隔離表面後進行 KAZE/Delaunay 微結構分析|`2005/0001`、`3009/0001`、`3010/0001`|
|`bezel_marker_service`|Bezel 字／刻度骨架、角點、厚度及 bump 分析|`0016/0001`、`0017/0001`、`0027/0001`|
|`side_crown_service`|Crown 側視齒數、齒深、半徑與齒距分析|`0006/0001`|
|`texture_service`|KAZE 特徵與 Delaunay 表面 texture/bump 統計|`0026/0001`|
|`lume_hour_shape_service`|Mercedes 時針 lume 三扇區與金屬框幾何|`0023/0001`|
|`lume_hour_texture_service`|Mercedes 時針 lume 各扇區 texture 分析|`0023/0001`|
|`lume_hand_shape_service`|分／秒針 lume 圓形或矩形幾何、frame width|`0024/0001`、`0025/0001`|
|`lume_hand_texture_service`|分／秒針 lume 區域 texture/bump 分析|`0024/0001`、`0025/0001`|
|`diallume_shape_service`|錶盤夜光 marker 圓形與同心環分析|`0022/0001`|
|`diallume_texture_service`|錶盤夜光 marker texture/bump 分析|`0022/0001`|
|`front_stitch_service`|5 張正面 macro 圖拼接、錶盤中心、指針、lume、OCR|`0004–0008/0001`|
|`sidepoint_service`|多角度側面影像拼接及低對比刻字 OCR|`0002/0001`、`0003/0001`|

### 有算法，但沒有 internalnum route

- `movement1_service`：數字／刻印骨架、厚度、切角及 bump 分析。
- `movement2_service`：Movement 刻字分割、EasyOCR、筆畫幾何及 bump 分析。
- `crown_service`：Crown v1 template matching、發光點及 bezel peak 分析。
- `links_service`：Bracelet link／slot 分割、數量及骨架長度。
- `bracelet_service`：對焦分數、內外環擬合及同心度。
- `doctr_service`：DocTR/Tesseract OCR 與完整字形拓樸特徵。
- `stitched_band2_service`：SAM bracelet ROI instance segmentation、過濾及去重。
- `openbacktop_service`：Open-back 中心、圓環、螺絲、金屬角及 holder 幾何。
- `lume_service`：設定不完整；`run_lume_cli.py`、`lume_config.yaml` 和同名 `services` 區塊均不存在，無法確認實際算法。

另外有 7 組 internalnum 同時觸發多個算法，例如 `0004/0001` 同時對應 OCR 與 front stitch，`0022–0025/0001` 則分別同時執行 shape 和 texture 分析。

[查看完整算法、displayname、實作類別與 internalnum 對照報告](C:/Users/a3146/Documents/Codex/2026-09-11/referenced-chatgpt-conversation-this-is-an/outputs/tasks_service_algorithm_internalnum_report.md)

分析基準

- Branch / commit：`dev` / `19e241b27869`
- `config/system_config.yaml` SHA-256 前 12 碼：`325450dc0f4f`
- `config/internalnum_config.yaml` SHA-256 前 12 碼：`8178e5f1f42e`
- `system_config.yaml` 受 `.gitignore` 排除；因此本報告分析的是目前工作區實際設定，不是 commit 內檔案。
- internalnum route 只採 `services.<service_name>.internalnums` 的明確設定，不從算法名稱或 displayname 猜測。
- 算法內容由 `tasks.<service>.cli_script`、其匯入的 `tasks/task_algos/*_algo.py` 與對應 `task_config` 交叉確認。

## 摘要

| 項目 | 數量 |
|---|---:|
| tasks service 總數 | 26 |
| 有 internalnum route 的 service | 17 |
| 無 internalnum route 的 service | 9 |
| service-to-internalnum route 數 | 31 |
| unique internalnum pair | 24 |
| 同一 pair 被多個 service 使用 | 7 |

## 完整對照表

| tasks service | mode | 實作 | 算法內容 | internalnum1 / internalnum2 | 狀態 |
|---|---|---|---|---|---|
| lume_service | cli | — | 設定指向不存在的 `run_lume_cli.py` 與 `lume_config.yaml`；目前無法由此 task 追到可執行算法，不應直接推定為 dial-lume 算法。 | — | tasks only；無 services route；缺 CLI wrapper、task_config |
| features_letter_service | api | `features_algo.py` / `FeaturesProcessor` | 共用 `FeaturesProcessor`：U-Net 分割文字特徵；計算筆畫寬度、骨架長度、X/Y 投影、輪廓特徵與可選的表面 bump/texture 指標。 | `0011` / `0001` — Y in Officially<br>`0013` / `0001` — M in Made at near 6 | 有 internalnum route |
| features_crown_service | api | `features_algo.py` / `FeaturesProcessor` | 共用 `FeaturesProcessor`：U-Net 分割錶盤皇冠 Logo；分析面積、周長、圓度、凸缺陷與 crown tine 數量，並可做 bump/texture 分析。 | `0012` / `0001` — Logo on Dial | 有 internalnum route |
| features_marker_service | api | `features_algo.py` / `FeaturesProcessor` | 共用 `FeaturesProcessor`：U-Net 分割分鐘刻度；分析 rounded-rectangle 輪廓、四角圓角半徑、Hu moments／radial contour signature 與可選 texture。 | `0014` / `0001` — 60 minute marker<br>`0015` / `0001` — 1 minute marker | 有 internalnum route |
| movement1_service | cli | `movement1_algo.py` / `Movement1Processor` | U-Net 背景 mask 反相後擷取數字／刻印（程式針對類似數字 3）；做骨架排序、厚度 profile、直線／曲線分段、頭尾切角、亮度 type 分類及 bump 分析。 | — | services.internalnums 為空 |
| movement2_service | cli | `movement2_algo.py` / `Movement2Processor` | U-Net 分割 movement 刻字；connected-components 拆字，分析骨架／筆畫厚度與輪廓幾何，以亮度分 type；EasyOCR 多角度辨識並對文字區做 bump 分析。 | — | services.internalnums 為空 |
| crown_service | cli | `crown_algo.py` / `CrystalProcessor` | Crown v1，純傳統 CV：亮／暗雙 template 多尺度多角度比對；檢查發光點亮度、缺點／暗點及 crown 旁 bezel 的週期峰值。 | — | tasks only；無 services route |
| crown2_service | cli | `crown2_algo.py` / `CrystalProcessor` | Crown v2，純傳統 CV：top-hat + Otsu 分割亮點、template 尺度對齊；量測每點位置偏差、半徑、長寬比、solidity，以及 bezel profile 峰值。 | `0032` / `0001` — Crown on sapphire crystal | 有 internalnum route |
| links_service | cli | `links_algo.py` / `LinksProcessor` | U-Net 分割 bracelet link 與 slot；connected-components 計數，保留中心 ROI 內 link，骨架化後以 graph longest path 量測每節長度。 | — | services.internalnums 為空 |
| bracelet_service | cli | `bracelet_algo.py` / `PinGeometryProcessor` | 純 OpenCV：Laplacian variance 評估對焦；Otsu／形態學取得內外環，least-squares 擬合圓並量測同心度，可用 radial refinement 修正。 | — | services.internalnums 為空 |
| doctr_service | cli | `doctr_algo.py` / `DoctrProcessor` | DocTR 偵測文字／行與字元框，Tesseract 作交叉字元分割；抽取骨架端點／交點、筆畫寬、Hu moments、Fourier descriptors、projection profiles 等字形特徵。 | — | services.internalnums 為空 |
| ocr_service | cli | `ocr_algo.py` / `OCRProcessor` | DocTR 偵測 word（無結果時用 Tesseract），SAM 逐字分割，再以 `char_features` 計算骨架、筆畫與字元幾何特徵。 | `0004` / `0001` — Upper text<br>`0005` / `0001` — Lower text | 有 internalnum route |
| isolation_service | cli | `isolation_algo.py` / `IsolationProcessor` | U-Net 合併指定 class mask 隔離目標表面，套用 Gaussian／亮度／HSV／ROI 等前處理，再以 KAZE + Delaunay bump pipeline 分析微結構與粗糙度。 | `3009` / `0001` — Rachet wheel (Big yellow gear)<br>`3010` / `0001` — A letter or number in the serial<br>`2005` / `0001` — A letter or number in the serial | 有 internalnum route |
| bezel_marker_service | api | `bezel_marker_algo.py` / `BezelMarkerProcessor` | U-Net 分割金色 bezel 字／刻度；骨架與 graph longest path 分析線段、角點、筆畫厚度，含數字 3 的拓樸特例，並做 bump/particle 分析。 | `0016` / `0001` — Bezel at 30 (middle of 3 in 30)<br>`0017` / `0001` — Bezel at 45 mark (centered)<br>`0027` / `0001` — bottom subdial: 20: 0 in 20 | 有 internalnum route |
| side_crown_service | api | `side_crown_algo.py` / `SideCrownProcessor` | U-Net 分割側視 crown；把最大輪廓轉成極座標 radius-vs-angle，從 peaks／valleys 計算外內半徑、齒數、齒深、齒距對稱與圓度誤差。 | `0006` / `0001` — Crown | 有 internalnum route |
| texture_service | api | `texture_algo.py` / `TextureProcessor` | 薄封裝至 `SurfaceTextureAnalyzer`：KAZE 微特徵、focus／Otsu 區域 mask、Delaunay 空間統計，輸出 bump 數量、密度與分析面積。 | `0026` / `0001` — Dial area with no text just inside of 1 marker | 有 internalnum route |
| lume_hour_shape_service | api | `lume_hour_shape_algo.py` / `LumeHourShapeProcessor` | U-Net 分割 Mercedes hour-hand lume 與 Y 型金屬框；取最大 3 個扇形 sector，量測內尖角、兩直邊長、弧半徑及金屬框平均寬度。 | `0023` / `0001` — Hour hand lume - near lower text for mercedes hand | 有 internalnum route |
| lume_hour_texture_service | api | `lume_hour_texture_algo.py` / `LumeHourTextureProcessor` | U-Net 分割 Mercedes hour-hand lume；取最大 3 個扇形 sector，逐區執行 KAZE + Delaunay texture/bump 分析並統計密度。 | `0023` / `0001` — Hour hand lume - near lower text for mercedes hand | 有 internalnum route |
| lume_hand_shape_service | api | `lume_hand_shape_algo.py` / `LumeHandShapeProcessor` | U-Net 分割指針 lume 與深色框；依面積比分類圓形／矩形，量測尺寸、旋轉角、圓角半徑與 frame width。 | `0024` / `0001` — Minute hand lume - at end futher from center<br>`0025` / `0001` — Second hand lume | 有 internalnum route |
| lume_hand_texture_service | api | `lume_hand_texture_algo.py` / `LumeHandTextureProcessor` | U-Net 取得 hand-lume mask 並填洞；只在 lume 區域執行 KAZE + Delaunay texture/bump 分析。 | `0024` / `0001` — Minute hand lume - at end futher from center<br>`0025` / `0001` — Second hand lume | 有 internalnum route |
| diallume_shape_service | api | `diallume_shape_algo.py` / `DialLumeShapeProcessor` | U-Net 取得 dial lume mask 並填洞；找圓形 lume、半徑／area ratio，並以 edge/contour 偵測同心環。 | `0022` / `0001` — Lume of the 1 hour marker | 有 internalnum route |
| diallume_texture_service | api | `diallume_texture_algo.py` / `DialLumeTextureProcessor` | U-Net 取得 dial-lume mask 並填洞；在該區域執行 KAZE + Delaunay texture/bump 分析。 | `0022` / `0001` — Lume of the 1 hour marker | 有 internalnum route |
| front_stitch_service | cli | `front_stitch_algo.py` / `FrontStitchProcessor` | 彙整 5 張正面 macro 圖與 masks，使用 stage kinematic prior 進行拼接；再分析 dial center、lume 形狀、時／分／秒針幾何、同心環與可選 LLM OCR。 | `0004` / `0001` — Upper text<br>`0005` / `0001` — Lower text<br>`0006` / `0001` — Crown<br>`0007` / `0001` — Bottom right lug<br>`0008` / `0001` — Dial Left | 有 internalnum route |
| stitched_band2_service | cli | `stitched_band2_algo.py` / `StitchedBand2Processor` | SAM class-agnostic instance segmentation：依 view/camera profile 在 bracelet ROI 內 resize 或 tiled 推論，做尺寸／位置過濾、IoU 去重與可選 LLM OCR。 | — | tasks only；無 services route |
| sidepoint_service | cli | `sidepoint_algo.py` / `SidepointProcessor` | 把多角度側面圖以 phase correlation translation 配準並 alpha blend；強化低對比刻字後用 DocTR+SAM、LLM 或 hybrid OCR，並重組字元／單字／行。 | `0002` / `0001` — Top Side<br>`0003` / `0001` — Bottom Side | 有 internalnum route |
| openbacktop_service | api | `openbacktop_algo.py` / `OpenBackTopProcessor` | 純 OpenCV 幾何分析：找 open-back 中心／外徑／同心環、中央黑圓、黑螺絲、金屬梯形角、cover edge/kink 與 holder cylinders。 | — | services.internalnums 為空 |

## 無 internalnum route 的 tasks service

| service_name | wrapper | task_config | 狀態 |
|---|---|---|---|
| lume_service | tasks/cli_wrappers/run_lume_cli.py | config/lume_config.yaml | tasks only；無 services route；缺 CLI wrapper、task_config |
| movement1_service | tasks/cli_wrappers/run_movement1_cli.py | config/movement1_config.yaml | services.internalnums 為空 |
| movement2_service | tasks/cli_wrappers/run_movement2_cli.py | config/movement2_config.yaml | services.internalnums 為空 |
| crown_service | tasks/cli_wrappers/run_crown_cli.py | config/crown_config.yaml | tasks only；無 services route |
| links_service | tasks/cli_wrappers/run_links_cli.py | config/links_config.yaml | services.internalnums 為空 |
| bracelet_service | tasks/cli_wrappers/run_bracelet_cli.py | config/bracelet_config.yaml | services.internalnums 為空 |
| doctr_service | tasks/cli_wrappers/run_doctr_cli.py | config/doctr_config.yaml | services.internalnums 為空 |
| stitched_band2_service | tasks/cli_wrappers/run_stitched_band2_cli.py | config/stitched_band2_config.yaml | tasks only；無 services route |
| openbacktop_service | tasks/cli_wrappers/run_openbacktop_cli.py | config/openbacktop_config.yaml | services.internalnums 為空 |

## 同一 internalnum pair 對多個 tasks service

| internalnum1 | internalnum2 | displayname | tasks services |
|---|---|---|---|
| 0004 | 0001 | Upper text | ocr_service; front_stitch_service |
| 0005 | 0001 | Lower text | ocr_service; front_stitch_service |
| 0006 | 0001 | Crown | side_crown_service; front_stitch_service |
| 0022 | 0001 | Lume of the 1 hour marker | diallume_shape_service; diallume_texture_service |
| 0023 | 0001 | Hour hand lume - near lower text for mercedes hand | lume_hour_shape_service; lume_hour_texture_service |
| 0024 | 0001 | Minute hand lume - at end futher from center | lume_hand_shape_service; lume_hand_texture_service |
| 0025 | 0001 | Second hand lume | lume_hand_shape_service; lume_hand_texture_service |

