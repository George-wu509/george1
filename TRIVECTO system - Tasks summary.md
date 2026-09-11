

|                                                     |     |
| --------------------------------------------------- | --- |
| [[#### App 設定對照表（dev）]]                             |     |
| [[#### 整理tasks設定對照表]]                               |     |
| [[#### 是否有針對stitched image的task service]]           |     |
| [[#### stitched image有用unet辨識分針, 時針, Hour Markers]] |     |
|                                                     |     |

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




#### 整理tasks設定對照表
```
請幫我整理有列在config/system_config的tasks service的算法內容, 以及這些tasks service是作用在哪個internalnum1, internalnum2
```
已整理完成。目前 `system_config.yaml` 的 `tasks` 區塊共有 26 個 service：

- 17 個有明確 internalnum route
- 9 個沒有 internalnum route
- 共 31 條 service route，涉及 24 組唯一 internalnum pair

注意：internalnum 並不直接寫在 `tasks`，而是透過 `services.<service_name>.internalnums` 建立對應。

### 有 internalnum route

| tasks service               | 算法內容                                 | internalnum1 / internalnum2         |
| --------------------------- | ------------------------------------ | ----------------------------------- |
| `features_letter_service`   | U-Net 文字分割、筆畫寬度、骨架、投影與 texture       | `0011/0001`、`0013/0001`             |
| `features_crown_service`    | Crown Logo 輪廓、圓度、凸缺陷與 tine 分析        | `0012/0001`                         |
| `features_marker_service`   | 分鐘刻度輪廓與 rounded-corner 分析            | `0014/0001`、`0015/0001`             |
| `crown2_service`            | Top-hat/Otsu 亮點分割、位置偏差與形狀分析          | `0032/0001`                         |
| `ocr_service`               | DocTR/Tesseract 文字偵測、SAM 字元分割、筆畫特徵   | `0004/0001`、`0005/0001`             |
| `isolation_service`         | U-Net 隔離表面後進行 KAZE/Delaunay 微結構分析    | `2005/0001`、`3009/0001`、`3010/0001` |
| `bezel_marker_service`      | Bezel 字／刻度骨架、角點、厚度及 bump 分析          | `0016/0001`、`0017/0001`、`0027/0001` |
| `side_crown_service`        | Crown 側視齒數、齒深、半徑與齒距分析                | `0006/0001`                         |
| `texture_service`           | KAZE 特徵與 Delaunay 表面 texture/bump 統計 | `0026/0001`                         |
| `lume_hour_shape_service`   | Mercedes 時針 lume 三扇區與金屬框幾何           | `0023/0001`                         |
| `lume_hour_texture_service` | Mercedes 時針 lume 各扇區 texture 分析      | `0023/0001`                         |
| `lume_hand_shape_service`   | 分／秒針 lume 圓形或矩形幾何、frame width        | `0024/0001`、`0025/0001`             |
| `lume_hand_texture_service` | 分／秒針 lume 區域 texture/bump 分析         | `0024/0001`、`0025/0001`             |
| `diallume_shape_service`    | 錶盤夜光 marker 圓形與同心環分析                 | `0022/0001`                         |
| `diallume_texture_service`  | 錶盤夜光 marker texture/bump 分析          | `0022/0001`                         |
| `front_stitch_service`      | 5 張正面 macro 圖拼接、錶盤中心、指針、lume、OCR     | `0004–0008/0001`                    |
| `sidepoint_service`         | 多角度側面影像拼接及低對比刻字 OCR                  | `0002/0001`、`0003/0001`             |

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




#### 是否有針對stitched image的task service
```
在App main.py裡面有把front view的0004, 0005, 0008, 0009, 0010進行stitch的front stitched image, 那得到這個image有沒有針對這個image的task service? 有分析到那些設質呢? 請詳細整理說明
```
有。我把目前可取得的 `main.py` 系列程式、`system_config.yaml` 與 Front task routing 對過一次後，可以把這件事分成兩層來看：

**第一層：0004 / 0005 / 0008 / 0009 / 0010 的確有專門的 `front_stitch_service`。**  
**第二層：目前沒有看到「stitch 完成後，再把 final front stitched image 自動送進 OCR / texture / lume / feature 等其他 task service」的明確 routing。**

所以 `front_stitch_service` 本身主要是 **aggregator / stitching task**，不是目前看來的「完整 Front image feature analyzer」。

---

## 1. 這 5 張 Front image 是哪五個 point？

目前 `FrontViewCollection` 的結構順序是：

`glasspoint1 → toppoint1 → sidepoint1 → macropoint1 → macropoint2 → macropoint3 → macropoint4 → macropoint5 → macropoint6 → macropoint7 ...` watchview_structure(1).pyPY

這和你現在的 internalnum1 對應關係吻合：

|internalnum1|實際 Front point|是否參與 Stitch|目前另外綁定的 task|
|---|---|---|---|
|**0004**|`Front.macropoint1.std_1`|✅|`ocr_service`|
|**0005**|`Front.macropoint2.std_1`|✅|`ocr_service`|
|0006|`Front.macropoint3.std_1`|❌|其他 Front task|
|0007|`Front.macropoint4.std_1`|❌|—|
|**0008**|`Front.macropoint5.std_1`|✅|沒看到其他 direct task|
|**0009**|`Front.macropoint6.std_1`|✅|沒看到其他 direct task|
|**0010**|`Front.macropoint7.std_1`|✅|沒看到其他 direct task|

`system_config.yaml` 明確把這五個 selector 全部綁到：

```
Front.macropoint1.std_1
Front.macropoint2.std_1
Front.macropoint5.std_1
Front.macropoint6.std_1
Front.macropoint7.std_1
```

而且它們共同屬於 `front_stitch_service`。

---

# 2. 確實存在一個專門的 `front_stitch_service`

目前設定是：

- Service：`front_stitch_service`
- Python API server：`tasks/api_servers/server_front_stitch.py`
- port：`5021`
- config：`config/front_stitch_config.yaml`
- `aggregator: true`
- `suppress_single_image_analysis: true`

另外 CLI 模式也已經有完整 task registration：

- `service_name: front_stitch_service`
- `tasks/cli_wrappers/run_front_stitch_cli.py`
- `config/front_stitch_config.yaml`
- timeout = 120 秒

所以這不是單純在 `main.py` 裡用 OpenCV 臨時把五張圖拼起來而已，而是架構上已經把它正式視為一個 **Task Service**。

---

# 3. `aggregator: true` 是這套設計最重要的地方

一般的 Moonlight task，例如：

```
texture_service
ocr_service
bezel_marker_service
```

通常是一張 image 進去：

> image → task service → JSON / mask / visualization

但 `front_stitch_service` 不一樣。

它被設定成：

```
aggregator: true
suppress_single_image_analysis: true
```

意思是它不是：

> 0004 → stitch  
> 0005 → stitch  
> 0008 → stitch  
> ...

各自獨立分析。

而是概念上：

```
0004 ─┐
0005 ─┤
0008 ─┼──→ front_stitch_service
0009 ─┤
0010 ─┘
              ↓
       Front Stitched Image
```

這就是為什麼叫 **aggregator**。

---

# 4. main.py 是怎麼把每張圖送進 Stitch Aggregator 的？

我目前 Library 裡真正包含新 Front Stitch integration 的版本是 `main(6).py`。在 capture 完成之後，程式先取得該 image 所需要的 `req_algos`。

如果其中包含 front stitch task，就會呼叫：

```
workflow_engine.handle_front_stitch_asset(...)
```

而不是直接把這一張圖當成一般 task 跑掉。

傳進去的資料包括：

```
watchid
run_id
view_name
point_name
capture_id
image_path
template_xy
analysis_dir
user_ctx
```

特別值得注意的是：

### `template_xy`

程式不只把 image 傳給 stitcher，也把該拍攝點的 template X/Y 位置傳進去：

```
template_xy={
    "x": ...,
    "y": ...
}
```

這表示 Stitch service 有取得：

> 「這張照片是哪個 point」  
> ＋  
> 「拍這張照片時 camera/watch 的相對 X/Y 位置」

這對穩定的 deterministic stitching 非常重要，因為不必完全只依賴影像 feature matching 猜相對位置。

---

# 5. `suppress_single_image_analysis: true` 到底代表什麼？

這點很容易誤會。

當一張照片例如 `0008 / macropoint5` 拍完後：

```
single_image_algos = req_algos
```

如果其中有 `front_stitch_service`：

1. 先呼叫 `handle_front_stitch_asset()`
2. 然後呼叫：

```
_filter_single_image_tasks(req_algos)
```

把 aggregator 類 task 從一般 single-image analysis 中移除。

因此不會發生：

```
0008
 ↓
front_stitch_service(0008)

0009
 ↓
front_stitch_service(0009)

0010
 ↓
front_stitch_service(0010)
```

而是累積成：

```
0004
0005
0008
0009
0010
  ↓
Front Stitch Aggregator
  ↓
Front Stitched Image
```

這個設計是合理的。

---

# 6. 但 0004 / 0005 還有另外的 OCR

這又是另一個重要差別。

`Front.macropoint1.std_1` 和：

`Front.macropoint2.std_1`

除了被送到 `front_stitch_service` 之外，還明確綁：

```
ocr_service
```

所以實際上比較像：

```
0004 ───────→ OCR service
   │
   └────────→ Stitch aggregator

0005 ───────→ OCR service
   │
   └────────→ Stitch aggregator

0008 ───────→ Stitch aggregator
0009 ───────→ Stitch aggregator
0010 ───────→ Stitch aggregator
```

`_filter_single_image_tasks()` 的目的正是：

> 把 `front_stitch_service` 從 single-image tasks 中拿掉，  
> 但仍然保留這張圖真正需要單獨執行的其他 task，例如 OCR。

---

# 7. 所以 0004 / 0005 的 OCR 是分析 stitched image 嗎？

**不是。**

目前 routing 顯示：

```
OCR
input:
Front.macropoint1.std_1
Front.macropoint2.std_1
```

也就是 OCR 是直接看：

> 0004 原始 macro image  
> 0005 原始 macro image

而不是：

> Front Stitched Image → OCR

這個差別相當重要。

---

# 8. 那 Front Stitched Image 本身目前又被哪些 Task Service 分析？

這是我檢查後最重要的結論：

## 目前沒有看到明確的 downstream task routing。

我目前沒有找到類似：

```
ocr_service:
  watchView: Front.stitched

features_service:
  watchView: Front.stitched

bezel_service:
  watchView: Front.stitched

texture_service:
  watchView: Front.stitched
```

或：

```
front_stitched_image
    ↓
some_analysis_service
```

這樣的 mapping。

目前明確存在的是：

```
5 source images
       ↓
front_stitch_service
       ↓
Front Stitched Image
```

而不是：

```
5 source images
       ↓
front_stitch_service
       ↓
Front Stitched Image
       ↓
OCR
       ↓
Segmentation
       ↓
Dial geometry
       ↓
Bezel geometry
       ↓
Feature analysis
```

至少我目前可取得的 `system_config.yaml` 沒有這第二段。

---

# 9. 那 `front_stitch_service` 自己到底「分析」哪些特質？

這裡必須區分兩種東西。

### A. 我現在可以確定它收到的資訊

程式明確傳入：

|資訊|用途|
|---|---|
|`watchid`|哪一支 watch|
|`run_id`|哪一次 scan|
|`view_name`|Front|
|`point_name`|macro1 / macro2 / macro5 / macro6 / macro7|
|`capture_id`|std_1|
|`image_path`|原始影像|
|`template_xy`|拍攝位置|
|`analysis_dir`|analysis output directory|
|`user_ctx`|session / 權限資訊|

而 main 端至少會讀 stitch service 的：

```
stitch_res.get("status", "complete")
```

所以 aggregator 有一個 result/status 回傳。

### B. 我目前不能從現有 Library 檔案確認的內容

我還沒有取得這三份實際 implementation：

```
config/front_stitch_config.yaml

tasks/api_servers/server_front_stitch.py

tasks/cli_wrappers/run_front_stitch_cli.py
```

以及真正實作：

```
WorkflowManager.handle_front_stitch_asset()
```

的 WorkflowManager 原始碼。

所以目前**不能負責任地說**它一定計算了：

- feature matching 數量
- homography matrix
- reprojection error
- overlap %
- seam quality
- blend quality
- registration error
- scale error
- rotation error
- stitched coverage
- alignment confidence

這些是 Stitch service「可能很合理會有的品質指標」，但目前看到的 code reference 還不足以證明它真的有輸出這些值。

---

# 10. Front View 其他分析 Task 不等於 Front Stitched Image analysis

目前系統 Front 還有很多 task，例如：

|Service|Input|
|---|---|
|`texture_service`|`Front.micropoint16.std_1`|
|`lume_hour_shape_service`|`Front.micropoint13.std_1`|
|`lume_hour_texture_service`|`Front.micropoint13.hdl_1`|
|`lume_hand_shape_service`|`Front.micropoint14.std_1`, `micropoint15.std_1`|
|`lume_hand_texture_service`|`Front.micropoint14.hdl_1`, `micropoint15.hdl_1`|
|`diallume_shape_service`|`Front.micropoint12.std_1`|
|`diallume_texture_service`|`Front.micropoint12.hdl_1`|

但這些全部是：

**各自指定的 Micro image analysis**

而不是：

**Front Stitched Image analysis**。

所以目前架構應該畫成：

```
                         ┌──→ OCR  (0004)
0004 ───────────────────┤
                         └───────┐
                                 │
                         ┌──→ OCR│ (0005)
0005 ───────────────────┤       │
                         └───────┤
                                 │
0008 ────────────────────────────┤
                                 ├─→ front_stitch_service
0009 ────────────────────────────┤        │
                                 │        ▼
0010 ────────────────────────────┘   Front Stitched Image
                                          │
                                          │
                              [目前沒有看到明確 downstream
                               feature-analysis routing]


其他 Micro images：
micropoint12 ─→ Dial Lume Shape / Texture
micropoint13 ─→ Hour Lume Shape / Texture
micropoint14 ─→ Hand Lume
micropoint15 ─→ Hand Lume
micropoint16 ─→ Texture
...
```

---

# 11. Stitch 前的原始 images 本身也有正式 DB / S3 asset tracking

這五張 source images 並不是暫存後就消失。

Capture 完成後，程式會建立：

- `local_path`
- `raw_image_id`
- `s3_key`
- UI asset registry
- V3 image asset metadata

V3 image asset 還會保存：

```
watchid
run_id
view_name
point_name
capture_id
asset_role
asset_index
local_path
s3_key
metadata
```

其中 `asset_role` 會是例如：

```
raw_single
hdr_fused
```

main(6).pyPY

所以整體 lineage 是可以追的：

```
Watch
 ↓
Run
 ↓
Front point
 ↓
Capture instance
 ↓
Raw image asset
 ↓
Front Stitch Aggregator
 ↓
Front Stitched Asset
```

這其實很適合你現在正在建的 Authentication pipeline，因為之後可以清楚追溯某一個 stitched evidence 到底是由哪五張原圖產生的。

---

# 12. 「分析結果」和 PDF 報告又是另一層

一般 task analysis 的流程是：

```
_run_analysis_safe(
    task_name="multi",
    image_path=image_path,
    output_dir=analysis_dir,
    tasks=req_algos,
    user=user_ctx
)
```

之後再：

```
_process_step_results(...)
```

把結果寫入 Local DB / Analysis directory / S3 queue。

PDF report 的 generic renderer 可以顯示 task JSON 裡面的：

```
metrics
visuals
image
data
```

例如 `metrics` 會轉成 Statistical Results & Metrics table，而 `visuals` 可以加入額外 visualization。main(4).pyPY

但這只是**報告層支援 generic task output**；不能反過來證明 `front_stitch_service` 現在一定輸出了哪些 stitch metrics。

---

# 13. 所以目前可以做一個很清楚的結論

### ✅ 已經有的

**Front Stitched Image 的產生 Task：**

```
front_stitch_service
```

輸入：

```
0004 = macropoint1
0005 = macropoint2
0008 = macropoint5
0009 = macropoint6
0010 = macropoint7
```

它是：

```
aggregator: true
suppress_single_image_analysis: true
```

也就是一個真正的多圖片 aggregator task。

### ✅ 來源圖片另外會做的分析

目前明確看到：

```
0004 → OCR
0005 → OCR

0008 → 沒看到其他 direct analysis
0009 → 沒看到其他 direct analysis
0010 → 沒看到其他 direct analysis
```

其中 `0004/0005 → OCR` 是直接分析原圖，不是 stitched image。system_config.yamlYAML

### ⚠️ 目前沒有看到的

目前沒有看到：

```
Front Stitched Image
        ↓
OCR
Dial segmentation
Dial geometry
Bezel geometry
Hands geometry
Marker positions
Global symmetry
Global color
Global dimensions
Logo/text relative locations
...
```

這種 **stitched-image-specific downstream analysis pipeline**。

---

## 14. 這一點對 Authentication 特別重要

以你現在做的 **Hybrid Statistical–Bayesian Authentication** 架構來說，我認為 Front Stitched Image 的價值其實不應該只停留在「做一張漂亮的完整 Front image」。

它很適合產生一組 **global spatial / geometric evidence**，例如：

```
Front Stitched Image
        │
        ├─ Global Dial Geometry
        │    ├─ dial center
        │    ├─ dial radius / ellipse
        │    └─ circularity
        │
        ├─ Marker Spatial Layout
        │    ├─ marker angular positions
        │    ├─ radial position
        │    ├─ spacing
        │    └─ symmetry
        │
        ├─ Text / Logo Layout
        │    ├─ ROLEX logo position
        │    ├─ text baselines
        │    ├─ text-to-center distance
        │    └─ relative spacing
        │
        ├─ Hands Geometry
        │
        ├─ Date Window Geometry
        │
        ├─ Bezel ↔ Dial Registration
        │
        └─ Global Appearance
             ├─ color distribution
             ├─ symmetry
             └─ component spatial relationships
```

這些資料對 Authentication 往往比單張局部 image 更有價值，因為 stitched image 能提供「**各 component 之間的相對位置**」。

### 因此，目前程式的狀態可以一句話總結：

> **現在已經有 `front_stitch_service` 負責把 0004、0005、0008、0009、0010 聚合成 Front Stitched Image；但從目前的 task routing 看，這張 stitched image 本身尚未明確接到第二階段的 global feature-analysis services。0004/0005 的 OCR 等現有分析仍然是在原始單張 image 上執行。**

如果你的目標是讓 **Front Stitched Image 真正成為 Authentication evidence**，下一步最值得做的是把目前的 `front_stitch_service` 往後接一層 **`front_global_analysis_service`**，專門萃取 global geometry / relative-position / symmetry / layout features，而不是重複目前各 Micro image 已經在做的 texture/lume 細節分析。



#### stitched image有用unet辨識分針, 時針, Hour Markers
```
所以在執行App main.py當拍攝了0004,0005,0008,0009,0010之後是否自動進行stitch? 以及這張錶面stitched image儲存在哪裡? 我記得之前有code針對這張錶面stitched image有用unet辨識分針, 時針, Hour Markers等做segmentation以及相關分析, 請幫我搜尋code是否有
```
我重新沿著目前能搜尋到的 `main.py` 快照、`system_config.yaml`、歷史 log，以及 U-Net training/inference code 查了一輪。結論比上一則可以更明確一些：

**目前 App 的程式確實已經做成「自動 Front stitch aggregator」；但是你記得的「stitched 錶面 → U-Net → 分針／時針／Hour Markers segmentation → 幾何分析」那一段程式，我有找到 U-Net 本體與 segmentation infrastructure，卻沒有在目前 App 的 production routing 中找到它和 `front_stitch_service` 接起來。**

## 1. 0004、0005、0008、0009、0010 拍完後，會不會自動 Stitch？

**會自動進入 stitch 流程，不需要另外按 Stitch 按鈕。**

目前 `system_config.yaml` 明確設定：

```
front_stitch_service:
    ...
    aggregator: true
    suppress_single_image_analysis: true
    watchView:
        Front.macropoint1.std_1,
        Front.macropoint2.std_1,
        Front.macropoint5.std_1,
        Front.macropoint6.std_1,
        Front.macropoint7.std_1
```

也就是依目前 Front internalnum 順序，正是你說的 **0004、0005、0008、0009、0010**。

而且 App capture code 在每一張圖片拍完、Raw image 已經存好之後，會檢查該 image 的 `req_algos`。如果裡面包含 `front_stitch_service`，它會**立即自動呼叫**：

```
self.workflow_engine.handle_front_stitch_asset(
    watchid=watchid,
    run_id=v3_run_id,
    view_name=view_key,
    point_name=field_name,
    capture_id=c_id,
    image_path=local_path,
    template_xy={...},
    analysis_dir=analysis_dir,
    user_ctx=...
)
```

所以實際架構是：

```
0004 拍完 ─→ register 到 Front Stitch Aggregator
0005 拍完 ─→ register 到 Front Stitch Aggregator
0008 拍完 ─→ register 到 Front Stitch Aggregator
0009 拍完 ─→ register 到 Front Stitch Aggregator
0010 拍完 ─→ register 到 Front Stitch Aggregator
                         │
                         ▼
              五張資料條件滿足
                         │
                         ▼
                Front Stitched Image
```

Simulation worker 也有完全相同的自動呼叫，所以不是只有 production hardware mode 才有這個設計。

### 但有一個地方目前不能 100% 從現有檔案證明

真正判斷：

> 「現在是否已經集滿 0004、0005、0008、0009、0010？集滿才正式 stitch。」

這個 gate 不在 `main.py`，而是在：

```
WorkflowManager.handle_front_stitch_asset()
```

以及：

```
config/front_stitch_config.yaml
tasks/api_servers/server_front_stitch.py
```

裡面。

目前 Library 搜尋不到這幾個檔案的實體內容，所以我可以確認：

**每張都會自動送進 aggregator。**

而根據：

```
aggregator: true
watchView: [五張指定 image]
```

架構很明確就是為了「五張集滿再產生 stitched result」，但我目前不能假裝看過 `handle_front_stitch_asset()` 裡面那個 `if all_required_images_present:` 的實際實作。

---

# 2. 那五張原始圖片存在哪裡？

原始拍攝照片還是存到：

```
D:\Moonlight\ImagingLibWatch\
    Local_Data\
        <run_xxx>\
            Raw\
                <uuid>.png
```

實際現有 capture report 就顯示例如：

```
D:\Moonlight\ImagingLibWatch\
Local_Data\
run_16613T(V3)\
Raw\
efb8811a3bc9462485b9beb00c7d5c5b.png
```

front.micro16.txtTXT

所以 0004、0005、0008、0009、0010 本身不是被 stitch 取代，而是五張 Raw 都會保留。

---

# 3. 那最後的 Front Stitched Image 存在哪裡？

這裡目前能確認到「directory 層級」，但**還不能確認最後 filename**。

因為 `main.py` 呼叫 `handle_front_stitch_asset()` 時，有明確把：

```
analysis_dir=analysis_dir
```

傳進去。

而目前 App analysis directory 的實際形式是：

```
D:\Moonlight\ImagingLibWatch\
Local_Data\
run_<WatchID>\
Analysis\
Exp_<timestamp>_<id>\
```

例如實際 log 中：

```
D:\Moonlight\ImagingLibWatch\
Local_Data\
run_16613T(V3)_1781193553\
Analysis\
Exp_20260611_085913_8e77bfa2
```

因此目前最有根據的判斷是：

```
Local_Data
└─ run_<watch>
   ├─ Raw
   │   ├─ 0004 source image
   │   ├─ 0005 source image
   │   ├─ 0008 source image
   │   ├─ 0009 source image
   │   └─ 0010 source image
   │
   └─ Analysis
       └─ Exp_xxxxxxxxx
           └─ [Front Stitch output]
```

**但我現在不會猜它叫 `front_stitched.png`、`stitched_front.png` 或其他名字。**

因為真正決定：

- filename
- 是否有子目錄
- 是否另存 stitch report JSON
- 是否另存 transformation matrices
- 是否同步 S3
- DB task name

的是目前搜尋不到內容的 `handle_front_stitch_asset()` / `server_front_stitch.py` / `front_stitch_config.yaml`。

一般 analysis output 會掃描 `analysis_dir` 中新產生的檔案，再註冊 Local DB / S3；一般 Analysis 的 S3 結構則是：

```
<DeviceID>/<WatchID>/Analysis/<ExpID>/<filename>
```

main(6).pyPY

但 Front Stitch aggregator 是否完全走這個相同 registration path，我需要看到它自己的 implementation 才能確定。

---

# 4. 我確實找到你記得的 U-Net code

這部分你的記憶不是空穴來風。

目前 Library 裡確實存在完整的 semantic segmentation U-Net 設定：

```
project_name: semantic_segmentation_unet

data:
    img_size: 256
    num_classes: 17
```

也就是：

**Class 0 = Background + 16 個 object classes。**

Model 是：

```
U-Net
encoder = efficientnet-b0
```

而且有舊 checkpoint：

```
checkpoints/unet_v2.pth
```

和 fine-tuned model：

```
checkpoints/unet_model_finetuned.pth
```

unet_config.yamlYAML

---

# 5. 而且不只是 training code，也真的有 inference code

`inference_unet.py` 有完整的：

```
class UnetInference:
```

它建立：

```
smp.Unet(
    encoder_name=encoder_name,
    classes=self.num_classes,
    activation=None
)
```

並載入訓練好的 checkpoint，然後切成 `eval()` inference mode。

更重要的是，它不是只輸出一張彩色 segmentation。

它還會針對每個 segmentation class 個別輸出 binary mask：

```
per_class_masks/
    <image_name>/
        <image_name>_class_1.png
        <image_name>_class_2.png
        <image_name>_class_3.png
        ...
```

實際 code 就是：

```
for cls_id in unique_classes:
    if cls_id == 0:
        continue

    binary_mask = np.where(
        pred_mask == cls_id,
        255,
        0
    )

    save_name = class_folder / \
        f"{stem}_class_{cls_id}.png"
```

所以確實曾經有一套：

> image → U-Net → multi-class segmentation → 每一個 component individual mask

的設計。

---

# 6. 但我現在找到的這個 U-Net inference 是「standalone」，沒有接到 Front Stitch

這是這次搜尋最關鍵的發現。

`inference_unet.py` 的執行入口目前是：

```
INPUT_IMAGES_DIR =
"D:\Provenance Laboratories projects\...\inference_test"

CONFIG_PATH = "configs/unet_config.yaml"

engine = UnetInference(CONFIG_PATH)
engine.run_folder(INPUT_IMAGES_DIR)
```

也就是它目前像是一個：

```
任意 input folder
       ↓
UnetInference
       ↓
Segmentation
       ↓
per_class_masks
```

的 standalone inference utility。

我沒有找到：

```
handle_front_stitch_asset()
       ↓
stitched_path
       ↓
UnetInference(stitched_path)
```

這條接線。

---

# 7. 更重要的是：目前 production `system_config.yaml` 沒有把 Stitched Image 接到 U-Net

我直接搜尋目前 production service routing。

`system_config.yaml` 裡甚至沒有一個：

```
stitched
```

selector。

也沒有 service 名字直接叫：

```
segmentation_service
```

而 `front_stitch_service` 自己目前是：

```
front_stitch_service:
    script: server_front_stitch.py
    config_path: front_stitch_config.yaml
    aggregator: true
    suppress_single_image_analysis: true
```

**它沒有：**

```
model_name: micro_unet
```

這和其他真正用 U-Net 的 service 很不一樣。

---

# 8. 現在 production 裡其實有大量 U-Net，但都是分析「局部 Micro image」

例如目前 production config 裡有：

|Service|U-Net|分析哪張圖|
|---|---|---|
|`features_letter_service`|`micro_unet`|`Front.micropoint1`, `micropoint3`|
|`features_crown_service`|`micro_unet`|`Front.micropoint2`|
|`features_marker_service`|`micro_unet`|`Front.micropoint4`, `micropoint5`|
|`side_crown_service`|`micro_unet`|`Front.macropoint3`|
|`lume_hour_shape_service`|`micro_unet`|`Front.micropoint13`|
|`lume_hour_texture_service`|`micro_unet`|`Front.micropoint13.hdl_1`|
|`lume_hand_shape_service`|`micro_unet`|`Front.micropoint14`, `micropoint15`|
|`lume_hand_texture_service`|`micro_unet`|`Front.micropoint14.hdl_1`, `micropoint15.hdl_1`|
|`diallume_shape_service`|`micro_unet`|`Front.micropoint12`|
|`diallume_texture_service`|`micro_unet`|`Front.micropoint12.hdl_1`|

前三組設定可直接看到 `model_name: micro_unet`。

Hour/Hand/Lume 那幾組也全部明確指定 `micro_unet`。system_config.yamlYAML

所以現在 production 架構是：

```
Micropoint image
      ↓
micro_unet
      ↓
segmentation
      ↓
feature-specific analysis
```

而不是目前能查到的：

```
Front Stitched Image
      ↓
full-face U-Net
      ↓
Hour Hand
Minute Hand
Second Hand
Hour Markers
...
```

---

# 9. 我也搜尋了你特別提到的 Hour / Minute / Markers class mapping

這裡有一個問題：

目前找到的 `unet_config.yaml` 只知道：

```
num_classes = 17
```

但沒有 class semantic names。

而 `inference_unet.py` 自己輸出的 label description 還只是：

```
desc = "Background" if i == 0 else f"Object Class {i}"
```

也就是：

```
0 = Background
1 = Object Class 1
2 = Object Class 2
...
16 = Object Class 16
```

沒有在這支 inference code 裡寫：

```
1 = Hour Hand
2 = Minute Hand
3 = Second Hand
4 = Hour Markers
...
```

inference_unet.pyPY

我也直接搜尋了：

- `Hour`
- `Minute`
- `Marker`
- `Second`

在目前可取得的 dataset labeling code 中，都沒有找到對應的 semantic class definition。 dataset_labeling_ui.pyPY

所以我目前還不能說：

> 「Class 3 就是 Hour Hand、Class 4 就是 Minute Hand。」

那會變成猜測。

---

# 10. 但你記得的「Hour Markers」確實和目前 Authentication design 很吻合

目前 Authentication specification 裡，Dial evidence group 已經明確列到：

```
Dial
 ├─ lower_text_typography
 ├─ lower_text_geometry
 ├─ crown_logo
 ├─ hour_markers
 ├─ date_window
 ├─ printing
 ├─ color
 └─ texture
```

Moonlight_Authentication_Phase3_Hierarchical_Bayesian_Fusion.docxDOCX

不過這只能證明現在 Authentication 層期待 `hour_markers` feature，**不能證明它現在是由 stitched-image U-Net 產生的**。

---

# 11. 我目前對你記得的舊流程的判斷

我認為很可能以前確實有兩套東西：

```
A. WatchSegAI / full-face segmentation
-------------------------------------
Full watch/dial image
        ↓
17-class U-Net
        ↓
Hour Hand
Minute Hand
Second Hand
Hour Markers
Dial / bezel / ...
        ↓
Geometry / shape / spatial analysis
```

以及現在 App production 演變成：

```
B. ImagingLibWatch production tasks
------------------------------------
0004 + 0005 + 0008 + 0009 + 0010
                ↓
       front_stitch_service
                ↓
        Front Stitched Image


Micropoint 1
    ↓
Letter micro_unet

Micropoint 2
    ↓
Crown micro_unet

Micropoint 4/5
    ↓
Marker micro_unet

Micropoint 13
    ↓
Hour lume micro_unet

Micropoint 14/15
    ↓
Hand lume micro_unet
```

也就是**full-face UNet 的 code 還在，但目前 production routing 看起來已經沒有把它接在 stitched image 後面。**

這很可能就是為什麼你會很明確記得「以前有」，但現在 `system_config.yaml` 看不到。

---

# 12. 還有一個很值得注意的版本線索

我檢查的歷史 `log.txt` 裡，Front 一共拍完 **32 images** 後就直接進入 upload / pipeline finished：

```
UI load complete, captured 32 images
...
Starting Local Save & AWS Upload Sequence
...
PIPELINE FINISHED: FRONT
```

而我對這份 log 搜尋：

```
FrontStitch
stitched
front_stitch
```

三個都是 0 match。 log.txtTXT

所以這份較早的 run log **沒有實際跑出現在這套 Front Stitch integration**。

相對地，目前較新的 `main(6).py` 已經有：

```
handle_front_stitch_asset()
```

這表示 Front Stitch integration 很可能是後來才加回 App pipeline 的。

---

## 最終核實結果

|問題|目前核實結果|
|---|---|
|0004/0005/0008/0009/0010 是否屬於 stitch set？|**是**|
|拍攝後是否自動送入 stitch？|**是**|
|是否需要使用者另外按 Stitch？|**不需要**|
|是否等五張全部完成才產生 final stitched image？|**架構明確如此設計，但真正 gate 在目前缺失的 `handle_front_stitch_asset()`，尚不能 100% 看實作確認**|
|五張原圖位置|`Local_Data\<run>\Raw\...`|
|Stitched image 的 destination|`analysis_dir` 被傳給 stitch service，應位於該 run 的 `Analysis\Exp_...` 範圍；**確切檔名/子目錄尚無法確認**|
|有沒有 U-Net code？|**有，而且是完整 17-class semantic U-Net**|
|有沒有 U-Net inference？|**有**|
|能不能產生 per-class segmentation masks？|**可以**|
|現在有沒有證據顯示 stitched image 自動送進這個 U-Net？|**目前沒有找到**|
|現在 production 有沒有使用 U-Net？|**有很多，但目前主要綁定各 Front micropoint**|
|找到 Hour Hand / Minute Hand / Hour Marker 的 17-class ID mapping 嗎？|**目前沒找到**|
|你記得的 stitched-face segmentation code 是否可能真的存在過？|**很可能；目前找到 U-Net 基礎 code，但缺少當年 stitched→U-Net 的 glue/config/class map**|

### 現在最值得檢查的 4 個實際 repo 檔案

如果要把最後這一塊完全釘死，真正關鍵不是再看 `main.py`，而是你目前本機 repo 裡的：

```
config/front_stitch_config.yaml
tasks/api_servers/server_front_stitch.py
tasks/cli_wrappers/run_front_stitch_cli.py
Controller/.../workflow_manager.py
    └─ handle_front_stitch_asset()
```

尤其是 **`handle_front_stitch_asset()`**。這一個 function 可以一次回答：

> 第五張到達時怎麼判斷完成 → 怎麼 stitch → stitched image 實際存哪個 filename → 有沒有接 U-Net → 有沒有註冊 DB/S3 → 後續 task 是什麼。

目前這幾個檔案沒有出現在我能搜尋的 Library 內容中，所以我不會編造那最後一段實作。就目前查到的證據來說，**「自動 stitch」已經存在；「stitched image → full-face 17-class U-Net segmentation」則有舊 U-Net infrastructure，但目前沒有在 production task routing 中接起來。**