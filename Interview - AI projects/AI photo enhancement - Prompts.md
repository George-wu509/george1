
---
- 1. **Denoise（去噪）**



---
- **2. Sharpen（銳化）**



---
- **3. Image Inpainting（圖像修補）**



---
- 4. **Deblur（去模糊）**

Ref: https://www.cnblogs.com/bossma/p/17559785.html
prompt: 
ultra detailed, masterpiece, best quality, an photo of a old man in a hat and a hat on his heads, with greying temples, (looking at viewer), a character portrait, mingei,simple background, clean

negative prompt: 
easy_negative, NSFW, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, skin spots, acnes, skin blemishes,age spot, (ugly:1.331), (duplicate:1.331),(morbid:1.21), (mutilated:1.21), (tranny:1.331),flower,lines,white point,plant,



---
- **5. Super-Resolution（超分辨率）**



---
- **6. Fix White Balance（修正白平衡）**



---
- **7. Apply to HDR（應用高動態範圍處理）**




---
- **8. Improve Image Quality（提升圖像質量）**



---
- **9. Colorize（圖像上色）**

ref: https://www.cnblogs.com/bossma/p/17720726.html



---
- **10. Correct Light（光線校正)**



---
- **11. Remove Multi Object（移除多物體）**



---
- **12. Reposition Multi Object（重新定位多物體）**



---
- **13. Add Multi Object（新增多物體）**



---
- **14. Remove and Regenerate Background**


---









#### 用Image J製作Mask
Step1. 如果圖像是彩色的，轉換為灰度圖像
	Image > Type > 8-bit
Step2. 使用選區工具選擇Mask區域: 矩形選區, 自由選區, 魔棒工具
Step3. 生成二值化遮罩
     Edit > Fill   將選中區域填充為白色, 未選中的區域會保留為黑色
Step3b. 反轉選區（如果需要）
	 如果要選中非感興趣區域，點擊 `Edit > Invert Selection
Step3c. 自動生成遮罩（可選）
	 調整閾值: Image > Adjust > Threshold
	 使用滑塊選擇目標區域
	 點擊 `Apply` 應用閾值
	 如果需要平滑或填補細節，使用 `Process > Binary > Options`
Step4. 處理完成的遮罩保存為新的圖像
	File > Save As > PNG

#### Model File要放在哪裡
<基礎模型>
**<下載>**
基礎模型通常可以從以下兩個可靠來源下載
打開 [Hugging Face](https://huggingface.co/)
搜索 CompVis/stable-diffusion-v-1-4 或 stabilityai/stable-diffusion-2-1-base。
**<位置>**
基礎模型 - control_sd15_canny.pth, control_sd15_depth.pth
\webui\models\stable-diffusion\
**<設定>**
運行 WebUI，等待載入完成, 在 WebUI 的「Settings」或「Model」菜單中，找到模型選擇器。從下拉菜單中選擇剛添加的模型（如 `stable-diffusion-v1-5` 或 `stable-diffusion-v2-1`）。選定模型後，點擊「Apply Settings」以應用配置。



**<位置>**
ControlNet模型 - control_sd15_canny.pth, control_sd15_depth.pth
\webui\extensions\sd-webui-controlnet\models\



