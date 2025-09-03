
```
請幫我詳細規劃以下的研究計畫. 這個Project是基於Stable Diffusion跟ControlNet跟其他需要的libraries. 輸入是一張image, 希望設計成處理相機照片. 可以以prompt控制對這張image進行Denoise（去噪）, Sharpen（銳化）, Image Inpainting（圖像修補）, Dehaze（除霧）, Deblur（去模糊）, Super-Resolution（超解析）, Fix White Balance（修正白平衡）, Apply to HDR（生成HDR效果）,Improve Image Quality（提升圖像質量）,Colorize（圖像上色）,Correct Light（光照校正）,Remove, Reposition, or Add Object, Remove and Generate Background. 風格改變, 可以做到像相機變焦x100. 
```

ImageAlchemy Github:
https://github.com/George-wu509/ImageAlchemy

Resume:
**ImageAlchemy: Prompt-Driven Image Enhancement and Generative Super Resolution:**

Engineered a Python library using Stable Diffusion, ControlNet, and SAM to provide a unified, high-level API for complex image editing. Key functionalities include prompt-driven super resolution, AI inpainting, and precise object manipulation, all built upon a modular architecture with PyTorch and the Hugging Face Diffusers library.