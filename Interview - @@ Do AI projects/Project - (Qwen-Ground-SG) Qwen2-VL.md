
```
想對Project作一些更新跟補充. 這個Project是基於Qwen2-VL跟Grounded SAM跟其他需要的libraries. 輸入可以是一張image or 一段video. 這個package可以實現圖像/影片描述, 視覺問答 (VQA), 多模態推理, 視覺定位 (Visual Grounding), 影片描述等基本功能, 也可以基於user輸入的text進行image or video上多目標多物體的object detection, segmentation跟tracking. 也可以對image or viode上可根據text的object做場景圖生成 (Scene Graph Generation), 或者對image or viode上所有object做場景圖生成 (Scene Graph Generation). 並將這Scene Graph可以輸入到Qwen2-VL以提升圖像/影片描述, 視覺問答 (VQA), 多模態推理, 視覺定位 (Visual Grounding), 影片描述的回答準確度. 如果有基於得Scene Graph Generation結果可以做額外的分析等也可以加入functions. 希望這是放在Github做成一個python library. 也請提供github readme, 以及安裝python environment的方便方法譬如y
```

Qwen-Ground-SG github
https://github.com/George-wu509/Qwen-Ground-SG

Resume:
**Vision-language model based Visual Understanding and Scene Graph Generation Framework:**

Developed a Visual understanding library integrating a Vision Language Model (Qwen2-VL) with Grounded SAM to generate dynamic Scene Graphs from images and videos. This system enriches the VLM's context, significantly improving performance in downstream tasks like VQA, complex reasoning, and multi-object tracking by providing structured semantic understanding of the scene.




My Qwen2-VL [Colab](https://colab.research.google.com/drive/1Zahrn91uzsndMvaLefk8xQot4qsAQgIS?usp=sharing#scrollTo=N-kIKVdhxczd)   Qwen2-VL: Expert Vision Language Model for Video Analysis and Q&A.ipynb

Original Qwen2-VL [Colab](https://colab.research.google.com/drive/1Zahrn91uzsndMvaLefk8xQot4qsAQgIS?usp=sharing)

My Github:  Qwen-Ground-SG: A Multi-Modal Library for Enhanced Visual Understanding
https://github.com/George-wu509/Qwen-Ground-SG#




|                         |     |
| ----------------------- | --- |
| 圖像/影片描述                 |     |
| 視覺問答 (VQA)              |     |
| 多模態推理                   |     |
| 視覺定位 (Visual Grounding) |     |
| 物件偵測/分割                 |     |
| 物件追蹤 (Object Tracking)  |     |
| 影片描述                    |     |
|                         |     |
|                         |     |

推薦的目錄結構如下：

```
visiongraphx/
├── pyproject.toml        # 專案元資料與建置設定
├── README.md             # 專案說明文件
├── environment.yaml      # Conda 環境定義檔
├── docs/                 # 詳細文件（例如 Sphinx）
├── examples/             # 使用範例腳本
├── src/
│   └── visiongraphx/
│       ├── __init__.py
│       ├── pipeline.py     # 主要的用戶介面 API
│       ├── core/           # 核心資料結構 (SceneGraph, TrackedObject 等)
│       ├── models/         # 對底層模型的封裝
│       │   ├── vlm.py      # Qwen2.5-VL 封裝
│       │   ├── perception.py # Grounded-SAM-2 封裝
│       │   └── tracking.py   # BoT-SORT 封裝
│       ├── processing/     # 媒體載入與預處理
│       │   └── loader.py
│       └── utils/          # 輔助函數，例如序列化
└── tests/                  # 單元測試與整合測試
```