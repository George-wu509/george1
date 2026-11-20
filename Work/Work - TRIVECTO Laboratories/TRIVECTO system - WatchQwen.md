

|     |     |
| --- | --- |
|     |     |
|     |     |
|     |     |
|     |     |
|     |     |
|     |     |
|     |     |


```
我決定create一個新的repo WatchQwen, 目的就是測試Qwen3.5-VL或更新版本看他可以對我的watch image能達成甚麼功能譬如單張影像描述、VQA、影像中物件狀態的語意理解, OCR/文件理解、圖表/layout、物件定位、JSON/結構化輸出、多圖與影片理解等能力. 主要啟動檔包括cli以及一個UI版本, 在這個UI左邊區域有按鈕可以載入一張image, 以及有輸入輸出對話框可以跟model互動測試各功能的inference能力. 右邊區域則是顯示image. 如果任務有輸出image也會呈現在這裡. 我會自己建立repo並跟github同步, 請提供python file跟其他相關files給我就好, 我也會去下載checkpoint. 也請提供用conda建立python environment的檔案包括requirement及指令等等等
```


已經幫你做了一份可搬到新 repo 的 scaffold，放在：

WatchQwen

我用官方 Qwen3-VL/Hugging Face 資訊做預設：Qwen/Qwen3-VL-8B-Instruct。官方 model card/README 目前明確支援 AutoModelForImageTextToText、AutoProcessor，並建議 transformers>=4.57.0、qwen-vl-utils==0.0.14。參考：[![](https://huggingface.co/favicon.ico)HF model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)、[Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL)。

**主要檔案**

- README.md：setup、CLI/UI 使用方式
- environment.yml：Conda env
- requirements.txt：Qwen/UI dependencies
- cli.py：單圖、多圖、影片 CLI inference
- ui.py：PySide6 UI，左邊控制/輸入輸出，右邊顯示影像與 box overlay
- qwen_runner.py：共用 Qwen loader/inference wrapper
- presets.py：描述、VQA、OCR、JSON、定位等 prompt presets
- box_utils.py：解析 JSON bbox 並畫到 image 上

**建立環境**

powershell

`cd D:\YourNewRepo\WatchQwen conda env create -f environment.yml conda activate watchqwen pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 pip install -r requirements.txt`

**啟動 UI**

powershell

`python ui.py --model D:\models\Qwen3-VL-8B-Instruct`

**CLI 範例**

powershell

`python cli.py --model D:\models\Qwen3-VL-8B-Instruct --image D:\data\watch.png --preset watch_json python cli.py --model D:\models\Qwen3-VL-8B-Instruct --image front.png --image side.png --preset compare python cli.py --model D:\models\Qwen3-VL-8B-Instruct --video rotation.mp4 --prompt "Summarize the visible watch condition."`

我也跑了語法檢查：

powershell

`python -m py_compile WatchQwen\cli.py WatchQwen\ui.py WatchQwen\watchqwen\qwen_runner.py WatchQwen`