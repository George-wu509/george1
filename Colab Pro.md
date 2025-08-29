
# 充分發揮 Colab 訂閱的價值

---

## 更快速的 GPU

使用者購買任一種 Colab 付費方案後，就能使用更快的 GPU 和更多記憶體。如要升級筆記本的 GPU 設定，請在選單中依序點選「執行階段」>「變更執行階段類型」，然後從加速器選項中選取要使用的加速器 (實際情況視可用性而定)。

如果使用的是免付費版 Colab，則會依據配額限制和可用性授予 Nvidia T4 GPU 的存取權。

你隨時可以執行下列儲存格，瞭解目前系統指派給你的 GPU。如果下方程式碼儲存格的執行結果為「Not connected to a GPU」，你可以變更執行階段。請在選單中依序按一下「執行階段」>「變更執行階段類型」以啟用 GPU 加速器，然後重新執行程式碼儲存格。

---

[ ]

gpu_info = !nvidia-smi  
gpu_info = '\n'.join(gpu_info)  
if gpu_info.find('failed') >= 0:  
  print('Not connected to a GPU')  
else:  
  print(gpu_info)  

---

如要透過筆記本使用 GPU，請依序選取「執行階段」>「變更執行階段類型」選單，將選項設為「硬體加速器」。

---

## 更多記憶體

使用者購買任一種 Colab 付費方案後，就能存取大量記憶體的 VM (如果可用)。具備大量記憶體的 VM 一律會搭載效能更強大的 GPU。 你隨時可以執行下列程式碼儲存格，查看可用記憶體容量。如果下方程式碼儲存格的執行結果為「Not using a high-RAM runtime」，你可以啟用大量 RAM 執行階段，步驟如下：前往選單，依序點選「執行階段」>「變更執行階段類型」。在「執行階段規格」切換鈕中選取「大量 RAM」，然後重新執行程式碼儲存格。

---

[ ]

from psutil import virtual_memory  
ram_gb = virtual_memory().total / 1e9  
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))  
  
if ram_gb < 20:  
  print('Not using a high-RAM runtime')  
else:  
  print('You are using a high-RAM runtime!')  

---

## 較長時間的執行階段

所有 Colab 執行階段都會在一段時間後重設，如果執行階段並未執行程式碼，則會更快重設。與使用免付費版 Colab 的使用者相比，Colab Pro 和 Pro+ 使用者可以使用更長的執行階段。

## 背景執行

Colab Pro+ 為使用者提供背景執行功能，即使瀏覽器分頁已關閉，筆記本仍會持續執行。只要有可用的運算單元，Pro+ 執行階段一律會啟用這項功能。

---

## 在 Colab Pro 中放寬資源限制

你在 Colab 中的資源設有上限。如要讓 Colab 發揮最大效益，請避免在非必要時使用資源。舉例來說，請只在必要時才使用 GPU，並在使用完畢後關閉 Colab 分頁。

如果用量已達上限，你可以透過即付即用購買更多運算單元，以放寬這些限制。任何人都可以透過[即付即用](https://colab.research.google.com/signup)購買運算單元，不必訂閱方案。

---

## 歡迎提供意見！

歡迎與我們分享你的寶貴意見，只要依序按一下「說明」>「提供意見...」選單，即可提供意見。如果 Colab Pro 的用量已達上限，建議你訂閱 Pro+。

如果 Colab Pro、Pro+ 或 Pay As You Go 的帳單 (付款) 發生錯誤或其他問題，請傳送電子郵件至 [colab-billing@google.com](mailto:colab-billing@google.com)。

---

## 其他資源

### 在 Colab 中使用筆記本

- [Colab 總覽](https://colab.research.google.com/notebooks/basic_features_overview.ipynb)
- [Markdown 指南](https://colab.research.google.com/notebooks/markdown_guide.ipynb)
- [匯入程式庫及安裝依附元件](https://colab.research.google.com/notebooks/snippets/importing_libraries.ipynb)
- [儲存和載入 GitHub 中的筆記本](https://colab.research.google.com/github/googlecolab/colabtools/blob/main/notebooks/colab-github-demo.ipynb)
- [互動式表單](https://colab.research.google.com/notebooks/forms.ipynb)
- [互動式小工具](https://colab.research.google.com/notebooks/widgets.ipynb)

### 處理資料

- [載入資料：雲端硬碟、試算表及 Google Cloud Storage](https://colab.research.google.com/notebooks/io.ipynb)
- [圖表：將資料視覺化](https://colab.research.google.com/notebooks/charts.ipynb)
- [開始使用 BigQuery](https://colab.research.google.com/notebooks/bigquery.ipynb)

### 機器學習密集課程

以下是一些 Google 線上機器學習課程的筆記本。詳情請參閱[完整的課程網站](https://developers.google.com/machine-learning/crash-course/)。

- [Pandas DataFrame 簡介](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/pandas_dataframe_ultraquick_tutorial.ipynb)
- [以 tf.keras 使用合成資料進行線性迴歸](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/linear_regression_with_synthetic_data.ipynb)

### 使用加速硬體

- [搭配 GPU 使用 TensorFlow](https://colab.research.google.com/notebooks/gpu.ipynb)
- [Colab 中的 TPU](https://colab.research.google.com/notebooks/tpu.ipynb)

---

## 機器學習範例

如要查看 Colab 的互動式機器學習分析端對端範例，請參閱這些使用 [TensorFlow Hub](https://www.google.com/url?q=https%3A%2F%2Ftfhub.dev) 模型的教學課程。

一些精選範例如下：

- [重新訓練圖片分類工具](https://tensorflow.org/hub/tutorials/tf2_image_retraining)：以預先訓練的圖片分類工具為基礎，建立一個分辨花朵的 Keras 模型。
- [文字分類](https://tensorflow.org/hub/tutorials/tf2_text_classification)：將 IMDB 電影評論分類為_正面_或_負面_。
- [風格轉換](https://tensorflow.org/hub/tutorials/tf2_arbitrary_image_stylization)：運用深度學習轉換圖片的風格。
- [支援多種語言的 Universal Sentence Encoder 問與答](https://tensorflow.org/hub/tutorials/retrieval_with_tf_hub_universal_encoder_qa)：使用機器學習模型來回答 SQuAD 資料集的問題。
- [影片畫面內插](https://tensorflow.org/hub/tutorials/tweening_conv3d)：預測影片在第一個與最後一個畫面之間的內容。

---