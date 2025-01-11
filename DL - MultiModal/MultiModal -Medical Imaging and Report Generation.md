
https://www.gehealthcare.com/insights/article/latest-advances-in-research-building-a-multimodal-xray-foundation-model

![[materials/1.jpg]]# Building a multimodal X-ray foundation model

To address these challenges, we are pioneering the development of foundation models in healthcare. These sophisticated AI systems are fine-tuned for healthcare datasets, with the goal of enabling superior performance and adaptability across diverse applications. If fully developed, multimodal medical LLMs could become the foundation for new assistive technologies in professional medicine, medical research, and consumer applications. As with our past initiatives, we emphasize the critical need for a comprehensive evaluation of these technologies in collaboration with the medical community and the broader healthcare ecosystem
為了應對這些挑戰，我們正在率先開發醫療保健領域的基礎模型。這些複雜的人工智慧系統針對醫療保健資料集進行了微調，其目標是在不同的應用程式中實現卓越的效能和適應性。如果充分發展，多模式醫學法學碩士可以成為專業醫學、醫學研究和消費者應用領域新輔助技術的基礎。與我們過去的舉措一樣，我們強調迫切需要與醫學界和更廣泛的醫療保健生態系統合作對這些技術進行全面評估

In internal testing, our full-body X-ray model stands out even with limited training data or when faced with out-of-domain challenges, showcasing its robustness and generalizability. Remarkably, when we fine-tuned the model using only chest-specific training data, it still showed significant improvements in non-chest-related tasks, such as anatomy detection and lead marker detection, outperforming in our experiments existing chest-specialized pre-trained models.
在內部測試中，即使訓練資料有限或面臨域外挑戰，我們的全身 X 射線模型也能脫穎而出，展現了其穩健性和通用性。值得注意的是，當我們僅使用胸部特定的訓練資料對模型進行微調時，它在與胸部無關的任務（例如解剖檢測和先導標記檢測）上仍然顯示出顯著的改進，在我們的實驗中優於現有的胸部專用預訓練模型。

Our ongoing research is focused on rigorously comparing our X-ray model against the latest publicly available models. The initial results are compelling, suggesting the X-ray model could perform critical tasks like segmentation, classification, and visual localization with high accuracy.
們正在進行的研究重點是將我們的 X 光模型與最新的公開模型進行嚴格比較。初步結果令人信服，顯示 X 光模型可以高精度地執行分割、分類和視覺定位等關鍵任務。

Today’s announcement builds on GE HealthCare’s [pioneering research in healthcare foundation models](https://www.gehealthcare.com/insights/article/sonosamtrack-a-pioneering-research-analysis-of-ultrasound-imaging-with-ai). Our ultrasound research model, SonoSAMTrack combines a promptable foundation model for segmenting objects of interest on ultrasound images with a state-of-the-art contour tracking model.
今天的公告建立在 GE HealthCare[在醫療保健基礎模型方面的開創性研究](https://www.gehealthcare.com/insights/article/sonosamtrack-a-pioneering-research-analysis-of-ultrasound-imaging-with-ai)的基礎上。我們的超音波研究模型 SonoSAMTrack 結合了用於分割超音波影像上感興趣物件的快速基礎模型和最先進的輪廓追蹤模型。


## Foundation Models in Action

We used a dataset of X-ray images and corresponding radiology reports sourced from various imaging sites and regions, expanding beyond the traditional focus on chest X-rays. This curated dataset encompasses a wide range of anatomies, providers, manufacturers, demographics, and pathologies, aiming to enhance the generalizability and applicability of our models across diverse clinical scenarios. The training process for our foundation model also uses licensed datasets that are anonymized and compliant with privacy and healthcare requirements.

Potential applications of the foundation model include:

1. **Report generation:** By fine-tuning our pre-trained models on specific datasets like IU-X-ray, we've achieved substantial gains in report quality compared to baseline models. Our full-body pre-trained model showed performance improvements on metrics that measure semantic similarity and summarization capability (e.g., CIDEr and ROUGE) when compared to other similar models. If successful, this advancement could streamline a process that was time-consuming and susceptible to human error, allowing clinicians to spend more time focused on patient care.
	報告產生：透過在 IU-X 射線等特定資料集上微調我們的預訓練模型，與基線模型相比，我們在報告品質方面取得了顯著的進步。與其他類似模型相比，我們的全身預訓練模型在測量語意相似性和摘要能力（例如 CIDEr 和 ROUGE）的指標上表現出了效能改進。如果成功，這項進展可以簡化一個耗時且容易出現人為錯誤的流程，使臨床醫生能夠將更多時間專注於病患照護。

2. **Classification:** We are evaluating our model's performance on a diverse set of downstream classification tasks, including disease diagnosis, gender identification, view detection, anatomical landmark localization, marker detection, and mirror reflection identification. Using a linear probing setup with a three-layer flat structure Multi-layer Perceptron (MLP) as the classifier head, initial results reveal that our model demonstrated significant improvements in mean average AUROC (mAUC) scores across various tasks.
	分類：我們正在評估我們的模型在各種下游分類任務上的性能，包括疾病診斷、性別識別、視圖檢測、解剖標誌定位、標記檢測和鏡面反射識別。使用具有三層平面結構多層感知器 (MLP) 的線性探測設定作為分類器頭，初步結果表明，我們的模型在各種任務中的平均 AUROC (mAUC) 分數方面表現出顯著改善。

3. **Grounding**: We are testing the utility of our pre-trained model on grounding tasks. This task involves locating the relevant region in a medical image corresponding to a textual phrase query. These  tasks can potentially enable grounded medical interpretations which are critical for the deployment of responsible and explainable AI.  Our model, when fine-tuned on a dataset specialized for the chest region, achieved significant improvements over both baseline and chest-specialized pre-trained models on the MiOU metric, which measures the overlap between the predicted segmentation area and the ground truth. This advancement could enhance the interpretability of AI insights, allowing clinicians to visually pinpoint the exact areas within an image that the model is focusing on, helping building trust in AI-driven outputs.
	接地：我們正在測試預訓練模型在接地任務上的實用性。此任務涉及在醫學影像中定位與文字短語查詢相對應的相關區域。這些任務有可能實現紮根的醫學解釋，這對於部署負責任且可解釋的人工智慧至關重要。我們的模型在專門針對胸部區域的資料集上進行微調後，在MiOU 指標上比基線和胸部專用預訓練模型取得了顯著的改進，MiOU 指標測量了預測分割區域和地面真實情況之間的重疊。這項進步可以增強人工智慧見解的可解釋性，使臨床醫生能夠直觀地找出模型所關注的影像中的確切區域，從而幫助建立對人工智慧驅動輸出的信任。