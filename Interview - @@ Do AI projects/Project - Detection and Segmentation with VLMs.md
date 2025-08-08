
post:
https://www.linkedin.com/posts/skalskip92_opensource-computervision-objectdetection-activity-7351205050318303232-i3lz?utm_source=share&utm_medium=member_desktop&rcm=ACoAABMNj2MBAJSP3cWd4xpiz4wB7qdx43hvW18


![[Pasted image 20250807001551.png]]

using supervision-0.26.0

```
Detection and Segmentation with VLMs 🔥🔥🔥  
  
TL;DR: VLMs are making real progress in detection and segmentation, and supervision-0.26.0 lets you easily parse and visualize their prediction results. Check the comments for links to ready-to-use notebooks and HF Spaces with real examples. 👇🏻  
  
- We added support for parsing and visualizing results from [Alibaba Cloud](https://www.linkedin.com/company/alibaba-cloud-computing-company/) Qwen2.5-VL, [Moondream AI](https://www.linkedin.com/company/moondream-ai/), and both [Google DeepMind](https://www.linkedin.com/company/googledeepmind/) Gemini 2.0 and 2.5.  
- Improved LabelAnnotator with a smart mode where labels never overlap and always stay visible. Added support for text wrapping and line breaks, so even long text prompts are displayed clearly.  
  
- Added support for pose estimation models in [Hugging Face](https://www.linkedin.com/company/huggingface/) transformers, so you can now visualize ViTPose and ViTPose++ skeletons directly in supervision.  
- Rewrote the mean average precision implementation. Results now match pycocotools, and the interface is much simpler and easier to use for benchmarking any dataset or model.  
  
⮑ 🔗 supervision-0.26.0 release notes: [https://lnkd.in/dJg7YdyK](https://lnkd.in/dJg7YdyK)
```