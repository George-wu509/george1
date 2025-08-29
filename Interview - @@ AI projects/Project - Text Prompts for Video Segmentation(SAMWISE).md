
post: https://www.linkedin.com/posts/skalskip92_cvpr2025-computervision-deeplearning-activity-7338900951736406016-EvoB?utm_source=share&utm_medium=member_desktop&rcm=ACoAABMNj2MBAJSP3cWd4xpiz4wB7qdx43hvW18

![[Pasted image 20250807001904.png]]
SAMWISE: Text Prompts for Video Segmentation 🔥 🔥 🔥  

![[Pasted image 20250807002037.png]]


HF space showing how to combo SAM2 with VLMs: [https://huggingface.co/spaces/SkalskiP/florence-sam](https://huggingface.co/spaces/SkalskiP/florence-sam)


```
SAM2 is great at video object tracking using visual prompts, but it does not understand text. In my demos, I often showed how combining SAM2 with VLMs enabled language-guided image segmentation, but SAMWISE takes this even further by allowing direct text-driven video object segmentation without large VLMs or extra fine-tuning.  
  
Now you can segment and track objects in videos just by describing them, for example "the person in red who enters from the left." The model follows your prompt, tracks the right object as it appears, and can even fix its own tracking mistakes.  
  
SAMWISE achieves this by adding a lightweight adapter that lets SAM2 combine visual and language information and model changes over time. It can recognize when it is tracking the wrong object and automatically switch focus to the correct one, even during occlusions or complex motion. All this works efficiently without retraining SAM2 or using external models, making it practical for real-world, long video scenarios.  
  
⮑ 🔗 top CVPR 2025 papers: [https://lnkd.in/dbNRXHW2](https://lnkd.in/dbNRXHW2)
```
