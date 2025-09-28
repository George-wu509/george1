
linkedin post [link](https://lnkd.in/p/g6kFTYax)

![[Pasted image 20250919013702.png]]


Zero-Shot Depth That Actually Generalizes? Meet FE2E.  
  
I’ve been testing FE2E (From Editor to Dense Geometry Estimator) on mixed indoor/outdoor scenes—see the short demo below.  
  
The core idea is simple but powerful: for dense geometry, start from an image editor (DiT) rather than a text-to-image generator. Editors carry stronger structural priors, so fine-tuning converges cleaner and predicts sharper depth & normals.  
  
Why this is interesting (for practitioners):  
Editor → Estimator: They adapt a DiT editing model for monocular depth + normals in one forward pass. arXiv  
Deterministic training: Reformulates the editor’s flow-matching into a “consistent velocity” objective—no timestep dependence, fewer quirks.  
Stability trick: Log-depth quantization to play nicely with bfloat16 while preserving metric depth fidelity.  
Results (reported):  
Zero-shot SOTA-style gains across standard depth/normal benches.  
>35% improvement on ETH3D vs. prior methods, outperforming DepthAnything families trained on ~100× more data—with ~71K training images.  
  
If you ship AR/robotics or large-scale 3-D perception, FE2E’s recipe—editing-model priors + consistent-velocity loss + joint heads—is a practical path to robust zero-shot geometry without gargantuan data/compute.  
  
Link to the paper : [https://lnkd.in/d6KKpqM2](https://lnkd.in/d6KKpqM2)  
  
Video: My quick reel of FE2E depth on varied scenes (indoor clutter, glass, foliage, low-light).


  
Link to the code and the demo : [https://amap-ml.github.io/FE2E/](https://amap-ml.github.io/FE2E/)