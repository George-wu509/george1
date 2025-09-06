
![[Pasted image 20250904170759.png]]


https://www.linkedin.com/feed/?trk=nav_logo

https://mlhonk.substack.com/p/36-dift-diffusion-features

Image generation was just the beginning.  
  
Diffusion models can do way more than you think! 😎  
  
Turns out if you take a pair of images, encode them into latent space, and simulate a denoising step with a diffusion U-Net, the intermediate noisy features are so semantically rich that you can use them to find pixel-perfect matches across images with zero extra training.  
  
These correspondences are key for tasks such as tracking, 3D reconstruction, and more abstract semantic alignment. All it takes is a forward pass and nearest neighbor matching, no fancy machine learning on top!  
  
More broadly, this behavior isn’t unique to diffusion models. It is also typical of strong modern vision backbones like DINO and CLIP, with better results the more recent the training (say hello to our new friend DINOv3).  
  
In this week's article, we break down DIFT, making the case for diffusion-based image generators. Link in the comments, enjoy!