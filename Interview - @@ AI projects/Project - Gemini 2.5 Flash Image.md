
linkedin post: https://www.linkedin.com/posts/massimiliano-viola_everybody-is-literally-going-bananas-about-activity-7371575588974686208-FJ5B?utm_source=share&utm_medium=member_desktop&rcm=ACoAABMNj2MBAJSP3cWd4xpiz4wB7qdx43hvW18

![[Pasted image 20250910123734.png]]

Everybody is literally going bananas about image editing these days! 🍌  
  
Relatable, as Google's new Gemini 2.5 Flash Image is just so good. And so are many other models like FLUX Kontext, GPT Image, and Qwen Image Edit.  
  
But as always with deep learning, the secret sauce lies in the data, and frontier labs don’t seem to disclose what goes into these models in their technical reports.  
  
So what does it take to train an image editing model? How do we get triples of an image, a text instruction describing the edit, and the corresponding modified image? 🤔  
  
I asked myself these questions recently, and found a paper reproducing an image editing model in an open-source manner. Its data pipeline is really fascinating and consists of a whole suite of specialized tools working together, including SAM 2, ControlNet with Stable Diffusion 3.5, various VLMs, and popular inpainting models.  
  
It's a bit of a longer read with lots of models mentioned, but surely satisfying if you were also wondering how Nano Banana pulls off some of its magic. Full breakdown in the blog post in the comments, enjoy!


How to generate a dataset for text-guided image editing!  
https://mlhonk.substack.com/p/37-image-editing-with-step1x-edit