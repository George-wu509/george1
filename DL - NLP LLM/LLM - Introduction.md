
|                                                                    |     |
| ------------------------------------------------------------------ | --- |
| [[#### How Do LLMs Actually Work? ]]                               |     |
| [[#### Speed vs. Precision Memory vs. Semantics Embeddings Shape]] |     |
| [[#### EFT vs LoRA vs Other LLM Optimization Techniques]]          |     |



![[Pasted image 20250911111814.png]]




#### How Do LLMs Actually Work? 
  
LLMs like ChatGPT, Claude, and DeepSeek generate human-like text and power smart AI tools.  
  
But what’s going on under the hood?  
  
Let’s break it down step-by-step ⬇️  
  
  
[1.] Learning from Billions of Words  
  
LLMs are trained on massive datasets — including books, websites, code, and more.  
  
All that text is:  
◾ Cleaned and preprocessed  
◾ Split into tokens (small chunks of language the model can understand)  
  
[2.] Learning Word Relationships with Transformers  
  
LLMs use a neural network architecture called Transformers to understand word context and patterns.  
  
They get better over time using gradient descent, adjusting internal weights to reduce prediction errors.  
  
Basically: trial and error at scale.  
  
[3.] Specializing for Real-World Jobs  
  
Once trained, LLMs are fine-tuned for specific use cases like:  
◾ Coding assistance  
◾ Customer support  
◾ Knowledge work  
  
This happens through techniques like supervised learning,  
RLHF (Reinforcement Learning from Human Feedback) and LoRA (a lightweight fine-tuning method).  
  
[4.] Creating Smart, Contextual Output  
  
You type a prompt — the model:  
◾ Predicts the next most likely token  
◾ Builds a response from those predictions  
  
To improve factual accuracy, some LLMs use RAG (Retrieval-Augmented Generation) to pull info from external sources (docs, databases) first.  
  
Then decoding strategies like:  
◾ Beam search (explores multiple paths)  
◾ Nucleus sampling (adds randomness for creativity)  
help refine the output.  
  
  
[5.] Safer, Faster, Leaner Models  
  
Before going live, LLMs go through:  
◾ Safety filtering – to reduce bias or harmful output  
◾ Optimization – using quantization and pruning to make models faster and lighter, both in the cloud and on-device  
  
What are the challenges?  
  
LLMs are powerful, but not perfect:  
⚠️ Can hallucinate (make up facts)  
⚠️ May reflect bias  
⚠️ Requires massive compute power  
  
Engineers tackle these with:  
✔ RAG  
✔ Speculative decoding (guess-ahead to boost speed)  
✔ Hybrid edge/cloud deployment  
✔ Continuous safety tuning  
  
  
LLMs = extremely advanced pattern recognition machines — not sentient, but evolving fast.

Reference: linkedin post about LLM [link](https://www.linkedin.com/feed/update/urn:li:activity:7369685881617690625?updateEntityUrn=urn%3Ali%3Afs_updateV2%3A%28urn%3Ali%3Aactivity%3A7369685881617690625%2CFEED_DETAIL%2CEMPTY%2CDEFAULT%2Cfalse%29)





#### Speed vs. Precision Memory vs. Semantics Embeddings Shape

How we retrieve and reason over text in RAG. Here are 6 embedding types every retrieval-first system designer should know, and when to use them.  
  
1️⃣ Sparse Embeddings:  
High-dimensional vectors with numerous zeros, ideal for keyword-centric searches (reminiscent of BM25).  
Example: Indexing product titles for precise keyword matches like “waterproof hiking boots”.  
  
2️⃣ Dense Embeddings:  
Vectors with low-to-moderate dimensions and predominantly non-zero values, capturing semantic nuances.  
Example: Uncovering articles on “remote work culture” even with variations in exact wording.  
  
3️⃣ Quantized Embeddings:  
Dense vectors compressed to lower-precision types (e.g., float32 → int8) for memory efficiency and accelerated search processes.  
Example: Swift retrieval of semantic vectors in-memory on a compact instance for efficient customer support access.  
  
4️⃣ Binary Embeddings:  
Intense quantization to 0/1 bits for minimal memory usage and rapid bitwise comparisons.  
Example: Quick preliminary screening of documents in a vast index before a meticulous reranking procedure.  
  
5️⃣ Variable-Dimension Embeddings:  
Adaptable sizes or hierarchical encodings (reminiscent of matryoshka dolls) tailoring detail to the specific task at hand.  
Example: Employing small vectors for concise queries and nested/hierarchical vectors for in-depth context in lengthy documents.  
  
6️⃣ Multi-Vector Embeddings:  
Utilizing multiple vectors to represent a single item (e.g., token- or passage-level) for nuanced meaning capture.  
Example: Embracing ColBERT-style retrieval with one vector per token for precise matching of query tokens within extensive documents.  
  
Which of these embeddings aligns with your needs: latency, storage, or nuance? Share your thoughts below!

Reference linkedin [post:](https://www.linkedin.com/feed/update/urn:li:activity:7371470306265362432?updateEntityUrn=urn%3Ali%3Afs_updateV2%3A%28urn%3Ali%3Aactivity%3A7371470306265362432%2CFEED_DETAIL%2CEMPTY%2CDEFAULT%2Cfalse%29) 





#### EFT vs LoRA vs Other LLM Optimization Techniques

Mathematical Deep Dive: PEFT vs LoRA vs Other LLM Optimization Techniques 🧮  

When fine-tuning large language models, the choice of optimization technique fundamentally determines computational efficiency and model performance. Let's examine the mathematical foundations:  

**LoRA (Low-Rank Adaptation)** 
🔹 Core Principle: Decompose weight updates into low-rank matrices  
W₀ + ΔW = W₀ + BA  
Where: • W₀ ∈ ℝᵈˣᵈ (original weight matrix) • B ∈ ℝᵈˣʳ, A ∈ ℝʳˣᵈ (low-rank decomposition) • r << d (rank constraint)  
Parameter Reduction: From d² to 2dr parameters Memory Complexity: O(dr) vs O(d²)  

**Prefix Tuning** 
🔹 Mathematical Formulation: h₍ = f(concat(P₍, X₍), θ)  
Where P₍ ∈ ℝˡˣᵈ represents learnable prefix tokens  
Optimization Objective: min_P ∑ᵢ L(y₍ | concat(P, x₍))  

**AdaLoRA (Adaptive LoRA)** 
🔹 Dynamic Rank Allocation: r₍ᵗ⁺¹ = r₍ᵗ + Δr₍ · importance_score₍  
Singular Value Regularization: L_total = L_task + λ∑ᵢ ||σᵢ||₁  

**QLoRA Quantization** 
🔹 4-bit Quantization Mapping: W_quantized = Q(W) = round((W - min(W)) × (2⁴-1)/(max(W) - min(W)))  
Memory Reduction: ~75% with negligible performance loss  

----------------------------------------------------------------
== Comparative Analysis ==

LoRA: • Param Efficiency: 2dr/d² • Memory: O(dr) • Training Speed: 1.2x faster • Performance: 98-99% full FT  

Prefix Tuning: • Param Efficiency: l×d/total • Memory: O(ld) • Training Speed: 1.5x faster • Performance: 95-97% full FT  

AdaLoRA: • Param Efficiency: Dynamic • Memory: O(r_avg×d) • Training Speed: 1.1x faster • Performance: 99%+ full FT  

QLoRA: • Param Efficiency: Same as LoRA • Memory: 0.25x • Training Speed: 0.8x slower • Performance: 97-98% full FT  

Mathematical Insights 🧠  
Rank-Performance Trade-off: Performance ≈ 1 - exp(-αr/d)  
Optimal Rank Selection: r* = argmin_r [L_validation(r) + β·cost(r)]  
The mathematical elegance of these techniques lies in their ability to approximate full parameter fine-tuning while operating in significantly lower-dimensional subspaces. LoRA's low-rank assumption leverages the intrinsic dimensionality hypothesis, while prefix tuning exploits the transformer's attention mechanism's capacity for context conditioning.  

Key Takeaway 💡 The choice between techniques should be guided by your computational constraints, target performance, and the mathematical properties of your specific use case.  
What optimization techniques are you finding most effective in your LLM workflows?  

Reference:
linkedin post [link](https://www.linkedin.com/feed/update/urn:li:activity:7369535239460671489?updateEntityUrn=urn%3Ali%3Afs_updateV2%3A%28urn%3Ali%3Aactivity%3A7369535239460671489%2CFEED_DETAIL%2CEMPTY%2CDEFAULT%2Cfalse%29)