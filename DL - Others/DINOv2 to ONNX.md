
1. Export to ONNX supported? [#19](https://github.com/facebookresearch/dinov2/issues/19)
2. Additional model exports (ONNX, CoreML, ...) [#167](https://github.com/facebookresearch/dinov2/issues/167)
3. How to convert DINOv2 to ONNX? [#216](https://github.com/facebookresearch/dinov2/issues/216)
4. ONNX error with DINOv2 with registers [#288](https://github.com/facebookresearch/dinov2/issues/288)

#19

-------------------------------------------------------------------

Step1

將dinov2/dinov2/models/vision_transformer.py 中的def vit_small (vit_base, vit_large, vitgiant2)(line348, 362, 376, 393)的block_fn從 block_fn=partial(Block, attn_class=MemEffAttention), 改成

block_fn=partial(Block, attn_class=Attention),

Step2

device使用cpu

#216

-------------------------------------------------------------------

1.

修改dinov2/dinov2/models/vision_transformer.py 中def interpolate_pos_encoding(self, x, w, h)的line204

2.

Dynamic input shape when ONNX inference?