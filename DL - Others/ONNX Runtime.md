
import onnxruntime as ort    導入onnxruntime模組

opt = ort.SessionOptions()   

inference_session = ort.InferenceSession(weights_path, opt)  

创建一个InferenceSession的实例并传给它一个模型地址

onnx_output = inference_session.run(output_names, input_tensors)    调用run方法进行模型推理