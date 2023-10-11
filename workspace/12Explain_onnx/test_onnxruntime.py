import onnxruntime
import numpy as np

input_val = np.array([2], dtype=np.float16)
sess = onnxruntime.InferenceSession("multi-mul.onnx")
input = {'x': input_val}
output = sess.run(None, input)
print(output)