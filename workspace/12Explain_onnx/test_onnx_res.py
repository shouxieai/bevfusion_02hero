import torch
import numpy as np
import onnx
from onnx import helper

# 创建输入
input_tensor_type = helper.make_tensor_type_proto(
    elem_type=helper.TensorProto.DataType.FLOAT16,
    shape=[5, 5]
)
input1 = helper.make_value_info(name='input1', type_proto=input_tensor_type)
input2 = helper.make_value_info(name='input2', type_proto=input_tensor_type)

# 创建权重和偏置初始化器
w1_vals = torch.randn([5, 5], dtype=torch.float16)
w1 = helper.make_tensor(
    name='w1',
    data_type=helper.TensorProto.FLOAT16,
    dims=w1_vals.shape,
    vals=w1_vals.data.numpy().astype(np.float16).tobytes(),
    raw=True,
)

w2_vals = torch.randn([5, 5], dtype=torch.float16)
w2 = helper.make_tensor(
    name='w2',
    data_type=helper.TensorProto.FLOAT16,
    dims=w2_vals.shape,
    vals=w2_vals.data.numpy().astype(np.float16).tobytes(),
    raw=True,
)

w3_vals = torch.randn([5, 5], dtype=torch.float16)
w3 = helper.make_tensor(
    name='w3',
    data_type=helper.TensorProto.FLOAT16,
    dims=w3_vals.shape,
    vals=w3_vals.data.numpy().astype(np.float16).tobytes(),
    raw=True,
)

# 创建输出
output_tensor_type = helper.make_tensor_type_proto(
    elem_type=helper.TensorProto.DataType.FLOAT16,
    shape=[5, 5]
)
output1 = helper.make_value_info(name='output1', type_proto=output_tensor_type)
output2 = helper.make_value_info(name='output2', type_proto=output_tensor_type)

# 创建节点
node1 = helper.make_node('Mul', ['input1', 'w1'], ['o1'])
node2 = helper.make_node('Mul', ['o1', 'w2'], ['o2'])
node3 = helper.make_node('Mul', ['input1', 'w3'], ['o3'])
node4 = helper.make_node('Add', ['o2', 'o3'], ['output1'])

node5 = helper.make_node('Mul', ['input2', 'w3'], ['o4'])
node6 = helper.make_node('Add', ['o3', 'o4'], ['output2'])

# 创建图
graph = helper.make_graph(
    nodes=[node1, node2, node3, node4, node5, node6],
    name='cus_mul_add_model',
    inputs=[input1, input2],
    outputs=[output1, output2],
    initializer=[w1, w2, w3],
)

# 创建模型
save_onnx = "cus_mul_add_model.onnx"
model_def = helper.make_model(graph, producer_name='pytorch', producer_version='1.9')

onnx.checker.check_model(model_def)

print(f"🚀 The export is completed. ONNX save as {save_onnx} 🤗, Have a nice day~")
onnx.save_model(model_def, save_onnx)

import onnxruntime

input1_vals = np.random.randn(5, 5).astype(np.float16)
input2_vals = np.random.randn(5, 5).astype(np.float16)

sess = onnxruntime.InferenceSession("cus_mul_add_model.onnx")
inputs = {'input1': input1_vals, 'input2': input2_vals}
output1, output2 = sess.run(None, inputs)
