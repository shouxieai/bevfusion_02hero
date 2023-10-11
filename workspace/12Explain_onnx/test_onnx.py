import torch
import numpy as np
import onnx
from onnx import helper

# 输入1: name - 'input1', shape - [1], 类型 - FLOAT
# 创建一个 TypeProto 消息，这个消息描述了一个具有指定元素类型和形状的张量类型
tensor_type = helper.make_tensor_type_proto(
    elem_type=helper.TensorProto.DataType.FLOAT16,
    shape=[1]
)
# 创建一个 ValueInfoProto 消息，该消息描述了一个具有指定名称和类型的值。
input1 = helper.make_value_info(name='input1', type_proto=tensor_type)

# 输入2: name - 'input2', shape - [1], 类型 - FLOAT
tensor_type = helper.make_tensor_type_proto(
    elem_type=helper.TensorProto.DataType.FLOAT16,
    shape=[1]
)
input2 = helper.make_value_info(name='input2', type_proto=tensor_type)

# 输出: name - 'output', shape - [1], 类型 - FLOAT
tensor_type = helper.make_tensor_type_proto(
    elem_type=helper.TensorProto.DataType.FLOAT16,
    shape=[1]
)
output = helper.make_value_info(name='output', type_proto=tensor_type)

# 创建一个节点 (operation) 来表示两个输入的相加.
# 使用 helper.make_node 创建一个节点，其中 op_type 为 'Add'，inputs 为 ['input1', 'input2']，outputs 为 ['output']
node1 = helper.make_node('Add', ['input1', 'input2'], ['output'])

# 创建图 (GraphProto)，添加初始化器到图中
graph = helper.make_graph(
    nodes=[node1],
    name='add_model',
    inputs=[input1, input2],
    outputs=[output],
)

# 创建模型 (ModelProto)
save_onnx = "cus_model.onnx"
model_def = helper.make_model(graph, producer_name='pytorch', producer_version='1.9')

print(f"🚀 The export is completed. ONNX save as {save_onnx} 🤗, Have a nice day~")
onnx.save_model(model_def, save_onnx)

from onnxsim import simplify
model_def = onnx.load(save_onnx)
model_sim, check = simplify(model_def)
onnx.save_model(model_def, "cus_model_sim.onnx")

import onnxruntime

sess = onnxruntime.InferenceSession("cus_model_sim.onnx")
input1_vals = np.random.rand(1).astype(np.float16)
input2_vals = np.random.rand(1).astype(np.float16)
inputs = {"input1": input1_vals, "input2": input2_vals}
output = sess.run(None, inputs)
print(f'{input1_vals[0]} + {input2_vals[0]} = {output[0][0]}')