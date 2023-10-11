import onnx
import onnx.helper as helper
import torch
import numpy as np

num = torch.tensor([2], dtype=torch.float16)
weight = torch.tensor([3], dtype=torch.float16)

a = helper.make_value_info(
    name='x',
    type_proto=helper.make_tensor_type_proto(
        elem_type=helper.TensorProto.DataType.FLOAT16,
        shape=num.size()
    )
)

b = helper.make_value_info(
    name='y',
    type_proto=helper.make_tensor_type_proto(
        helper.TensorProto.DataType.FLOAT16,
        shape=num.size()
    )
)

w = helper.make_tensor(
    name='m',
    data_type=helper.TensorProto.DataType.FLOAT16,
    dims=list(weight.shape),
    vals=weight.data.numpy().astype(np.float16).tobytes(),
    raw=True
)

node1 = helper.make_node(
    op_type='Mul',
    inputs=['x', 'm'],
    outputs=['o1'],
    name='Mul1'
)

node2 = helper.make_node(
    op_type='Mul',
    inputs=['o1', 'm'],
    outputs=['o2'],
    name='Mul2'
)

node3 = helper.make_node(
    op_type='Mul',
    inputs=['o2', 'm'],
    outputs=['o3'],
    name='Mul3'
)

node4 = helper.make_node(
    op_type='Mul',
    inputs=['o3', 'm'],
    outputs=['y'],
    name='Mul4'
)

graph = helper.make_graph(
    nodes=[node1, node2, node3, node4],
    name='Multi-Mul',
    inputs=[a],
    outputs=[b],
    initializer=[w]
)


# opset = [helper.make_operatorsetid("ai.onnx", 13)]

# model = helper.make_model(
#     graph=graph, 
#     opset_imports=opset,
#     producer_name='pytorch',
#     producer_version="1.9"
# )

model = helper.make_model(graph=graph)

onnx.checker.check_model(model)
onnx.save_model(model, "multi-mul.onnx")
