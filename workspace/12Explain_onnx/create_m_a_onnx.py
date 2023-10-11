import torch
import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto

def create_onnx():
    # ValueInfoProto
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [10, 10])
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 10])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [10, 10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10, 10])
    p = helper.make_tensor_value_info('p', TensorProto.FLOAT, [10, 10])
    q = helper.make_tensor_value_info('q', TensorProto.FLOAT, [10, 10])
    
    # NodeProto
    # mul = helper.make_node('Mul', ['a', 'x'], 'c', 'multiply')
    # add = helper.make_node('Add', ['c', 'b'], 'y', 'add')
    add = helper.make_node('Add', ['a', 'x'], 'c', 'add')
    add1 = helper.make_node('Add', ['c', 'b'], 'd', 'add')
    add2 = helper.make_node('Add', ['a', 'p'], 'h', 'add')
    add3 = helper.make_node('Add', ['h', 'q'], 'i', 'add')
    add4 = helper.make_node('Add', ['d', 'i'], 'y', 'add')
    
    # GraphProto
    # graph = helper.make_graph([add, mul], 'sample-linear', [a, x, b], [y])
    graph = helper.make_graph([add, add2, add1, add3, add4], 'sample-linear', [b, a, q, p, x], [y])
    
    # ModelProto
    model = helper.make_model(graph)

    # 检查 model 是否有错误
    onnx.checker.check_model(model)
    print(model)

    # 保存 model
    onnx.save(model, "sample-linear.onnx")
    
    return model

if __name__ == "__main__":
    model = create_onnx()
    