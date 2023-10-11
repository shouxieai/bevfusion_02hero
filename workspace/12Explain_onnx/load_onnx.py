import onnx
import onnx.helper as helper
import numpy as np

model = onnx.load("demo.onnx")

#打印信息
print("==============node信息")
# print(helper.printable_graph(model.graph))
print(model)

conv_weight = model.graph.initializer[0]
conv_bias = model.graph.initializer[1]

# initializer里有dims这个属性是可以通过打印model看到的
# dims在onnx-ml.proto文件中是repeated类型的，即数组类型，所以要用索引去取！
print(conv_weight.dims)
# 取node节点的第一个元素
print(f"===================={model.graph.node[1].name}==========================")
print(model.graph.node[1])

# 数据是以protobuf的格式存储的，因此当中的数值会以bytes的类型保存，通过np.frombuffer方法还原成类型为float32的ndarray
print(f"===================={conv_weight.name}==========================")
print(conv_weight.name, np.frombuffer(conv_weight.raw_data, dtype=np.float32))

print(f"===================={conv_bias.name}==========================")
print(conv_bias.name, np.frombuffer(conv_bias.raw_data, dtype=np.float32))
