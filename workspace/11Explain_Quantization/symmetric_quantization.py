import numpy as np

# 截断操作
def saturate(x):
    return np.clip(x, -127, 127)

# 缩放
def scale_cal(x):
    max_val = np.max(np.abs(x))
    return max_val / 127

# 量化
def quant_float_data(x, scale):
    xq = saturate(np.round(x / scale))
    return xq

# 反量化
def dequant_data(xq, scale):
    x = (xq * scale).astype('float32')
    return x

if __name__ == '__main__':
    np.random.seed(4)
    # data_float32 = np.random.randn(3).astype('float32')
    data_float32 = np.array([1.62, -1.62, 0, -0.52, 1.62], dtype='float32')
    print(f"input: {data_float32}")
    scale = scale_cal(data_float32)
    print(f"scale: {scale}")
    data_int8 = quant_float_data(data_float32, scale)
    print(f"quant: {data_int8}")
    data_dequant_float = dequant_data(data_int8, scale)
    print(f"dequant: {data_dequant_float}")
    print(f"diff: {data_dequant_float - data_float32}")
