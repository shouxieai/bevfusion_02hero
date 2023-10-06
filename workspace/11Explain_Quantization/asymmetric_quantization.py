import numpy as np

# 截断操作
def saturate(x, int_max, int_min):
    return np.clip(x, int_min, int_max)

# 计算缩放和偏移量
def scale_z_cal(x, int_max, int_min):
    scale = (x.max() - x.min()) / (int_max - int_min)
    z = int_max - np.round((x.max() / scale))
    return scale, z

# 量化
def quant_float_data(x, scale, z, int_max, int_min):
    xq = saturate(np.round(x / scale + z), int_max, int_min)
    return xq

# 反量化
def dequant_data(xq, scale, z):
    x = ((xq - z) * scale).astype('float32')
    return x

if __name__ == '__main__':
    # np.random.seed(0)
    data_float32 = np.random.randn(3).astype('float32')
    # data_float32 = np.random.randn(100).astype('float32')
    # data_float32[99] = 100
    # data_float32 = np.array([-0.61, -0.52, 1.62], dtype='float32')
    print(f"input: {data_float32}")
    
    # uint8 bound
    # int_max = 255
    # int_min = 0
    
    # int8 bound
    int_max = 127
    int_min = -128
    
    scale, z = scale_z_cal(data_float32, int_max, int_min)
    print(f"scale: {scale}, z: {z}")
    data_int8 = quant_float_data(data_float32, scale, z, int_max, int_min)
    print(f"quant: {data_int8}")
    data_dequant_float = dequant_data(data_int8, scale, z)
    print(f"dequant: {data_dequant_float}")
    print(f"diff: {data_dequant_float - data_float32}")
