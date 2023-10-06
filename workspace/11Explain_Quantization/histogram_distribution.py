import numpy as np

# 缩放
def scale_cal(x):
    max_val = np.max(np.abs(x))
    return max_val / 127

def histogram_bins(x):
    hist, bins = np.histogram(x, 100)
    total = len(x)
    left = 0
    right = len(hist) - 1
    limit = 0.99
    
    while True:
        cover_percent = hist[left:right+1].sum() / total
        if cover_percent <= limit:
            break
        
        if hist[left] < hist[right]:
            left += 1
        else:
            right -= 1
    
    left_val = bins[left]
    right_val = bins[right]
    
    dynamic_range = max(abs(left_val), abs(right_val))
    
    return dynamic_range / 127.

if __name__ == '__main__':
    np.random.seed(1)
    data_float32 = np.random.randn(1000).astype('float32')
    scale = scale_cal(data_float32)
    scale1 = histogram_bins(data_float32)
    print(f"scale: {scale}")
