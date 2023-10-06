import numpy as np

def smooth_data(p, eps=0.0001):
    is_zero = (p == 0).astype(np.float32)
    is_nonzero = (p != 0).astype(np.float32)
    n_zeros = is_zero.sum()
    n_nonzeros = is_nonzero.sum()
    
    eps1 = eps * n_zeros / n_nonzeros
    hist = p.astype(np.float32)
    hist += eps * is_zero - eps1 * is_nonzero
    
    return hist

def cal_kl(p, q):
    KL = 0
    for i in range(len(p)):
        print(f"p[i]: {p[i]}, q[i]: {q[i]}")
        KL += p[i] * np.log(p[i] / q[i])
    return KL

if __name__ == '__main__':    
    p = [1, 0, 2, 3, 5, 3, 1, 7]
    bin = 4
    split_p = np.array_split(p, bin)
    q = []
    for arr in split_p:
        avg = np.sum(arr) / np.count_nonzero(arr)
        for item in arr:
            if item != 0:
                q.append(avg)
                continue
            q.append(0)
    print(q)
    
    p /= np.sum(p)
    q /= np.sum(q)
    
    p = smooth_data(p)
    q = smooth_data(q)
    
    # 计算KL散度
    print(f"kl_divergence: {cal_kl(p, q)}")
    