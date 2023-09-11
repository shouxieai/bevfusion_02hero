import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.stats as stats

def generator_P(size):
    walk = []
    avg = random.uniform(3.000, 600.999)
    std = random.uniform(500.000, 1024.000)
    for _ in range(size):
        walk.append(random.gauss(avg, std))
    return walk

def smooth_distribution(p, eps=0.0001):
    is_zero = (p == 0)
    is_nonzero = (p != 0)
    n_zeros = np.sum(is_zero)
    n_nonzeros = np.sum(is_nonzero)

    if not n_nonzeros:
        raise ValueError("The discrete probability distribution is malformed. All entries are 0.")
    
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, f'eps1 equals to {eps1} which is too large.'
    hist = p.astype(np.float32)
    hist += eps * is_zero - eps1 * is_nonzero
    assert (hist <= 0).sum() == 0, 'distribution still has zero'
    
    return hist

def threshold_distribution(distribution, target_bin=128):
    distribution = distribution[1:]
    length = distribution.size
    # choose bins from 128 to 2048
    threshold_sum = sum(distribution[target_bin:])
    # generate empty array to store kl divergence
    kl_divergence = np.zeros(length - target_bin)
    
    for threshold in range(target_bin, length):
        # choose threshold bins
        sliced_nd_hist = distribution[:threshold].copy()
        
        # generate reference distribution p
        p = sliced_nd_hist.copy()
        p[threshold - 1] += threshold_sum
        threshold_sum = threshold_sum - distribution[threshold]
        
        # whether all the values of p is non-zero
        is_nonzeros = (p != 0).astype(np.int64)
        
        quantized_bins = np.zeros(target_bin, dtype=np.int64)
        # calculate how many bins should be merged to generate
        # quantized distribution q
        num_merged_bins = sliced_nd_hist.size // target_bin
        
        # merge hist into q bins
        for j in range(target_bin):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum()
        
        # expand quantized bins into p.size bins
        q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
        for j in range(target_bin):
            start = j * num_merged_bins
            if j == target_bin - 1:
                stop = -1
            else:
                stop = start + num_merged_bins
            
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        
        # smooth operation, avoid contering nan
        p = smooth_distribution(p)
        q = smooth_distribution(q)
        
        # calculate KL divergence between p and q
        kl_divergence[threshold - target_bin] = stats.entropy(p, q)
    
    # find the index of the minimum KL divergence
    min_kl_divergence = np.argmin(kl_divergence)
    # calculate the threshold
    threshold_value = min_kl_divergence + target_bin
    
    return threshold_value

if __name__ == '__main__':
    size = 2048
    P = generator_P(size)
    P = np.array(P)
    P = P[P > 0]
    print(f"最大的激活值：{max(np.absolute(P))}")
    
    hist, bins = np.histogram(P, bins=2048)
    threshold = threshold_distribution(hist, target_bin=128)
    print(f"threshold 所在组：{threshold}")
    print(f"threshold 所在组的区间范围：{bins[threshold]}")
    
    plt.title("ReLU Activation Value Histogram")
    plt.xlabel("Activation values")
    plt.ylabel("Normalized number of counts")
    plt.hist(P, bins=2047)
    plt.vlines(bins[threshold], 0, 5, colors='r', linestyles='dashed')
    plt.show()