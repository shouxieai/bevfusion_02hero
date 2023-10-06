import numpy as np
import matplotlib.pyplot as plt

def cal_kl(p, q):
    KL = 0
    for i in range(len(p)):
        KL += p[i] * np.log(p[i] / q[i])
    return KL

def kl_test(x, kl_threshold = 0.01):
    while True:
        y = [np.random.uniform(1, size + 1) for i in range(size)]
        y /= np.sum(y)
        kl_result = cal_kl(x, y)
        if kl_result < kl_threshold:
            print(f"result: {kl_result}")
            y_out = y
            plt.plot(x)
            plt.plot(y)
            break
    
    return y_out

if __name__ == '__main__':
    np.random.seed(1)
    size = 10
    kl_threshold = 0.01
    x = [np.random.uniform(1, size + 1) for i in range(size)]
    x = x / np.sum(x)
    y_out = kl_test(x, kl_threshold)
    print(f"x: {x}\ny: {y_out}")
    plt.xlabel("Index")
    plt.ylabel("Vaule")
    plt.title(f"KL threshold: {kl_threshold}")
    plt.show()
    