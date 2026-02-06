import numpy as np
# 手动实现最大池化
def manual_max_pooling(img, k=2, s=2):
    h, w = img.shape
    # 1. 计算输出大小
    out_h = (h - k) // s + 1
    out_w = (w - k) // s + 1
    res = np.zeros((out_h, out_w))
    
    # 2. 开始滑动
    for i in range(0, h - k + 1, s):
        for j in range(0, w - k + 1, s):
            # 3. 切片并找最大值
            window = img[i:i+k, j:j+k]
            res[i//s, j//s] = np.max(window)
            
    return res

# 手动实现平均池化
def manual_avg_pooling(img, k=2, s=2):
    h, w = img.shape
    # 1. 计算输出大小
    out_h = (h - k) // s + 1
    out_w = (w - k) // s + 1
    res = np.zeros((out_h, out_w))
    
    # 2. 开始滑动
    for i in range(0, h - k + 1, s):
        for j in range(0, w - k + 1, s):
            # 3. 切片并找平均值
            window = img[i:i+k, j:j+k]
            res[i//s, j//s] = np.mean(window)
            
    return res
# 测试一下
test_img = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12],
                     [13, 14, 15, 16]])
print(test_img)
print("最大池化结果：\n", manual_max_pooling(test_img))
print("平均池化结果：\n", manual_avg_pooling(test_img))