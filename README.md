[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=6404706&assignment_repo_type=AssignmentRepo)
# 基于图像拼接的 A Look Into the Past
成员及分工
>PB18000083 王文灏
>
>PB18?????? 周振
>

## 问题描述
>通过图像拼接可以将一张老照片与一张现代照片拼接在一起,展示一些历经百年沧桑仍未发生变化的风景
>
>素材选取为德国慕尼黑的一座建筑分别于1910年和2017年拍摄的照片,通过图像拼接将两张照片拼在一起
>
>问题可以抽象为特征点匹配,检测特征点通过SIFT实现,利用RANSAC选出最优的4组特征点进行匹配,
>拼接通过特征点匹配得到的变换矩阵来实现

## 原理分析
### SIFT算法
SIFT算法通过高斯金字塔得到特征点并计算描述子。分为四个步骤：
>尺度空间极值检测: 在所有尺度上通过高斯微分函数搜索潜在的对于尺度和旋转不变的兴趣点。
>
>关键点定位: 通过拟合精确确定关键点的位置,同时去除低对比度和不稳定的点。
>
>方向计算: 基于图像局部的梯度方向,计算每个关键点的主方向和可能的辅方向。
>
>描述子的计算: 对关键点周围图像区域分块并计算块内梯度直方图,得到的具有唯一性的向量为描述子。描述子抽象描述了周围区域的图像信息。

### RANSAC算法
RANSAC是根据一组包含异常数据的样本数据集,计算出数据的数学模型参数,得到有效样本数据的算法。分为四个步骤:
>1.随机从数据集中随机抽出4个不共线的样本数据,计算出变换矩阵H,记为模型M
>
>2.计算数据集中所有数据与模型M的投影误差,若误差小于阈值,加入内点集I
>
>3.如果当前内点集I元素个数大于最优内点集I_best,则更新I_best=I,同时更新迭代次数k
>
>4.如果迭代次数大于最大次数则退出,否则迭代次数加1并重复上述步骤

## 代码实现
### SIFT检测特征点
```python
def detectAndDescribe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   # 将彩色图片转换成灰度图
    descriptor = cv2.xfeatures2d.SIFT_create()    # 建立SIFT生成器
    (kps, features) = descriptor.detectAndCompute(image, None)    # 检测SIFT特征点，并计算描述子
    kps = np.float32([kp.pt for kp in kps])
    return (kps, features)
```

### RANSAC匹配特征点
```python
def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio = 0.75, reprojThresh = 4.0):
    matcher = cv2.BFMatcher()
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)    # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
    matches = []
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:        # 当最近距离跟次近距离的比值小于阈值时，保留此匹配对
            matches.append((m[0].trainIdx, m[0].queryIdx))
    if len(matches) > 4:                                                 # 当筛选后的匹配对大于4时，计算视角变换矩阵
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        return (matches, H, status)
    return None
```

### 画出特征点连线
```python
def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        if s == 1:
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
    cv_show("drawImg", vis)
    return vis
```

### 图像拼接
```python
def stitch(imageA,imageB, ratio=0.75, reprojThresh=4.0):
    (kpsA, featuresA) = detectAndDescribe(imageA)
    (kpsB, featuresB) = detectAndDescribe(imageB)
    M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
    if M is None:
        return None  
    (matches, H, status) = M
    result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))    # 将图片A进行视角变换，result是变换后图片
    cv_show('result', result)
    result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB    # 将图片B传入result图片最左端
    cv_show('result', result)
    return result
````
