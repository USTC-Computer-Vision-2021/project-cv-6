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
>问题可以抽象为特征点匹配,检测特征点通过SIFT实现,利用RANSAC选出4组特征点进行匹配,
>拼接通过特征点匹配得到的变换矩阵来实现

## 原理分析
### SIFT算法
