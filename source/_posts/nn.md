---
title: 神经网络
date: 2018/10/14 19:28:30
tags: 
  - TensorFlow
  - Machine Learning
  - Deep Learning
category: Study
---

## 1. 激活函数去线性化
常用的非线性激活函数：
**ReLU**函数：$f(x) = \max(0, x)$, 整体的值在0和最大的x之间。
**sigmoid**函数:$f(x) = \frac{1}{1 + e^{-x}}$, 整体的值将会被限制在0和1之间。
**tanh**函数：$f(x) = \frac{1 - e^{-2x}}{1 + e^{-2x}}$, 整体的值将会限制在-1和1之间。

## 2. 损失函数
1.  **交叉熵(cross entropy)** 原本是被用来评估平均编码长度的，在深度学习中，常被用作评估分类效果。
    交叉熵公式：给定两个概率分布$p$和$q$，通过$q$表示$p$的交叉熵为：$H(p, q) = -\sum_xp(x)\log{q(x)}$
    交叉熵刻画的是两个概率分布之间的距离，但是神经网络的输出并不一定是一个概率分布，但是可以通过`softmax`函数将输出的结果变成一个概率分布。
    **softmax**函数的定义为： ${softmax(y)}_i={y'_i}=\frac{e^{y_i}}{\sum_{j=1}^n{e^{y_i}}}$
    通过`softmax`函数，原始神经网络的输出被用作置信度来生成新的输出。
    在`TensorFlow`中可以简单的使用 `-tf.reduce_mean(y * tf.log(tf.clip_by_value(pred, 1e-10, 1.)))` 来实现。
    注意：预测值`pred`一般会先经过`softmax`或者是`sigmoid`函数。

2.  **均方误差(MSE, mean squared error)** 
    均方误差函数的定义为： $MSE(y, y') = \frac{\sum_{i=1}^n{(y_i - y'_i)}^2}{n}$
    在`TensorFlow`中可以简单的使用 `mse = tf.reduce_mean(tf.square(y - pred))`

还有很多其他的损失函数，在此不再进行详细的分析说明。

