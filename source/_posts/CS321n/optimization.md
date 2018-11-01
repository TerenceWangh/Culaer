---
title: 相关优化
date: 2018-10-29 20:13:00
tags:
  - Machine Learning
  - CS231n
  - Optimization
category: CS231n
---

## 相关介绍
之前我们介绍了图像分类的两个主要的模块：
  1. Score Function用来将原始的图片像素映射到分类的得分上，以前介绍了线性分类函数。
  2. Loss Function用来验证学得参数的计算分类的得分和真实分类之间的差距。这种函数有很多，之前介绍了Softmax和SVM。

再次回到线性分类$f(x_i, W) = Wx_i$和SVM，我们得到了下面的公式：
$$ L = \frac{1}{N}\sum_i{\sum_{j \neq y_i}{[\max(0, {f(x_i; W)}_j - {f(x_i; W)}_{y_i} + 1)] + \alpha R(W)}} $$
我们能够看出，如果对于图片$x_i$，当参数集$W$能够和真实分类预测一致时，计算得到的损失是非常小的。现在开始介绍第三个部分也是比较重要的一部分：**优化器**。优化器是寻找最小化损失的参数集$W$的过程。
**写在前面**：一旦我们理解了这三个部分是如何相互运作的，我们再回到第一个部分（基于参数的函数映射），然后将其拓展为一个远比线性函数复杂的函数：首先是神经网络，然后是卷积神经网络。而损失函数和最优化过程这两个部分将会保持相对稳定。

## 损失函数可视化
损失函数通常是定义在一个非常高维的空间上(例如，在CIFAR-10中，权重矩阵大小为[10 x 3073]共有30730个参数)，很难去可视化它们。然而，我们依然能够通过将高维空间压缩成一维的线或者是二维的空间去可视化它们。例如，我们能够产生一个随机的权重矩阵$W$，实际上是在空间上对应一个点，然后沿着一条射线并同时记录损失函数的值。然后，我们产生了一个随机的方向$W_1$并沿着该方向计算另一个点的损失：$L(W + aW_1)$。这样便产生了一个简单的图，其中a作为该图的x轴，损失作为该图的y轴。同时，我们也可以通过$L(W + aW_1 + bW_2)$在二维空间上计算损失。其中a,b对应x,y轴，损失用颜色来表示。具体的例子如下图：
{% asset_img svm_one.jpg %}
>多分类SVM(不包含规范化)对于一个样本的可视化图(左和中间的图)，以及对于100个样本的可视化图。
<hr />
我们通过数学公式来解释损失函数的分线段性。对于单个样本来说，损失函数可以表示为：
$$ L_i = \sum_{j \neq y_i}{[\max(0, w_j^Tx_i - w_{y_i}^Tx_i + 1)]} $$
很明显的可以看出数据的损失是以$W$为线性函数值的总和。而且，对于$W$的每一行，有的时候它前面是一个正号(对应错误分类)，有的时候它前面是一个负号(对应正确分类)。为了能够进一步进行说明，考虑一个简单一维的三分类样本，SVM总的损失值(不包含规范化损失)可写成：
$$ L_0 = \max(0, w_1^Tx_0 - w_0^Tx_0 + 1) + \max(0, w_2^Tx_0 - w_0^Tx_0 + 1) $$
$$ L_1 = \max(0, w_0^Tx_1 - w_1^Tx_1 + 1) + \max(0, w_2^Tx_1 - w_1^Tx_1 + 1) $$
$$ L_2 = \max(0, w_0^Tx_2 - w_2^Tx_2 + 1) + \max(0, w_1^Tx_2 - w_2^Tx_2 + 1) $$
$$ L = (L_0 + L_1 + L_2) / 3 $$
由于上面的样本都是一维的，因此，数据$x_i$以及权重$w_j$都是常数。观察$w_0$，可以看到上面的公式中都是$w_0$的线性函数，而且每一项都会与0进行比较取最大值。那么整体函数如下图所示：
{% asset_img svm_hd.jpg %}
>从一个维度方向上对数据损失值的展示。x轴方向就是权重，y轴方向是损失。数据的损失是几个函数的和，其中每个函数要么是某个权重的独立部分，要么是该权重的线性函数和0之间的比较。30730维度的损失函数也是类似这样的。
<hr />
通过SVM损失函数的碗状的形状可以猜测出它是一个凸函数，关于凸函数的最小化很很多论文去探讨如果更高效的进行。但是当我们将函数扩展到神经网络上时，目标函数就不是凸函数了，而是凹凸不平的复杂函数。
*不可导的损失函数*，由于max函数的存在，在函数中存在不可导点。但是梯度仍然存在，所以经常使用梯度而不是导数。

## 优化器
损失函数使我们能够衡量参数集$W$的质量。优化器的目标是找到使得损失函数最小化的参数集$W$。这里用SVM来一步一步介绍优化器，虽然SVM是凸函数，可能使用优化器有些奇怪。但是不要忘记，在神经网络中使用优化器不是那么简单的事，因此我们要一步一步的深入了解优化器。

### 策略1：随机搜索 (比较差的一种思想)
由于我们能够衡量给定参数集$W$的质量，因此一个简单的方法是尽可能的使用不同的随机的权重并记录那个权重具有最好的质量。这个过程可以使用下列程序描述：
``` python
# assume X_train is the data where each column is an example (e.g. 3073 x 50000).
# assume Y_train is the labels (e.g. 1D array of 50000).
# assume the function L evaluates the loss function.

bestloss = float('inf')
for num in range(1000):
  W = np.random.randn(10, 3073) * 0.0001  # generate random parameters
  loss = L(X_train, Y_train, W)           # get the loss over the entire training set
  if loss < bestloss:     # keep track of the best solution
    bestloss = loss
    bestW = W
  print 'in attempt %d the loss was %f, best %f' % (num, loss, bestloss)
```
>在上面的程序中，我们尝试了几个随机的权重响亮，有些能够更好的工作。我们使用最终最好的结果在测试集上进行测试：
``` python
# assume X_test is [3073 x 10000], Y_test [10000 x 1]
scores = Wbest.dot(Xte_cols)  # 10 x 10000, the class scores for all test example
# find the index with max score in each column (the predicted class)
Yte_predict = np.argmax(scores, axis=0)
# and calculate accuracy (fraction of predictions that are correct)
np.mean(Yte_predict == Yte)
```
上述的程序大概只能得到15.5%的准确率。但是一次随机的参数值能够达到10%左右，因此这还不算是特别坏的解决方案。
<br />
**核心思想:迭代求精。**核心思想是寻找最好的权重，当然，这非常困难乃至于是不可能的难题(尤其是复杂的神经网络中拥有大量参数)，但是如果将问题转化成：*对一个权重矩阵集$W$取优，使得损失值减小，那么问题的难度就大大降低了。换句话说，就是我们的方式从一个随机的$W$开始，然后对其迭代取优，每次使得损失值更小一点。
>我们的策略是从随机权重开始，然而迭代取优，从而获得更低的损失。

### 策略2： 随机本地搜索
第一个策略可以看作每走一步都尝试几个随机方向，如果某个方向是向山下的方向，就向该方向走一步。这次，我们从随机的$W$开始，然后生成一个随机的扰动增量$\deltaW$，如果在$W + \deltaW$处的损失更低，我们才会更新。代码如下：
``` python
W = np.random.randn(10, 3073) * 0.001
for i in range(1000):
  step_size = 0.0001
  Wtry = W + np.random.randn(10, 3073) * step_size
  loss = L(Xtr_cols, Ytr, Wtry)
  if loss < bestloss:
    W = Wtry
    bestloss = loss
  print 'iter %d loss is %f' % (i, bestloss)
```
>使用相同数据，这个方法能够达到**21.4%**的准确率。这个策略相对于第一个策略好，但是比较浪费计算资源。

### 策略3： 梯度下降
