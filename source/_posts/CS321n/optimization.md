---
title: 优化器
date: 2018-10-30 13:53:28
tags:
  - Machine Learning
  - CS231n
  - Optimization
category: CS231n
---

### 策略3： 梯度下降
在之前都是试图随机的在权重空间找到一个方向来降低损失。但是数学上，能够直接计算出最好的方向，而不是随机的去寻找方向。这个最好的方向就是损失函数的**梯度**。
在一维的函数中，斜率是函数在某一点的瞬时变化，梯度是函数的一般化表述，它关心的不是某个点，而是一个向量。在输入空间，每个维度上的梯度是斜率的向量(一般是指导数)。一维上的求导公式如下：
$$ \frac{df(x)}{dx} = \lim_{h \to 0}{\frac{f(x + h) - f(x)}{h}} $$
当函数的输入不是一个数而是一个向量时，我们称导数为**偏导数**，梯度就是在每一个维度偏导数组成的向量。

## 计算梯度
有两种方法能够计算梯度：一种缓慢的近似的简单方式(**数值梯度法**)，另一种时快速精确但是容易出错的(**分析梯度法**)。

### 数值梯度法
数值梯度法是指利用有限的差值去计算梯度。下面程序展示了计算输入函数`f`在向量`x`处的梯度。
``` python
def eval_numerical_gradient(f, x):
    """
    a native implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evalue the gradient at
    """

    fx = f(x)
    grad = np.zeros(x.shape)
    h = 0.00001

    # iterate over all indexes in x
    it = np.nditer(x, flag=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # evaluate function at (x + h)
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h
        fxh = f(x)  # evalute f(x + h)
        x[ix] = old_value

        # compute the partial derivative
        grad[ix] = (fxh - fx) / h
        it.iternext()

    return grad
```
>上述代码在每个维度上，通过改变很小的变化`h`，通过观察函数值的变化，计算函数在该维度上的偏导数。最后所有的梯度存储在变量grad中。

**实践考量:**
数学公式中$h$趋近于0，但是实际使用很小的数就够了。在不产生数值计算出错的前提下，尽量使用可能小的值。同时，采用**中心差值公式**$ ^{f(x + h) - f(x - h)}/_{2h} $的形式会更好([Read More](http://en.wikipedia.org/wiki/Numerical_differentiation))。
我们可以使用上面的公式计算任意一点的梯度。下面以CIFAR-10的损失函数为例在权重空间上随机去一些点计算其梯度。
``` python
# to use the generic code above we want a function that takes a single argument
# (the weights in our case) so we close over X_train and Y_train.
def CIFAR10_loss_fun(W):
    return L(X_train, Y_train, W)

W = np.random.rand(10, 3073) * 0.001                # random weight vector.
df = eval_numerical_gradient(CIFAR10_loss_fun, W)   # get the gradient.
```
梯度能够表示每个维度的斜率，因此我们可以按照下面方式进行更新权重。
``` python
loss_original = CIFAR10_loss_fun(W)
print("original loss: %f" % loss_original)

# lets see the effect of multiple step sizes.
for step_size_log in [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]:
    step_size = 10 ** step_size_log
    W_new = W - step_size * df
    loss_new = CIFAR10_loss_fun(W_new)
    print("for step size %f new loss: %f" % (step_size, loss_new))

# prints:
# original loss: 2.200718
# for step size 1.000000e-10 new loss: 2.200652
# for step size 1.000000e-09 new loss: 2.200057
# for step size 1.000000e-08 new loss: 2.194116
# for step size 1.000000e-07 new loss: 2.135493
# for step size 1.000000e-06 new loss: 1.647802
# for step size 1.000000e-05 new loss: 2.844355
# for step size 1.000000e-04 new loss: 25.558142
# for step size 1.000000e-03 new loss: 254.086573
# for step size 1.000000e-02 new loss: 2539.370888
# for step size 1.000000e-01 new loss: 25392.214036
```
*   **在负方向上更新**：在上面的代码中，我们沿着梯度的负方向更新，这是因为我们期望损失函数降低而不是升高。
*   **步长的影响**：梯度指明了函数在哪个方向上变化率时最大的，但是没有指明在该函数上应该走多远。选择步长(也叫做学习率)将是一个非常最要的部分。
*   **效率问题**：计算数值梯度的复杂性和参数的量化性相关。在这个例子中，有30730个参数，所以损失函数每走一步就需要计算30731次损失函数的梯度，显然当神经网络中有用上百万个参数的时候，这种方法时不适用的。

### 分析梯度法
使用分析梯度法能够快速的计算梯度，但是更容易出错。为了解决这个问题，在实际运用中，经常使用数值梯度法来检验分析梯度法实现的正确性，这个步骤也称之为**梯度检查**。
<br />
以SVM损失函数在某个数据点为例：
$$ L_i = \sum_{j \neq y_i}{[\max(0, w_j^Tx_i - w_{y_i}^Tx_i + \Delta)]} $$
对权重$w_{y_i}$进行微分可得：
$$  \nabla_{w_{y_i}}L_i = -\Big(\sum_{j \neq y_i}\mathbb{1}(w_j^Tx_i - w_{y_i}^Tx_i + \Delta > 0)\Big)x_i $$
其中$\mathbb{1}$为示性函数，如果括号中的条件为真，那么函数值为1，如果为假，则函数值为0。然上述公式看起来复杂，但在代码实现的时候比较简单：只需要计算没有满足边界值的分类的数量（因此对损失函数产生了贡献），然后乘以$x_i$就是梯度了。注意，这个梯度只是对应正确分类的$W$的行向量的梯度，那些$j \neq y_i$行的梯度是：
$$  \nabla_{w_{y_i}}L_i = \mathbb{1}(w_j^Tx_i - w_{y_i}^Tx_i + \Delta > 0)x_i $$
一旦微分的公式计算出来，那么是实现梯度的更新就变得非常容易了。

## 梯度下降
现在能够计算损失的梯度了，程序不断的计算梯度然后对参数进行更新。普通版本如下：
``` python
# Vanilla Gradient Descent

while True:
    weights_grad = evaluate_gradient(loss_func, data, weights)
    weights += -step_size * weights_grad
```
这个简单的循环在所有的神经网络库中都有，虽然也有其他实现最优化的方法。但是到目前为止，梯度下降时对神经网络损失最优化的最常用的方法。虽然我们可能会在循环中加一些操作，但是核心思想时不变的：_跟着梯度走，直到损失函数不再变化位置_。

### 小批量梯度下降(mini-batch gradient descent)
在大规模数据集下，训练数据有数百万个样本。计算整个训练集上的损失来更新简单的一个参数时没有必要的。一个常见的方法是仅仅拿出小批量的数据计算梯度。例如，目前最高级的神经网络中，一个典型的批量包含120万个训练集中的256个训练样本。这个小批量用来更新参数：
``` python
# Vanilla Minibatch Gradient Descent

while True:
    data_batch = sample_training_data(data, 256)
    weights_grad = evaluate(loss_func, data_batch, weights)
    weights += -step_size * weights_grad
```
这种方法之所以效果不错，是因为训练集中的数据是相关的。那么小批量数据的梯度就是整个数据集梯度的一个近似，因此在实践中，通过计算小批量数据的梯度可以实现更快速的收敛，并以此进行更频繁的参数更新。

### 随机梯度下降(Stochastic Gradient Descent)
当小批量的梯度下降的批量大小为1时的极端情况被称为随机梯度下降。这种策略在实际情况中相对很少见，因为向量化操作的代码一次计算100个数据远比100次计算一个数据要高效的多。即使SGD在技术上指每次使用一个数据来计算梯度，但是人们习惯使用SGD指代小批量梯度下降。小批量数据的大小是一个超参数，但是一般并不需要通过交叉验证来调参。它一般有存储器的限制来决定，或者干脆设置为相同的大小，如32，64，128等。之所以是2的指数，是因为在实际中许多向量化操作实现的时候，如果输入数据是2的倍数，那么运算更快。

## 反向传播理论

### 动机
反向传播一般是递归地使用**链式法则**(偏导数的链式法则)来计算梯度。理解该过程以及细微之处有利于更有效的开发，设计和调试神经网络。
**问题描述**：给定函数$f(x)$，$x$作为输入向量，我们的目标是计算$f$在$x$处的梯度。如果将该问题放到神经网络中，那么函数$f$对应损失函数，输入$x$对应训练数据和神经网络的权重。这里认为输入时给定的并且是固定的，权重作为控制变量。于是，即使很容易使用反向传播去计算给定输入样本$x_i$的梯度，但实际上我们仅仅计算参数($W, b$)的梯度来进行参数的更新。

### Sigmoid的例子
使用Sigmoid函数作为例子来简单讲解反向传播的相关知识。
公式如下，可以代表二维的神经并使用sigmoid激活函数：
$$ f(w, x) = \frac{1}{1 + e^{-(w_0x_0 + w_1x_1 + w_2)}} $$
上面的函数使用导数法则，并进行拆解，可以看成是下列的函数的组合：
$$
f(x) = \frac{1}{x} 
\hspace{1in} \rightarrow \hspace{1in} 
\frac{df}{dx} = -1/x^2 
\\\\
f_c(x) = c + x
\hspace{1in} \rightarrow \hspace{1in} 
\frac{df}{dx} = 1 
\\\\
f(x) = e^x
\hspace{1in} \rightarrow \hspace{1in} 
\frac{df}{dx} = e^x
\\\\
f_a(x) = ax
\hspace{1in} \rightarrow \hspace{1in} 
\frac{df}{dx} = a
$$
其中，函数$f_c, f_a$分别将输入平移了固定常数$c$以及对输入进行缩放了常数$a$。之所以特别的将这两个操作作为一元门进行介绍是由于我们需要在计算的时候考虑到这两个常数。上面整个链式操作如下图：
{% asset_img sigmod.png %}
我们可以从上图中看到一长串的链式函数来通过计算$w, x$的内积来获得最终的结果。这种函数常被称为*sigmoid函数*$\sigma(x)$。该函数的定义和导数如下：
$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$
$$ \rightarrow \hspace{0.3in} \frac{d\sigma(x)}{dx} = \frac{e^{-x}}{(1+e^{-x})^2} = \left( \frac{1 + e^{-x} - 1}{1 + e^{-x}} \right) \left( \frac{1}{1+e^{-x}} \right) = \left( 1 - \sigma(x) \right) \sigma(x) $$
正如上式给出的，梯度是非常简单的并且很容易进行计算。例如，sigmoid函数的输入为1.0，则输出为0.73。根据上面的导数公式很容易能够计算出局部的导数为 (1 - 0.73) * 0.73 ~= 0.2。对比上面的图，除了那种方式之外还可以简单的使用一个有效的公式就可实现梯度的计算。因此，在实际运用中，将很多操作组合成一个简单的运算是非常高效的。下面给出在神经网络中反向的代码：
``` python
w = [2, -3, -3]
x = [-1, -2]

# forward pass
dot = w[0] * x[0] + w[1] * x[1] + w[2]
f = 1.0 / (1 + math.exp(-dot))  # sigmoid function

# backward pass through the neuron(backpropagation)
ddot = (1 - f) * f  # gradient on dot variable, using the sigmoid gradient derivation
dx = [w[0] * ddot, w[1] * ddot] # backprop into x
dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot] # backprop into w
# we're done! we have the gradients on the inputs to the circuit.
```

### 分阶段反向传播
让我们转到另一个例子上，假设给定函数如下：
$$ f(x, y) = \frac{x + \sigma(y)}{\sigma(x) + {(x + y)}^2} $$
这个函数完全没用，并且你都不知道想要计算哪部分的梯度，但这对于反向传播来说是一个非常好的例子。如果你对这个函数进行微分时，你将会耗费大量的精力并且得到一个非常复杂的公式。由于我们不需要有一个明确的方程来验证梯度，所以计算是非常没有必要的。只需要知道如何计算它即可。下面是前向的代码：
``` python
x = 3
y = -4

# forward pass
sigy    = 1.0 / (1 + math.exp(-y))  # (1)
num     = x + sigy                  # (2)
sigx    = 1.0 / (1 + math.exp(-x))  # (3)
xpy     = x + y                     # (4)
xpysqr  = xpy ** 2                  # (5)
den     = sigx + xpysqr             # (6)
invden  = 1.0 / den                 # (7)
f       = num * invden              # (8)
```
我们这样组织代码，并使用多个中间变量，是由于每个中间变量都是由简单的计算得来的，并且我们很容易能够计算它的局部梯度。那么反向计算也会变得非常简单，现在我们就按上面的相反的步骤进行运算。同时，反向计算的每一步，都包含局部梯度的计算，并使用链式乘法将梯度串起来。每一行都标注了属于哪个部分的反向运算。
``` python
# backprop f = num * invden
dnum    = invden                            # (8)
dinvden = num                               # (8)
# backprop invden = 1.0 / den
dden    = (-1.0 / (den ** 2)) * dinvden     # (7)
# backprop den = sigx + xpysqr
dsigx   = (1) * dden                        # (6)
dxpysqr = (1) * dden                        # (6)
# backprop xpysqr = xpy ** 2
dxpy    = (2 * xpy) * dxpysqr               # (5)
# backprop xpy = x + y
dx      = (1) * dxpy                        # (4)
dy      = (1) * dxpy                        # (4)
# backprop sigx = 1.0 / (1 + math.exp(-x))
dx     += ((1 - sigx) * sigx) * dsigx       # (3)
# backprop num = x + sigy
dx     += (1) * dnum                        # (2)
dsigy   = (1) * dnum                        # (2)
# backprop sigy = 1.0 / (1 + math.exp(-y))
dy     += ((1 - sigy) * sigy) * dsigy       # (1)
```
