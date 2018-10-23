---
title: Tensorflow
date: 2018/10/14 16:28:30
tags:
  - TensorFlow
category: Study
---

# Tensorflow

## 计算图的使用
TensorFlow程序一般可以分为两个阶段，在第一阶段定义计算图所有的计算，第二阶段执行计算。

### 定义计算图中所有的计算：
1. TensorFlow中维护的集合列表

集合名称 | 集合内容 | 使用场景
------------|--------------|-----------
tf.GraphKeys.VARIABLE | 所有变量 | 持久化TensorFlow模型
tf.GraphKeys.TRAINABLE_VARIABLES | 可学习的变量 | 模型训练、生成模型可视化内容
tf.GraphKeys.SUMMARIES | 日志生成相关的张量 | TensorFlow计算可视化
tf.GraphKeys.QUEUE_RUNNERS | 处理输入的QueueRunner | 输入处理
tf.GraphKeys.MOVING_AVERAGE_VARIABLES | 所有计算了滑动平均值的变量 | 计算变量的滑动平均值


2. TensorFlow数据模型--**张量**
    >TensorFlow计算的结果不是一个具体的数字，而是一个张量的结构。一个张量包括三个属性，**名字**、**维度**和**类型**

    张量数据类型：实数(`tf.float32, tf.float64`)、整数(`tf.int8, tf.int16, tf.int32, tf.int64, tf.uint8`)、布尔型(`tf.bool`)以及复数(`tf.complex64, tf.complex128`)

    例如，可定义下面计算，**注意要保证数据类型的一致**
    ```
    result = tf.constant([1., 2.], name='a') + tf.constant([2., 3.], name='b')
    ```

### 执行计算 
1. TensorFlow运行模型--**会话**
    上面的计算只是定义了计算，得到的结果仅仅是一个**张量**，并不是一个确定的数值：
    `<tf.Tensor 'add_1:0' shape=(2,) dtype=float32>`
    如果要得到一个确定的结果，需要使用会话进行计算：
    ``` Python
    # create a session to calculate
    with tf.Session() as sess:
        sess.run(result)
    ```

## TensorFlow实现简单的神经网络
1. 定义神经网络的参数
    ``` Python
    # define the parameters
    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

    # define the features and labels
    x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
    y = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')
    ```
2. 定义神经网络的前向传播过程
    ``` Python
    a = tf.matmul(x, w1)
    pred = tf.matmul(a, w2)
    ```

3. 定义损失函数和反向传播算法
    ``` Python
    loss = -tf.reduce_mean(y * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
    ```
    这里使用了**Adam**优化器，常用的优化器还有**SGD**等

4. 随机生成数据集
    ``` Python
    rdm = numpy.random.RandomState(1)
    dataset_size = 128
    X = rdm.rand(dataset_size, 2) # The shape of inputs is 2
    Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]
    ```

5. 进行训练
    ``` Python
    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init_op)

        STEPS = 5000
        for i in range(STEPS):
            start = (i * batch_size) % dataset_size
            end = min(start + batch_size, dataset_size)

            session.run(optimizer, feed_dict={x: X[start:end], y: Y[start: end]})

            if i % 1000 == 0:
                total_cross_entropy = session.run(
                    loss, feed_dict={x: X, y: Y})
                print('After %d training step(s), cross entropy on all data is %g' % (i, total_cross_entropy))
    ```