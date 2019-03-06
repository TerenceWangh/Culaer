---
title: 如何安装TensorFlow
date: 2013/03/06 11:04：14
tags:
  - TensorFlow
category: tensorflow
description: 介绍如何在Linux编译并安装TensorFlow。
---

## TensorFlow

[Github主页](https://github.com/tensorflow/tensorflow)。

## 依赖环境

### Python依赖：
- python开发环境  
  `sudo apt install python-dev python-pip`
- python包  
  `pip install -U --user pip six numpy wheel mock`  
  `pip install -U --user keras_applications==1.0.6 --no-deps`  
  `pip install -U --user keras_preprocessing==1.0.5 --no-deps`  
- CUDA环境，按照需要安装。
- **Bazel**编译器，[参考](https://docs.bazel.build/versions/master/install.html)。另外，anaconda也支持安装bazel编译器。*不同版本的TensorFlow可能依赖不同的版本的Bazel，出现问题自行更换bazel。*

## 安装步骤

1. 下载源代码，去Github的[Release](https://github.com/tensorflow/tensorflow/releases)下载压缩文件。
1. 解压缩，并进入解压缩的文件夹。
1. 进行基本的配置：  
  `./configure`，主要配置Python的路径以及CUDA的位置。其他的不需要模块全部选择`N`。
1. 配置完成后进行编译：  
  不使用CUDA：`bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package`  
  使用CUDA：`bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package`
1. 编译过程需要花费很长的时间，编译完成后，还需要执行下列命令进行打包才能使用`pip`进行安装：  
  `./bazel-bin/tensorflow/tools/pip_package/build_pip_package <路径>`
1. 到指定的路径下面执行`pip install <包名>`即可。