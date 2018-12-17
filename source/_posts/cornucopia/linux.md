---
title: Linux操作系统相关笔记
date: 2018/12/16 16:39:46
tags:
  - Linux
category: 杂项
description: 该文章记录Linux使用中的一些基本操作，现在有安装字体的方法。
---

## Linux安装字体
Linux字体都在/usr/share/fonts这个目录下面，可对某种字体新建一个文件夹，将字体拷贝到文件夹下，然后使用`fc-cache -fv`命令将字体进行缓存，然后就能够使用了。
