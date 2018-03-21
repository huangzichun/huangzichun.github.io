---
layout:     post
title:      Paper Reading on Multi-label Classification
subtitle:   Adversarial Extreme Multi-label Classification
date:       2018-03-20
author:     HC
header-img: img/post-bg-digital-native.jpg
catalog: true
tags:
    - Paper Reading
    - Multi-label Classification
    - Machine Learning
---

> 今天组会，小伙伴分享的论文，赶紧做一波笔记

# 1. Background & Motivation

## 1.1 Multi-label Classification

我们先唠叨一下背景知识。

所谓的多标签分类问题，指的是一个instance $X$同时被打上多个标签。各个标签之间可以是有联系，也可以是独立的。比如，$X=$周杰伦，那么他的标签就可以是人，男人，歌手等等。如果把各个数据对应的标签组合看成一个新的标签，那么multi-label的问题，**等价于multi-class 的问题**，对应的class的大小为$2^{\mid Y \mid}$ 。

一般来说，如果假设各个标签之间相互独立，那么multi-label的问题可以通过**1 vs all**的方法解决，或者通过**ensemble**的方式。换句话说就是将各个标签割裂开来，分别单独训练一个预测模型。但是，对应第一种策略来说，标签的搜索空间太大，第二种转换模型的训练太复杂，所以Multi-label的key problem 之一是效率问题，更体现在模型的scalability，以及数据空间和标签空间的high dimension上。对于这种feature和label space的高维问题，一般称为Extreme multi-label classification

另一方面，前面所提及的两种策略都是割裂标签进行单独分析。一种更好的方式是考虑标签之间的相似性，或者说标签数据的结构信息（层次，包含，独立，关联等等），所以Multi-label的Key Problem 之二是怎么使用多标签信息帮助分类，怎么与数据X的信息进行整合。

当然，多标签问题同样也面临label sparsity和label imbalance等等其他问题。前者体现在label space的维度很高，但是一个数据$X$的标签可能就只有几个。而后者强调了tail label的问题，即很多的标签只有极其少的数据才具有。比如标签空间中有26种标签：[A, B, C, D, ..., X, Y, Z]，我们构建如下的数据与标签的矩阵M。如果数据$i$有标签$j$，那么$M_{ij}=1$

如果从数据的角度来看，大多数数据只是同时属于其中的两到三个类别。那么，这就是label sparsity，提现在矩阵$M$是稀疏的。如果从标签的角度来看，相比于其他标签，标签$Z$只对应了极少量的数据，那么$Z$就是一个tail label。

## 1.2 Motivation

根据题目来看，这篇文章应该是要做在含有大量tail label的条件下的Extreme multi-label问题。

如果分类器能在major label上保证很高的正确率，那么为什么还要关注tail label的预测正确与否？这个问题很像label imbalance，从现实背景出发，tail label在现实的场景中是很多的，并且这部分的tail label信息对具体应用来说是很重要的。比如在推荐系统中，漫威和DC的电影很多人都喜欢看，但是，这并不意味着我们不需要推荐雷锋侠了吧。另一方面，从模型训练的角度，通常，由于tail label的存在会影响模型的训练结果。tail label就好比在刻画两个数据的标签相似度的噪声数据。当然，这是一个具有很高营养价值的噪声，不能忽略。

为了解决这个问题，本文是把tail label的问题转换成了Adversarial Learning的问题，然后通过robust optimization来求解。文章认为，tail label存在的原因就是该类别标签对应的（训练数据和测试）数据太少了。说白了，（个人观点，仅供参考）作者认为tail label是敌手引入的Adversarial data，这部分数据是”有歧义，有问题“的噪声。所以，问题转为给定一些正常数据和一些敌手的噪声数据，我们需要建立一个对这部分噪声鲁棒的模型，使得在test data上也能判断出来他们的真实标签（即哪些是major标签，哪些是tail label）。



# 2. Method

> 相传robust optimization领域有一句motto：
>
> **Robust optimization is to harness the consequences of probability theory without paying the computational cost of having to use its axioms.**

本文首先采用标签割裂的思想，把标签分割开来，单独学习一个分类器。这让模型可以更直接的去处理tail label，但是，就extreme问题来说，除非采用并行策略，否则训练复杂度是很大的。

不管怎么样，问题已经变成了一个对单一标签的robust optimization的问题（robust optimization就是在数据或者参数存在不确定性的情况下的优化问题）。这里的robust是通过间接加入了“扰动数据”来实现的：

$$
\min_\limits{w}\max_\limits{(x^0_1, x^0_2, ..., x^0_N)}\sum_{i=1}^N \max(0, 1-s_i(<w,x_i-x^0_i>))
$$

式子中的$x^0$是扰动数据，并且假设uncertainty set 是$X^0=\{(x^0_{1}, x^0_{2}, ..., x^0_{N}) \mid \sum_{i=1}^N \mid \mid x^0_{i} \mid \mid _\infty  < \lambda' \}$ 。

这里用$\lambda'$参数限制了扰动的大小，$s_{i}$表示数据$x_{i}$的标签中是否含有对应的标签，含有则为1，否则是-1，而$w$是模型参数。这个优化问题是min-max，这里的max是说，在给定模型w的时候，要找到一组扰动数据，使得求和那一坨误差最大化，而最大化出现的条件就是当数据$x_{i}$存在某个标签的时候（$s_{i}=1$），扰动数据 $x^0_{i}$ 要和真实数据 $x_{i}$ 的预测结果要相似；反之，当数据 $x_{i}$ 不存在某个标签的时候（ $s_{i}=-1$ ），扰动数据 $x^0_{i}$ 要和真实数据 $x_{i}$ 的预测结果要有差异。然后外层对w求min，旨在找出一个模型能竟可能的正确分类，最小分类误差。

具体的求解方法，这里就不描（Chao）述（Xie）了。作者是将上式转换成另一个等价的形式，这个形式可以看成是regularized SVM。具体方法可以去看看另一篇文章 Robustness and Regularization of Support Vector Machines (JMLR'09)。（你会发现就方法论上，本文没有啥亮点，不过问题还是好的哈）。最后，作者采用了比较经典的Proximal gradient优化方法来求解，关于这个解法的组会ppt，可以在实验室网站上下载。

总的来说，对于每一个label，本文都会训练一个模型，并且这个模型使用了L1范数来约束，在一定程度上能处理高维的feature，但是label space的高维的问题，文章处理得不是很完美，甚至完全没考虑？另一方面，对于tail label的处理问题，文章将label进行割裂，单独处理每一个label的预测问题。所以tail label也受到了单独的robust的优化处理。



# Doubts

1. 类比class imbalance问题，文章对每个label进行单独建模，但是对于tail label来说，$s_{i}=1$的情况是很少的。这种情况下的优化问题会不会被其他数据($s_{i}=-1$) 的信息所掩盖呢？robust optimization会有什么帮助？
2. 在实验部分，文章的考核指标赋予了tail label很高的权重，所以实验结果不能表明整个模型对major label的影响。因为个人对实验中采用的对比算法不了解，合理性就不分析了
3. 感觉extreme问题和tail label问题，本文没有太突出







# Reference

[1]. Babbar, Rohit, and Bernhard Schölkopf. "Adversarial Extreme Multi-label Classification." *arXiv preprint arXiv:1803.01570* (2018).

