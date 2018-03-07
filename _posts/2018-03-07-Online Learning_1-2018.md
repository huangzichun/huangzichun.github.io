---
layout:     post
title:      Online Learning (1)
subtitle:   Introducation
date:       2018-03-07
author:     HC
header-img: img/post-bg-digital-native.jpg
catalog: true
tags:
    - Online Learning
    - Off-line Learning
---

> Online Learning笔记系列

# 1. Offline Learning

​	传统的机器学习是offline的。大致的步骤为：给定训练数据，训练模型，再将模型应用在测试数据上。这样做的前提是假设了训练数据与测试数据源于统一分布。但是在实际问题中，往往是不成立的，现实生活中的数据是实时到来，数据分布也会有所变化。比如一般来说，用户一年四季的购物喜好是不同的。对于这种数据流数据，传统机器学习会有很多问题，具体来说有：

1. **Evolving / Concept drift**

   ​	数据的分布往往是随着时间实时变化或演化的。比如一年中顾客购买衣物的习惯是随着季节而变化的，这种变化被称为概念漂移。一般来说，概念漂移现象很普遍，表现为数据的分布变化$P(X)$，或者后验分布变化$P(Y||X)$。从变化的快慢又可以分为渐变的概念漂移和突变的概念漂移。从字面上可以理解，前者变化缓慢，后者变化快速，这类概念漂移往往难以捕获，难以和噪声数据所带来的影响加以区别。

2. **Constraints in terms of memory and running time**

   ​	由于数据吞吐量与计算机硬件的限制，离线模型通常难以处理大规模的数据。当然，计算能力的问题可以通过分布式机器学习算法得到解决。对于变化的数据，采取离线更新模型的方式（这里的更新是指以离线的方式，在新训练集上重新训练模型，再放回线上应用场景之中），但是，新训练数据随着时间不断增大，模型训练将变得更加费时。

3. **Trade-off between Accuracy and Efficiency** 

   ​	如果要离线重新训练模型，那么为了保持效率，则必须牺牲模型精度。比如采用一定时间内的数据来做模型的重新训练等等

# 2. Online Learning

​	在线学习则是以当前数据驱动的方式，实时更新模型（这里的更新不是指重新训练模型，而是根据当前数据，对模型对必要的实时改修）。所以在线学习的优势体现在了他的Real-time update and prediction和Data Scalability之上。

$1/2 * 5 \mathbf{55}$


