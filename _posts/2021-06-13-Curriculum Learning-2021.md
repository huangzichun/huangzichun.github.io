---
layout:     post
title:      Curriculum Learning
subtitle:   课程学习の笔记
date:       2021-06-13
author:     HC
header-img: img/guimie_1.jpg
catalog: true
tags:
    - Curriculum Learning
    - Self-paced Learning
    - Automatic Curriculum Learning
    - Curriculum Sequencing
    - Curriculum Generation
---

> 认识了一个大佬，连续几年坚持在CSDN上更新技术文章，很是佩服。
>
> 因此，我也想把博客再维护起来，多总结总结，多思考思考

# 0. 说在前面的话

最近在学习一个小项目的时候，接触到了课程学习（Curriculum Learning），简单做了个Survey之后才发现，以前了解的自步学习（Self-paced Learning）也属于课程学习的范畴。因此，借此机会也整理一下课程学习的相关内容，便于以后深入学习。



由于深度模型能很好地支持模型结构的可视化，极大降低了学习门槛，再加上业界数据和计算性能的加成，Deep Learning无论是在学术界还是工业界，已经是飞速发展。只要给定数据和计算资料，我们便可以一股脑的把数据喂给模型，经过调参之后，模型输出我们预定的目标。这其中值得探索的问题是，人类或者动物在学习一个知识或者一个任务的时候，并不是毫无章法的。我们总是先从简单的知识开始，并逐渐去学习更加复杂的知识或概念，就像先上小学、中学后，再读大学一样，直接让一个小学生学习微积分怕是难上加难。这就是课程学习想要解决的问题，在训练模型的时候，训练样本的顺序会影响最终的学习效果，甚至模型收敛效率。课程学习的概念是2009年由Bangio提出，他认为模型训练的时候，应该先训练简单的数据，然后不断的提升数据复杂度。



**TO BE CONTINUE**

# 1. 课程与课程学习

## 1.1 课程 Curricula

课程是什么

## 1.2 课程学习 CL

课程学习是什么，有两类学习范式



课程学习中的三个组成部分

### 1.2.1 课程生成 Curriculum Generation

### 1.2.2 课程排序 Curriculum Shedule/Sequencing

### 1.2.3 课程知识迁移 Curriculum Knowledge Transfer



## 1.3 评估方法

- Time to threhold
- Asymptotic Performance
- Total Reward
- JumpStart



## 1.4 相似领域

- Active Learning
- Hard Example Mining
- Anti-Curriculum Learning



# 2. 课程学习算法

- Vanilla CL
- Self-paced Learning
- Balanced CL
- Self-paced CL
- Teacher-Student CL
- Progressive CL





# 3. CL for Reinforcement Learning

## 3.1 定义

## 3.2 Why CL for RL

## 3.3 What CL Control

## 3.4 What CL Optimize

## 3.5 分类

- intermediate task generation
- Curriculum representation
- Transfer Method
- Curriculum adaptivity
- Curriculum Sequencer
- Evaluation metric







## 3.6 算法

- Task Generation
- Sequencing
  - Sampling Sequencing
  - co-learning
  - reward and intial / terminal state distribution changes
  - no restriction
    - MDP-based Sequencing
    - Combinatorial optimization and search
    - Graph-based Sequencing
    - Auxiliary Problem
  - human in the loop
- Transfer Learning





# 4. Open Questions

- Task Knowledge Transfer
- Task Generation Automatically
- Reusing Curricula
- Task Generation + Sequencing
- Theoretical Results
- Understanding General Principles for Curriculum Design




