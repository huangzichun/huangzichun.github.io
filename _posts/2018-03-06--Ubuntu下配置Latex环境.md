---
layout:     post
title:      Ubuntu下配置Latex环境
subtitle:   附带Latex中文字体配置
date:       2018-03-06
author:     HC
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - Ubuntu
    - Latex
---

> 为了能在Ubuntu下用Latex写毕设论文，特此记录下Ubuntu下的Latex相关配置

# 1. 安装Latex和TexMaker

Ubuntu下安装Latex比较简单，执行：

```shell
sudo apt-get install texlive-full
```

这个包比较大，要占3G左右的空间。不过好在实验室网速还不错。装好之后安装TexMaker，也是一命令就行

```shell
sudo apt-get install texmaker
```

# 2. 安装中文字体

编译论文时中文报错：

```shell
pdfTex error: pdflatex.exe (file simsun.ttc): cannot open TrueType font file for reading
```

百度一下，发现是中文字体的问题，需要额外安装一下常用的中文字体。

## 2.1 下载常用字体文件和gbkfonts程序

首先需要下载Win下常用的字体文件，比如simsun.ttc，simhei.ttf，simkai.ttf，simsong.ttf等等。然后下载gbkfonts程序。这个程序据说是用来从TTF 汉字字体生成TeX 使用的汉字Type1 字体的转换软件。将下载好的这个程序放到/usr/bin目录下，赋予可执行权限。

## 2.2 用gbkfonts生成字体映射

1. 新建目录/usr/share/texmf/fonts/truetype/chinese/，用于存放上一步下载好的字体文件

2. 生成映射

   ```shell
   cd /usr/share/texmf/
   sudo gbkfonts simfang.ttf fs
   sudo gbkfonts simsun.ttc song
   sudo gbkfonts simkai.ttf kai
   sudo gbkfonts simhei.ttf hei
   ```

## 2.3 刷新 

```shell
sudo texhash
```



然后就ok咯