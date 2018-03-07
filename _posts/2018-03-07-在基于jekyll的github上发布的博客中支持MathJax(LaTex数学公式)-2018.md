---
layout:     post
title:      在基于jekyll的github上发布的博客中支持MathJax
subtitle:   LaTex数学公式
date:       2018-03-07
author:     HC
header-img: img/post-bg-digital-native.jpg
catalog: true
tags:
    - jekyll
    - MathJax
---

[^知乎答案]: https://www.zhihu.com/question/62114522

在head.html中加入以下代码[^知乎答案]

```html
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
      inlineMath: [['$','$']]
    }
  });
</script>
```


