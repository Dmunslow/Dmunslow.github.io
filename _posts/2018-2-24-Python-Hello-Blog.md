---
layout: post
title:  "Hello Blog - My First Python Post"
date:   2017-09-08 23:00
category: python
icon: python
keywords: python, numpy, matplotlib
preview: 0
---

```python
# this is a test page for the blog
import numpy as np
import matplotlib.pyplot as plt

# create array of numbers and years
numbers = [180, 215, 232, 225, 200, 276, 300, 369]
year = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008]

# create plot
plt.plot(year, numbers)

# display the plot
plt.show()


## Authors Note: this documnet was converted from a Jupyter Notebook to markdown, using nbconvert

```


![png]({{blog.duncanmunslow.com}}/_posts/images/test_0_0.png)

