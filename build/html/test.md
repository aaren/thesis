---
layout: page
title: test
---

Chapter title
=============

First bit
---------

Here are some words about things.

Here is an equation:

$$
\begin{equation}
\exp^{i\pi} + 1 = 0
\label{eq:life}
\end{equation}
$$

Second bit
----------

As we can see in #eq:life, maths is very useful.


Figures
-------

Let's try and generate a figure from code:

```{.python .input}
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
```

```{.python .input #fig:test caption='a lovely caption'}
x = np.linspace(2, 5)
plt.plot(x, x ** 2)
print "hello from ipython!"
```

Here is a reference to #fig:test.
