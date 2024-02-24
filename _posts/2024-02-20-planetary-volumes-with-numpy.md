---
layout: post
title: Finding the Volumes of Planets
image: "/posts/solar_system_wikimedia_commons.jpg"
tags: [Python, Numpy, Planets, Astronomy,Science]
mathjax: true
---
v2
In this micro-project we will explore the basic functionality of `numpy` when calculating the Volume of planets.

---

First, let's import `numpy` as `np`.

```py
import numpy as np
```

Now, we know that a volume of a solid is $V = \frac{4}{3}\pi r^2$.

We can write the radii of the planets of the solar system in kilometers in a `numpy array` as

```py
radii = np.array([2439.7,6051.8,6371,3389.7,69911,58232,25362,24622]) #in km
```

Then, we can calculate the volume of the planets all at the same time with numpy without using any cycles!

```py
volumes = 4/3 * np.pi * radii**3
print(volumes)
[6.08272087e+10 9.28415346e+11 1.08320692e+12 1.63144486e+11
 1.43128181e+15 8.27129915e+14 6.83343557e+13 6.25257040e+13]
```

we can extend this logic and calculate, for instance, the volume of 1 million planets with random radii between 1 and 1000 km! 

```py
radii = np.random.randint(1,1000,1000000)
volumes = 4/3 * np.pi * radii**3
```

How long does this take? Just a small fraction of a second!

---
*Image sourced at* | [https://en.wikipedia.org/wiki/Solar_System]([https://en.wikipedia.org/wiki/Solar_System)




