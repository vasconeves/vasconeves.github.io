---
layout: post
title: Basic image manipulation using numpy
image: "/posts/camaro.jpg"
tags: [Python, Numpy, Image manipulation, Camaro]
---

In this micro-project we will learn how to use `numpy`to perform basic image manipulation.

---

To this end we will need to import three libraries: `numpy`, `skimage` for input/output functionality, and `matplotlib` for visualization.

```python
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
```

We will be using the poor yellow camaro shown above as a guinea pig for our manipulations...

Yellow is a very nice color to play with as it is mostly composed of green and red colors in the RGB scale.

But first things first. We need to import this .jpg image as something that can be interpretable by `numpy` and `pyplot`. To that end we will use the `io.imread` function. If we print out the output variable we obtain a 1200x1600x3 numpy array as shown below using the `shape` function.

```py
camaro = io.imread("camaro.jpg")
print(camaro)
[[[ 83  81  43]
  [ 57  54  19]
  [ 34  31   0]
  ...
  [179 144 112]
  [179 144 114]
  [179 144 114]]

 [[ 95  93  55]
  [ 72  69  34]
  [ 46  43   8]
  ...
  [181 146 114]
  [181 146 116]
  [182 147 117]]

 [[101  99  61]
  [ 88  85  50]
  [ 67  63  28]
  ...
  [184 149 117]
  [184 149 117]
  [184 149 119]]

 ...

 [[ 12  10  11]
  [ 12  10  11]
  [ 12  10  11]
  ...
  [ 28  27  25]
  [ 27  26  24]
  [ 27  26  24]]

 [[ 12  10  11]
  [ 12  10  11]
  [ 11   9  10]
  ...
  [ 28  27  25]
  [ 27  26  24]
  [ 27  26  24]]

 [[ 13  11  12]
  [ 12  10  11]
  [ 10   8   9]
  ...
  [ 28  27  25]
  [ 27  26  24]
  [ 26  25  23]]]

camaro.shape
(1200, 1600, 3) 
```

To visualize our input image we just need to use `pyplot`

```py
plt.imshow(camaro)
```

![](/img/posts/camaro_original.png)

Ok, now that we have our image loaded into memory we can play around with it. Let's start by cropping the image!

### Cropping

Let's first do an horizontal crop. To do this,we slice the image using the indexes as shown below. The other dimensions do not change, hence the semicolon.

```py
cropped = camaro[0:500,:,:]
plt.imshow(cropped)
```

![](/img/posts/crop_hor.png)

We can also perform a vertical crop...

```py
cropped = camaro[:,400:1000,:]
plt.imshow(cropped)
```
![](/img/posts/crop_ver.png)

...and finally a crop that encompasses what we want: the camaro!

```py
cropped = camaro[350:1100,200:1400,:]
plt.imshow(cropped)
```

![](/img/posts/camaro_crop.png)

### Flipping

We can also easily flip our image. 

To do a vertical flip we just need to invert the values in the horizontal direction as shown below.

```py
vertical_flip = camaro[::-1,:,:]
plt.imshow(vertical_flip)
```

![](/img/posts/flip_vert.png)

To perform an horizontal flip we do the same, but this time on the vertical direction.

```py
horizontal_flip = camaro[:,::-1,:]
plt.imshow(horizontal_flip)
```

![](/img/posts/flip_hor.png)

It **looks** similar but it is not! Look **carefully**!

### Color channels

As you've noticed we have three layers or dimensions in this image. The three layers are its RGB engoding, where R is red, G is green and B is blue. 

The first layer is red...

```py
red = np.zeros(camaro.shape,dtype="uint8")
red[:,:,0] = camaro[:,:,0]
plt.imshow(red)
```

![](/img/posts/camaro_red.png)

...the second layer is green...

```py
green = np.zeros(camaro.shape,dtype="uint8")
green[:,:,1] = camaro[:,:,1]
plt.imshow(green)
```

![](/img/posts/camaro_green.png)

...and the third layer is blue!

```py
blue = np.zeros(camaro.shape,dtype="uint8")
blue[:,:,2] = camaro[:,:,2]
plt.imshow(blue)
```

![](/img/posts/camaro_blue.png)

And we can observe that the **yellow** camaro almost has no blue in it, as expected!

### Rainbow camaro!

To make it cool, let's stack the red, green, and blue images together! We can use `np.vstack` for this.

```py
camaro_rainbow = np.vstack((red,green,blue))
plt.imshow(camaro_rainbow)
```

![](/img/posts/camaro_rainbow.jpg)

**The image looks small but it is not!**

Now, we just need to print it and hang it on the wall!

Safe travels! ;)

---

*Image from wikimedia commons*
