# Objects Detection Algorithms
Building Neural Networks for Objects Detection Algorithms.
<br/>**Under construction. Coming soon. Will be released in May 2019.**
<br/>[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1317904.svg)](https://doi.org/10.5281/zenodo.1317904)

## Content
Theory and experimental results (on this page):

* [Objects Detection Algorithms Overview](#main-objects-detection-algorithms)
  * [YOLO v1](#yolo-v1)
  * [YOLO v2](#yolo-v2)
  * [YOLO v3](#yolo-v3)

<br/>

### <a id="main-objects-detection-algorithms">Objects Detection Algorithms Overview</a>
There are variety of algorithms for Detection Objects on the image. Let's consider the most popular ones like YOLO of versions 1, 2 and 3. Also, there is such algorithm as SSD and others.

<br/>

### <a id="yolo-v1">YOLO v1</a>

Consider following part of the code (related file soon.py):
```py
import numpy as np
```

<br/>

### <a id="yolo-v2">YOLO v2</a>

Consider following part of the code (related file soon.py):
```py
import numpy as np
```

<br/>

### <a id="yolo-v3">YOLO v3</a>

Consider following part of the code (related file soon.py):
```py
# Importing necessary libraries
import numpy as np
import cv2
import time

# Loading COCO class labels from file
# Opening file, reading, eliminating whitespaces, and splitting by '\n', which in turn creates list
labels = open('yolo-coco-data/coco.names').read().strip().split('\n')  # list of names

```


<br/>

### MIT License
### Copyright (c) 2019 Valentyn N Sichkar
### github.com/sichkar-valentyn
### Reference to:
Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform. DOI: 10.5281/zenodo.1317904
