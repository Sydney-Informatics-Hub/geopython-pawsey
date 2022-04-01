```python

```


```python
# Disable some warnings produced by pandas etc.
# (Don't do this in your actual analyses!)
import warnings
warnings.simplefilter('ignore', category=UserWarning)
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=RuntimeWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from sklearn import preprocessing
from sklearn import model_selection

#You probably do not have this library! Install it with pip:
#!pip install UpSetPlot
import upsetplot
%matplotlib inline
sns.set(font_scale = 1.5)
```
