# Exploratory Data Analysis (EDA) for ML

<div class="objectives">  
### Objectives

- Introduce some of the key packages for EDA and ML.
- Introduce and explore an dataset for ML
- Clean up a dataset
- Install additional Python libraries
</div>


First, let’s load the required libraries. We will use the sklearn library for our ML tasks, and the pandas, numpy, matplotlib seaborn and upsetplot libraries for general data processing and visualisation.


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from sklearn import preprocessing
from sklearn import model_selection
import upsetplot
#%matplotlib inline
sns.set(font_scale = 1.5)
```

## Load the data

Download the data and put it in your "data" folder. You will have to download it from the [GitHub repo](https://github.com/natbutter/gawler-exploration/blob/master/ML-DATA/training_data-Cu.csv) (right click on the Download button and select "Save link as.."). Our data is based on a submitted Manuscript (Butterworth and Barnett-Moore 2020) which was a finalist in the [Unearthed, ExploreSA: Gawler Challenge](https://unearthed.solutions/u/competitions/exploresa).

The dataset contains a mix of categorical and numerical values, representing various geophysical and geological measurements across the Gawler Craton in South Australia. 


```python
#ameshousingClean = pd.read_csv('data/AmesHousingClean.csv')
#ameshousingClean = pd.read_csv('../data/training_data-DIA.txt')
#training_data-Cu.txt

#Read in the data
#Set a value for NaNs
#Drop many of the columns (so it is easier to work with)
df = pd.read_csv('../data/training_data-Cu.txt',na_values='-9999.0')
cols = list(range(5,65))
cols.insert(0,0)
df.drop(df.columns[cols],axis=1,inplace=True)

df=df.astype({'archean27':'object','geol28':'object','random':'int64','deposit':'int64'})

```

## Exploratory data analysis

Exploratory data analysis involves looking at:

- the distribution of variables in your dataset
- whether any data is missing
- skewed
- correlated variables



```python
#What are the dimensions of the data?
df.shape
```




    (3138, 38)




```python
#Look at the data:
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lon</th>
      <th>lat</th>
      <th>res-25</th>
      <th>res-77</th>
      <th>res-309183</th>
      <th>neoFaults</th>
      <th>archFaults</th>
      <th>gairFaults</th>
      <th>aster1-AlOH-cont</th>
      <th>aster2-AlOH</th>
      <th>...</th>
      <th>mag21-tmi</th>
      <th>rad22-dose</th>
      <th>rad23-k</th>
      <th>rad24-th</th>
      <th>rad25-u</th>
      <th>grav26</th>
      <th>archean27</th>
      <th>geol28</th>
      <th>random</th>
      <th>deposit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>129.106649</td>
      <td>-26.135900</td>
      <td>1.9959</td>
      <td>1.9935</td>
      <td>2.5780</td>
      <td>0.858696</td>
      <td>0.874997</td>
      <td>2.718781</td>
      <td>1.907609</td>
      <td>NaN</td>
      <td>...</td>
      <td>-88.364891</td>
      <td>34.762928</td>
      <td>1.269402</td>
      <td>6.065621</td>
      <td>38.492386</td>
      <td>27.176790</td>
      <td>14552</td>
      <td>17296</td>
      <td>999</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>132.781571</td>
      <td>-26.151144</td>
      <td>2.0450</td>
      <td>2.0651</td>
      <td>2.3873</td>
      <td>0.607134</td>
      <td>0.936479</td>
      <td>1.468679</td>
      <td>2.032987</td>
      <td>1.076198</td>
      <td>...</td>
      <td>-190.025864</td>
      <td>89.423668</td>
      <td>3.169631</td>
      <td>15.980172</td>
      <td>56.650471</td>
      <td>-83.541550</td>
      <td>14552</td>
      <td>17068</td>
      <td>-999</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>132.816676</td>
      <td>-26.159202</td>
      <td>2.0450</td>
      <td>2.0651</td>
      <td>2.3873</td>
      <td>0.577540</td>
      <td>0.914588</td>
      <td>1.446256</td>
      <td>1.982274</td>
      <td>1.050442</td>
      <td>...</td>
      <td>-251.018036</td>
      <td>75.961006</td>
      <td>2.525403</td>
      <td>15.625917</td>
      <td>58.361298</td>
      <td>-81.498817</td>
      <td>14552</td>
      <td>17296</td>
      <td>-999</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>128.945869</td>
      <td>-26.179362</td>
      <td>1.9978</td>
      <td>1.9964</td>
      <td>2.6844</td>
      <td>0.810394</td>
      <td>0.826784</td>
      <td>2.813603</td>
      <td>1.947705</td>
      <td>NaN</td>
      <td>...</td>
      <td>873.983521</td>
      <td>46.321651</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>50.577263</td>
      <td>33.863503</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-999</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>132.549807</td>
      <td>-26.185500</td>
      <td>2.0694</td>
      <td>2.0999</td>
      <td>2.3574</td>
      <td>0.652131</td>
      <td>1.026991</td>
      <td>1.499793</td>
      <td>1.977050</td>
      <td>1.064977</td>
      <td>...</td>
      <td>71.432777</td>
      <td>47.194534</td>
      <td>2.367707</td>
      <td>6.874684</td>
      <td>29.794928</td>
      <td>-90.970375</td>
      <td>14552</td>
      <td>17296</td>
      <td>-999</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>




```python
#What types are each of the columns?
df.dtypes
```




    lon                     float64
    lat                     float64
    res-25                  float64
    res-77                  float64
    res-309183              float64
    neoFaults               float64
    archFaults              float64
    gairFaults              float64
    aster1-AlOH-cont        float64
    aster2-AlOH             float64
    aster3-FeOH-cont        float64
    aster4-Ferric-cont      float64
    aster5-Ferrous-cont     float64
    aster6-Ferrous-index    float64
    aster7-MgOH-comp        float64
    aster8-MgOH-cont        float64
    aster9-green            float64
    aster10-kaolin          float64
    aster11-opaque          float64
    aster12-quartz          float64
    aster13-regolith-b3     float64
    aster14-regolith-b4     float64
    aster15-silica          float64
    base16                  float64
    dem17                   float64
    dtb18                   float64
    mag19-2vd               float64
    mag20-rtp               float64
    mag21-tmi               float64
    rad22-dose              float64
    rad23-k                 float64
    rad24-th                float64
    rad25-u                 float64
    grav26                  float64
    archean27                object
    geol28                   object
    random                    int64
    deposit                   int64
    dtype: object




```python
#Get information about index type and column types, non-null values and memory usage.
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3138 entries, 0 to 3137
    Data columns (total 38 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   lon                   3138 non-null   float64
     1   lat                   3138 non-null   float64
     2   res-25                3138 non-null   float64
     3   res-77                3138 non-null   float64
     4   res-309183            3138 non-null   float64
     5   neoFaults             3138 non-null   float64
     6   archFaults            3138 non-null   float64
     7   gairFaults            3138 non-null   float64
     8   aster1-AlOH-cont      2811 non-null   float64
     9   aster2-AlOH           2010 non-null   float64
     10  aster3-FeOH-cont      1811 non-null   float64
     11  aster4-Ferric-cont    2811 non-null   float64
     12  aster5-Ferrous-cont   1644 non-null   float64
     13  aster6-Ferrous-index  2811 non-null   float64
     14  aster7-MgOH-comp      1644 non-null   float64
     15  aster8-MgOH-cont      1811 non-null   float64
     16  aster9-green          3129 non-null   float64
     17  aster10-kaolin        1811 non-null   float64
     18  aster11-opaque        753 non-null    float64
     19  aster12-quartz        3130 non-null   float64
     20  aster13-regolith-b3   3127 non-null   float64
     21  aster14-regolith-b4   3073 non-null   float64
     22  aster15-silica        3130 non-null   float64
     23  base16                3135 non-null   float64
     24  dem17                 3133 non-null   float64
     25  dtb18                 1490 non-null   float64
     26  mag19-2vd             3132 non-null   float64
     27  mag20-rtp             3132 non-null   float64
     28  mag21-tmi             3132 non-null   float64
     29  rad22-dose            2909 non-null   float64
     30  rad23-k               2900 non-null   float64
     31  rad24-th              2904 non-null   float64
     32  rad25-u               2909 non-null   float64
     33  grav26                3131 non-null   float64
     34  archean27             3135 non-null   object 
     35  geol28                3135 non-null   object 
     36  random                3138 non-null   int64  
     37  deposit               3138 non-null   int64  
    dtypes: float64(34), int64(2), object(2)
    memory usage: 931.7+ KB



```python
#Explore how many null values are in the dataset
df.isnull().sum(axis = 0)
```




    lon                        0
    lat                        0
    res-25                     0
    res-77                     0
    res-309183                 0
    neoFaults                  0
    archFaults                 0
    gairFaults                 0
    aster1-AlOH-cont         327
    aster2-AlOH             1128
    aster3-FeOH-cont        1327
    aster4-Ferric-cont       327
    aster5-Ferrous-cont     1494
    aster6-Ferrous-index     327
    aster7-MgOH-comp        1494
    aster8-MgOH-cont        1327
    aster9-green               9
    aster10-kaolin          1327
    aster11-opaque          2385
    aster12-quartz             8
    aster13-regolith-b3       11
    aster14-regolith-b4       65
    aster15-silica             8
    base16                     3
    dem17                      5
    dtb18                   1648
    mag19-2vd                  6
    mag20-rtp                  6
    mag21-tmi                  6
    rad22-dose               229
    rad23-k                  238
    rad24-th                 234
    rad25-u                  229
    grav26                     7
    archean27                  3
    geol28                     3
    random                     0
    deposit                    0
    dtype: int64




```python
#Find out what's the top missing:
missingNo = df.isnull().sum(axis = 0).sort_values(ascending = False)
missingNo = missingNo[missingNo.values  > 0]
missingNo
```




    aster11-opaque          2385
    dtb18                   1648
    aster5-Ferrous-cont     1494
    aster7-MgOH-comp        1494
    aster10-kaolin          1327
    aster3-FeOH-cont        1327
    aster8-MgOH-cont        1327
    aster2-AlOH             1128
    aster1-AlOH-cont         327
    aster4-Ferric-cont       327
    aster6-Ferrous-index     327
    rad23-k                  238
    rad24-th                 234
    rad25-u                  229
    rad22-dose               229
    aster14-regolith-b4       65
    aster13-regolith-b3       11
    aster9-green               9
    aster12-quartz             8
    aster15-silica             8
    grav26                     7
    mag19-2vd                  6
    mag20-rtp                  6
    mag21-tmi                  6
    dem17                      5
    base16                     3
    archean27                  3
    geol28                     3
    dtype: int64




```python
#Plot the missingness with Seaborn
f, ax = plt.subplots(figsize = (10, 10))
sns.barplot(missingNo.values, missingNo.index, ax = ax);
```

    c:\users\nbutter\miniconda3\envs\geopy\lib\site-packages\seaborn\_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning



    
![png](02-explore_the_data_files/02-explore_the_data_11_1.png)
    



```python
# Use upsetplot to see where missing values occur
# together
# Only use the top 5 columns
missing_cols = missingNo.index[:5].tolist()
missing_counts = (df.loc[:, missing_cols]
                  .isnull()
                  .groupby(missing_cols)
                  .size())

upsetplot.plot(missing_counts);
```


    
![png](02-explore_the_data_files/02-explore_the_data_12_0.png)
    


Why is this useful to know? Can our future data analysis deal with mising data?

## Explore the data to see whether there are any unusual relationships between variables 

#### Pull out numeric and categoric variables:

1. What data types do I have in my data? Can I infer that some of them are categorical, and others are not?


```python
df.dtypes.value_counts()
```




    float64    34
    object      2
    int64       2
    dtype: int64



2. Pull out the categorical and numerical variables


```python
numericVars = df.select_dtypes(exclude = ['int64','object']).columns
catVars = df.select_dtypes(include = ['object']).columns
```

3. Plot the first 11 numerical variables, and their relationship with whether deposit information.


```python
df.shape
```




    (3138, 38)




```python
#Select which columns to plot (all of them are too many), and be sure to include the "deposit" variable
cols = [np.append(np.arange(0, 11), 37)]
data = df[df.columns[cols]]
#Make a pairwise plot to find all the relationships in the data
sns.pairplot(data,hue ="deposit",palette="Set1",diag_kind="auto")
```

    c:\users\nbutter\miniconda3\envs\geopy\lib\site-packages\pandas\core\indexes\base.py:3941: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      result = getitem(key)





    <seaborn.axisgrid.PairGrid at 0x2bea1f1da48>




    
![png](02-explore_the_data_files/02-explore_the_data_21_2.png)
    



```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lon</th>
      <th>lat</th>
      <th>res-25</th>
      <th>res-77</th>
      <th>res-309183</th>
      <th>neoFaults</th>
      <th>archFaults</th>
      <th>gairFaults</th>
      <th>aster1-AlOH-cont</th>
      <th>aster2-AlOH</th>
      <th>aster3-FeOH-cont</th>
      <th>deposit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>129.106649</td>
      <td>-26.135900</td>
      <td>1.9959</td>
      <td>1.9935</td>
      <td>2.5780</td>
      <td>0.858696</td>
      <td>0.874997</td>
      <td>2.718781</td>
      <td>1.907609</td>
      <td>NaN</td>
      <td>2.081602</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>132.781571</td>
      <td>-26.151144</td>
      <td>2.0450</td>
      <td>2.0651</td>
      <td>2.3873</td>
      <td>0.607134</td>
      <td>0.936479</td>
      <td>1.468679</td>
      <td>2.032987</td>
      <td>1.076198</td>
      <td>1.982144</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>132.816676</td>
      <td>-26.159202</td>
      <td>2.0450</td>
      <td>2.0651</td>
      <td>2.3873</td>
      <td>0.577540</td>
      <td>0.914588</td>
      <td>1.446256</td>
      <td>1.982274</td>
      <td>1.050442</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>128.945869</td>
      <td>-26.179362</td>
      <td>1.9978</td>
      <td>1.9964</td>
      <td>2.6844</td>
      <td>0.810394</td>
      <td>0.826784</td>
      <td>2.813603</td>
      <td>1.947705</td>
      <td>NaN</td>
      <td>2.035556</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>132.549807</td>
      <td>-26.185500</td>
      <td>2.0694</td>
      <td>2.0999</td>
      <td>2.3574</td>
      <td>0.652131</td>
      <td>1.026991</td>
      <td>1.499793</td>
      <td>1.977050</td>
      <td>1.064977</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3133</th>
      <td>139.507786</td>
      <td>-28.121099</td>
      <td>1.9459</td>
      <td>1.9112</td>
      <td>2.1243</td>
      <td>0.394236</td>
      <td>1.618908</td>
      <td>3.613629</td>
      <td>2.040379</td>
      <td>1.121096</td>
      <td>1.988861</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3134</th>
      <td>140.411321</td>
      <td>-31.470315</td>
      <td>2.0034</td>
      <td>2.0012</td>
      <td>1.8547</td>
      <td>0.563287</td>
      <td>0.000466</td>
      <td>2.930950</td>
      <td>2.069335</td>
      <td>1.113665</td>
      <td>1.968808</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3135</th>
      <td>139.482998</td>
      <td>-35.971690</td>
      <td>-0.2622</td>
      <td>-0.0848</td>
      <td>1.9970</td>
      <td>0.037179</td>
      <td>1.046598</td>
      <td>3.022855</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3136</th>
      <td>137.737935</td>
      <td>-33.985776</td>
      <td>2.2775</td>
      <td>2.4106</td>
      <td>2.5290</td>
      <td>0.060346</td>
      <td>0.016358</td>
      <td>0.495392</td>
      <td>1.893778</td>
      <td>NaN</td>
      <td>2.109786</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3137</th>
      <td>134.390716</td>
      <td>-30.290083</td>
      <td>2.3658</td>
      <td>2.4969</td>
      <td>1.7539</td>
      <td>1.162410</td>
      <td>0.041454</td>
      <td>0.085713</td>
      <td>2.084495</td>
      <td>1.058559</td>
      <td>1.968388</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3138 rows × 12 columns</p>
</div>



<div class="challenge">

### Challenge

What variables are the most correlated? Hint: pandas has a function to find e.g. "pearson" corrrelations.

<details>
<summary>Solution</summary>

```python
df.corr()
    
#Or pick a variable that you want to sort by. And round out the sig figs.
#df.corr().round(2).sort_values('dem17', ascending = False)
```

   
</details>
</div>

But, no need to dig through a table! We can plot the relationships.


```python
corr = df.corr() 

# Draw the heatmap with the mask and correct aspect ratio
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr,
            cmap=plt.cm.BrBG, 
            vmin=-0.5, vmax=0.5, 
            square=True,
            xticklabels=True, yticklabels=True,
            ax=ax);
```


    
![png](02-explore_the_data_files/02-explore_the_data_25_0.png)
    



```python
#Plot a regression model through the data
sns.lmplot(
    data = df,
    x = 'res-25', y = 'res-77',hue='deposit'
);
```


    
![png](02-explore_the_data_files/02-explore_the_data_26_0.png)
    


<div class="keypoints">  
### Key points
- EDA is the first step of any analysis, and often very time consuming.
- Skipping EDA can result in substantial issues with subsequent analysis.

### Questions:
- What is the first step of any ML project (and often the most time consuming)?
    
</div>




```python

```


```python

```