# Spatial data mining with machine learning to reveal mineral exploration targets under cover in the Gawler Craton, South Australia
*Nathaniel Butterworth and Nicholas Barnett-Moore*


Last updated 24th September 2020

This notebook covers the full workflow and produces the final figures and grids. It should be as simple as running each cell.
Read the comments throughout the code along with the paper for a full understanding. 

Contact: nathaniel.butterworth@sydney.edu.au


```python
#Import libraries for data manipulations
import pandas as pd
import numpy as np
import random
import scipy
from scipy import io

#Import libraries for plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits import mplot3d
import matplotlib.mlab as ml
from cartopy.io.img_tiles import Stamen
from numpy import linspace, meshgrid
from matplotlib.mlab import griddata
from matplotlib.path import Path
from matplotlib.patches import PathPatch

#Import libraries for tif, shapefile, and geodata manipulations
import shapefile
from shapely.geometry import Point
from shapely.geometry import shape

#Import Machine Learning libraries
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


#Import libraries for multi-threading capabilities
#from dask import delayed,compute
#from dask.distributed import Client, progress
import time
```


```python
#Define simple helper functions used in workflow

def coregPoint(point,data,region):
    '''
    Finds the nearest neighbour to a point from a bunch of other points
    point - array([longitude,latitude])
    data - array
    region - integer, same units as data
    '''
    tree = scipy.spatial.cKDTree(data)
    dists, indexes = tree.query(point,k=1,distance_upper_bound=region) 

    if indexes==len(data):
        return 'inf'
    else:
        return (indexes,dists)
    
#
def points_in_circle(circle, arr):
    '''
    A generator to return all points whose indices are within given circle.
    http://stackoverflow.com/a/2774284
    Warning: If a point is near the the edges of the raster it will not loop 
    around to the other side of the raster!
    '''
    i0,j0,r = circle

    for i in range(intceil(i0-r),intceil(i0+r)):
        ri = np.sqrt(r**2-(i-i0)**2)
        for j in range(intceil(j0-ri),intceil(j0+ri)):
            if (i >= 0 and i < len(arr[:,0])) and (j>=0 and j < len(arr[0,:])):
                yield arr[i][j]

#            
def intceil(x):
    return int(np.ceil(x))                                            

#
def coregRaster(point,data,region):
    '''
    Finds the mean value of a raster, around a point with a specified radius.
    point - array([longitude,latitude])
    data - array
    region - integer, same units as data
    '''
    i0=point[1]
    j0=point[0]
    r=region #In units of degrees
    pts_iterator = points_in_circle((i0,j0,region), data)
    pts = np.array(list(pts_iterator))
    #remove values outside the region which for there is no data (0.0).
    #print(pts)
    pts = pts[pts != 0.]
    if np.isnan(np.nanmean(pts)):
        #print(point,"nan")
        #pts=np.median(data)
        pts=-9999.
        #print("returning",pts)

    return(np.nanmean(pts))

#Make a function that can turn point arrays into a full meshgrid
def grid(x, y, z, resX=100, resY=100):
    "Convert 3 column data to matplotlib grid"
    xi = linspace(min(x), max(x), resX)
    yi = linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi,interp='linear')
    X, Y = meshgrid(xi, yi)
    return X, Y, Z

#Define a function to read the netcdf files
def readnc(filename):
    tic=time.time()
    rasterfile=filename
    data = scipy.io.netcdf_file(rasterfile,'r')
    xdata=data.variables['lon'][:]
    ydata=data.variables['lat'][:]
    zdata=np.array(data.variables['Band1'][:])

    toc=time.time()
    print(rasterfile, "in", toc-tic)
    print("spacing x", xdata[2]-xdata[1], "y", ydata[2]-ydata[1], np.shape(zdata),np.min(xdata),np.max(xdata),np.min(ydata),np.max(ydata))

    return(xdata,ydata,zdata)

#Define a function to find what polygon a point lives inside (speed imporivements can be made here)
def shapeExplore(point,shapes,recs,record):
    #'record' is the column index you want returned
    for i in range(len(shapes)):
        boundary = shapes[i]
        if Point((point.lon,point.lat)).within(shape(boundary)):
            return(recs[i][record])
    #if you have been through the loop with no result
    return(-9999.)
```

# Part 1 
### Wrangling the raw data

### Deposit locations - mine and mineral occurances
The most importantt dataset for this workflow is the currently known locations of mineral occurences. Using the data we already know about these known-deposits we will build a model to predict where future occurences will be.


```python
#Set the filename
mineshape="SA-DATA/MinesMinerals/mines_and_mineral_occurrences_all.shp"

#Set shapefile attributes and assign
sf = shapefile.Reader(mineshape)
fields = [x[0] for x in sf.fields][1:]
records = sf.records()
shps = [s.points for s in sf.shapes()]

#write into a dataframe fo easy use
df = pd.DataFrame(columns=fields, data=records)
```


```python
#See what the dataframe looks like
df
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
      <th>MINDEP_NO</th>
      <th>DEP_NAME</th>
      <th>REFERENCE</th>
      <th>COMM_CODE</th>
      <th>COMMODS</th>
      <th>COMMOD_MAJ</th>
      <th>COMM_SPECS</th>
      <th>GCHEM_ASSC</th>
      <th>DISC_YEAR</th>
      <th>CLASS_CODE</th>
      <th>...</th>
      <th>NORTHING</th>
      <th>ZONE</th>
      <th>LONGITUDE</th>
      <th>LATITUDE</th>
      <th>SVY_METHOD</th>
      <th>HORZ_ACC</th>
      <th>SRCE_MAP</th>
      <th>SRCE_CNTRE</th>
      <th>COMMENTS</th>
      <th>O_MAP_SYMB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5219</td>
      <td>MOUNT DAVIES NO.2A</td>
      <td>RB 65/80</td>
      <td>Ni</td>
      <td>Nickel</td>
      <td>Ni</td>
      <td>ELMT</td>
      <td></td>
      <td>1893.0</td>
      <td>OCCURRENCE</td>
      <td>...</td>
      <td>7112524.68</td>
      <td>52</td>
      <td>129.200549</td>
      <td>-26.106335</td>
      <td>Digitised</td>
      <td>2000.0</td>
      <td>500k meis</td>
      <td></td>
      <td></td>
      <td>T\fe1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>52</td>
      <td>ONE STONE</td>
      <td>MRR 138</td>
      <td>Ni</td>
      <td>Nickel</td>
      <td>Ni</td>
      <td>ELMT</td>
      <td>Ni-Cr</td>
      <td>1975.0</td>
      <td>OCCURRENCE</td>
      <td>...</td>
      <td>7110551.56</td>
      <td>53</td>
      <td>132.775358</td>
      <td>-26.107124</td>
      <td>Sourced from documents (PLANS, ENV, RB,etc)</td>
      <td>500.0</td>
      <td>71-385</td>
      <td></td>
      <td></td>
      <td>T\si</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8314</td>
      <td>HINCKLEY RANGE</td>
      <td></td>
      <td>Fe</td>
      <td>Iron</td>
      <td>Fe</td>
      <td>ELMT</td>
      <td></td>
      <td>1961.0</td>
      <td>OCCURRENCE</td>
      <td>...</td>
      <td>7111381.52</td>
      <td>52</td>
      <td>129.101731</td>
      <td>-26.116761</td>
      <td>Sourced from documents (PLANS, ENV, RB,etc)</td>
      <td>500.0</td>
      <td></td>
      <td></td>
      <td></td>
      <td>Mg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>69</td>
      <td>KALKA</td>
      <td>RB 91/103</td>
      <td>V, ILM</td>
      <td>Vanadium, Ilmenite</td>
      <td>V</td>
      <td>ELMT</td>
      <td>Fe-V-Ti</td>
      <td>1968.0</td>
      <td>OCCURRENCE</td>
      <td>...</td>
      <td>7110521.49</td>
      <td>52</td>
      <td>129.116042</td>
      <td>-26.124516</td>
      <td>(DISUSED) Map Plot</td>
      <td>100.0</td>
      <td>1 MILE</td>
      <td>mgt polygon on digital map</td>
      <td></td>
      <td>Mg2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>65</td>
      <td>ECHIDNA</td>
      <td>RB 91/103</td>
      <td>Ni</td>
      <td>Nickel</td>
      <td>Ni</td>
      <td>ELMT</td>
      <td>Ni</td>
      <td>1991.0</td>
      <td>OCCURRENCE</td>
      <td>...</td>
      <td>7108531.53</td>
      <td>53</td>
      <td>132.770515</td>
      <td>-26.125281</td>
      <td>(DISUSED) Map Plot</td>
      <td>20.0</td>
      <td>50K GEOL</td>
      <td>DH ECHIDNA PROSPECT</td>
      <td></td>
      <td>LMb</td>
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
      <th>8672</th>
      <td>6937</td>
      <td>YARINGA</td>
      <td>RB 43/94</td>
      <td>QTZE</td>
      <td>Quartzite</td>
      <td>QTZE</td>
      <td>ROCK</td>
      <td></td>
      <td>1956.0</td>
      <td>OCCURRENCE</td>
      <td>...</td>
      <td>6066051.52</td>
      <td>54</td>
      <td>138.254441</td>
      <td>-35.517924</td>
      <td>Google Earth image</td>
      <td>200.0</td>
      <td>50k moc</td>
      <td>fenced yard</td>
      <td></td>
      <td>Eec</td>
    </tr>
    <tr>
      <th>8673</th>
      <td>4729</td>
      <td>WELCHS</td>
      <td>MSC #19</td>
      <td>SCHT</td>
      <td>Schist</td>
      <td>SCHT</td>
      <td>ROCK</td>
      <td></td>
      <td>1930.0</td>
      <td>OCCURRENCE</td>
      <td>...</td>
      <td>6066681.48</td>
      <td>54</td>
      <td>138.648619</td>
      <td>-35.520578</td>
      <td>Digital Image</td>
      <td>20.0</td>
      <td>50k topo</td>
      <td></td>
      <td></td>
      <td>Elb</td>
    </tr>
    <tr>
      <th>8674</th>
      <td>4718</td>
      <td>ARCADIAN</td>
      <td>MSC #2</td>
      <td>CLAY</td>
      <td>Clay</td>
      <td>CLAY</td>
      <td>ROCK</td>
      <td></td>
      <td>1921.0</td>
      <td>DEPOSIT</td>
      <td>...</td>
      <td>6066561.56</td>
      <td>54</td>
      <td>138.660599</td>
      <td>-35.521892</td>
      <td>Digital Image</td>
      <td>5.0</td>
      <td>Plan 1951-0327</td>
      <td>Pit</td>
      <td></td>
      <td>Q</td>
    </tr>
    <tr>
      <th>8675</th>
      <td>1436</td>
      <td>MCDONALD</td>
      <td>MSC #7</td>
      <td>Au</td>
      <td>Gold</td>
      <td>Au</td>
      <td>ELMT</td>
      <td>Au</td>
      <td>1901.0</td>
      <td>OCCURRENCE</td>
      <td>...</td>
      <td>6065991.54</td>
      <td>54</td>
      <td>138.436645</td>
      <td>-35.522477</td>
      <td>Google Earth image</td>
      <td>200.0</td>
      <td>50k moc</td>
      <td>qz float</td>
      <td></td>
      <td>qz</td>
    </tr>
    <tr>
      <th>8676</th>
      <td>8934</td>
      <td>FAIRFIELD FARM</td>
      <td></td>
      <td>SAND</td>
      <td>Sand</td>
      <td>SAND</td>
      <td>ROCK</td>
      <td></td>
      <td>1980.0</td>
      <td>OCCURRENCE</td>
      <td>...</td>
      <td>6066871.52</td>
      <td>54</td>
      <td>138.862467</td>
      <td>-35.522846</td>
      <td>Google Earth image</td>
      <td>20.0</td>
      <td></td>
      <td>pit</td>
      <td></td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>8677 rows × 43 columns</p>
</div>




```python
#We are building a model to target the Gawler region specifically.
#Load in the Gawler target region boundary
gawlshape="SA-DATA/GCAS_Boundary/GCAS_Boundary.shp"
shapeRead = shapefile.Reader(gawlshape)
shapes  = shapeRead.shapes()

#Save the boundary xy pairs in arrays we will use throughout the workflow
xval = [x[0] for x in shapes[0].points]
yval = [x[1] for x in shapes[0].points]
```

### Set the commodity we want to target


```python
commname='Mn'
```


```python
#Pull our all the occurences of the commodity and go from there
comm=df[df['COMM_CODE'].str.contains(commname)]
comm=comm.reset_index(drop=True)
print("Shape of "+ commname, comm.shape)

#Can make simple subsets of the data here as needed
#commsig=comm[comm.SIZE_VAL!="Low Significance"]
#comm=comm[comm.SIZE_VAL!="Low Significance"]
#comm=comm[comm.COX_CLASS == "Olympic Dam Cu-U-Au"]
#comm=comm[(comm.lon<max(xval)) & (comm.lon>min(xval)) & (comm.lat>min(yval)) & (comm.lat<max(yval))]

#Can save subset to a file
#comm.to_csv("copper-deposits.csv")
```

    Shape of Mn (115, 43)


## Wrangle the geophysical and geological datasets
Each geophysical dataset could offer instight into various commodities. Here we load in the pre-processed datasets and prepare them for further manipulations, data-mining, and machine learning.

### Resistivity xyz data


```python
#Read in the data
data_res=pd.read_csv("SA-DATA/Resistivity/AusLAMP_MT_Gawler.xyzr",
                     sep='\s+',header=0,names=['lat','lon','depth','resistivity'])
data_res.head()
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
      <th>lat</th>
      <th>lon</th>
      <th>depth</th>
      <th>resistivity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-27.363931</td>
      <td>128.680796</td>
      <td>-25.0</td>
      <td>2.0007</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-27.659362</td>
      <td>128.662322</td>
      <td>-25.0</td>
      <td>1.9979</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-27.886602</td>
      <td>128.647965</td>
      <td>-25.0</td>
      <td>1.9948</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-28.061394</td>
      <td>128.636833</td>
      <td>-25.0</td>
      <td>1.9918</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-28.195844</td>
      <td>128.628217</td>
      <td>-25.0</td>
      <td>1.9885</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Simplify the dataframe with some vectors
lon_res=data_res.lon.values
lat_res=data_res.lat.values
depth_res=data_res.depth.values
res_res=data_res.resistivity.values
```


```python
#Split the data into unique depth layers
f=[]
for i in data_res.depth.unique():
    f.append(data_res[data_res.depth==i].values)

f=np.array(f)
print("Resitivity in:", np.shape(f))

#Print unique depth layers
#[print(i) for i in data_res.depth.unique()]
```

    Resitivity in: (63, 11008, 4)



```python
#Set an array we can interrogate values of later
#This is the same for all resistivity vectors
lonlatres=np.c_[f[0,:,1],f[0,:,0]]
lonres=f[0,:,1]
latres=f[0,:,0]
```

### Faults and dykes vector polylines


```python
#Get fault data neo
faultshape="SA-DATA/Neoproterozoic - Ordovician faults_shp/Neoproterozoic - Ordovician faults.shp"
shapeRead = shapefile.Reader(faultshape)
shapes  = shapeRead.shapes()
Nshp    = len(shapes)

faultsNeo=[]
for i in range(0,Nshp):
    for j in shapes[i].points:
        faultsNeo.append([j[0],j[1]])
faultsNeo=np.array(faultsNeo)
```


```python
#Get fault data archean
faultshape="SA-DATA/Archaean - Early Mesoproterozoic faults_shp/Archaean - Early Mesoproterozoic faults.shp"
shapeRead = shapefile.Reader(faultshape)
shapes  = shapeRead.shapes()
Nshp    = len(shapes)

faultsArch=[]
for i in range(0,Nshp):
    for j in shapes[i].points:
        faultsArch.append([j[0],j[1]])
faultsArch=np.array(faultsArch)
```


```python
#Get fault data dolerite dykes swarms
faultshape="SA-DATA/Gairdner Dolerite_shp/Gairdner Dolerite.shp"
shapeRead = shapefile.Reader(faultshape)
shapes  = shapeRead.shapes()
Nshp    = len(shapes)

faultsGair=[]
for i in range(0,Nshp):
    for j in shapes[i].points:
        faultsGair.append([j[0],j[1]])
faultsGair=np.array(faultsGair)
```

### Netcdf formatted raster grids


```python
#TODO: Should be cleaned up and put into dictionary or similar.
#For now, reading individual datasets is fine
x1,y1,z1 = readnc("SA-DATA/aster-AlOH-cont.nc")
x2,y2,z2 = readnc("SA-DATA/aster-AlOH-comp.nc")
x3,y3,z3 = readnc("SA-DATA/aster-FeOH-cont.nc")
x4,y4,z4 = readnc("SA-DATA/aster-Ferric-cont.nc")
x5,y5,z5 = readnc("SA-DATA/aster-Ferrous-cont.nc")
x6,y6,z6 = readnc("SA-DATA/aster-Ferrous-index.nc")
x7,y7,z7 = readnc("SA-DATA/aster-MgOH-comp.nc")
x8,y8,z8 = readnc("SA-DATA/aster-MgOH-cont.nc")
x9,y9,z9 = readnc("SA-DATA/aster-green.nc")
x10,y10,z10 = readnc("SA-DATA/aster-kaolin.nc")
x11,y11,z11 = readnc("SA-DATA/aster-opaque.nc")
x12,y12,z12 = readnc("SA-DATA/aster-quartz.nc")
x13,y13,z13 = readnc("SA-DATA/aster-regolith-b3.nc")
x14,y14,z14 = readnc("SA-DATA/aster-regolith-b4.nc")
x15,y15,z15 = readnc("SA-DATA/aster-silica.nc")
x16,y16,z16 = readnc("SA-DATA/sa-base-elev.nc")
x17,y17,z17 = readnc("SA-DATA/sa-dem.nc")
x18,y18,z18 = readnc("SA-DATA/sa-base-dtb.nc")
x19,y19,z19 = readnc("SA-DATA/sa-mag-2vd.nc")
x20,y20,z20 = readnc("SA-DATA/sa-mag-rtp.nc")
x21,y21,z21 = readnc("SA-DATA/sa-mag-tmi.nc")
x22,y22,z22 = readnc("SA-DATA/sa-rad-dose.nc")
x23,y23,z23 = readnc("SA-DATA/sa-rad-k.nc")
x24,y24,z24 = readnc("SA-DATA/sa-rad-th.nc")
x25,y25,z25 = readnc("SA-DATA/sa-rad-u.nc")
x26,y26,z26 = readnc("SA-DATA/sa-grav.nc")
```

    /workspace/SA-DATA/aster-AlOH-cont.nc in 0.023343324661254883
    spacing x 0.010000000000019327 y 0.00999999999999801 (1238, 1200) 129.004775 140.994775 -38.374825 -26.004825000000004
    /workspace/SA-DATA/aster-AlOH-comp.nc in 0.016492128372192383
    spacing x 0.010000000000019327 y 0.00999999999999801 (1238, 1200) 129.004775 140.994775 -38.374825 -26.004825000000004
    /workspace/SA-DATA/aster-FeOH-cont.nc in 0.012454509735107422
    spacing x 0.010000000000019327 y 0.00999999999999801 (1238, 1200) 129.004775 140.994775 -38.374825 -26.004825000000004
    /workspace/SA-DATA/aster-Ferric-cont.nc in 0.012809514999389648
    spacing x 0.010000000000019327 y 0.00999999999999801 (1238, 1200) 129.004775 140.994775 -38.374825 -26.004825000000004
    /workspace/SA-DATA/aster-Ferrous-cont.nc in 0.011875629425048828
    spacing x 0.010000000000019327 y 0.00999999999999801 (1238, 1200) 129.004775 140.994775 -38.374825 -26.004825000000004
    /workspace/SA-DATA/aster-Ferrous-index.nc in 0.002715587615966797
    spacing x 0.010000000000019327 y 0.00999999999999801 (1238, 1200) 129.004775 140.994775 -38.374825 -26.004825000000004
    /workspace/SA-DATA/aster-MgOH-comp.nc in 0.01164865493774414
    spacing x 0.010000000000019327 y 0.00999999999999801 (1238, 1200) 129.004775 140.994775 -38.374825 -26.004825000000004
    /workspace/SA-DATA/aster-MgOH-cont.nc in 0.012345552444458008
    spacing x 0.010000000000019327 y 0.00999999999999801 (1238, 1200) 129.004775 140.994775 -38.374825 -26.004825000000004
    /workspace/SA-DATA/aster-green.nc in 0.013841867446899414
    spacing x 0.010000000000019327 y 0.00999999999999801 (1238, 1200) 129.004775 140.994775 -38.374825 -26.004825000000004
    /workspace/SA-DATA/aster-kaolin.nc in 0.021282672882080078
    spacing x 0.010000000000019327 y 0.00999999999999801 (1238, 1200) 129.004775 140.994775 -38.374825 -26.004825000000004
    /workspace/SA-DATA/aster-opaque.nc in 0.021758079528808594
    spacing x 0.010000000000019327 y 0.00999999999999801 (1238, 1200) 129.004775 140.994775 -38.374825 -26.004825000000004
    /workspace/SA-DATA/aster-quartz.nc in 0.015913009643554688
    spacing x 0.010000000000019327 y 0.00999999999999801 (1238, 1200) 129.00452499999997 140.99452499999998 -38.37423888888889 -26.004238888888892
    /workspace/SA-DATA/aster-regolith-b3.nc in 0.025715112686157227
    spacing x 0.010000000000019327 y 0.00999999999999801 (1238, 1200) 129.004775 140.994775 -38.374825 -26.004825000000004
    /workspace/SA-DATA/aster-regolith-b4.nc in 0.0213468074798584
    spacing x 0.010000000000019327 y 0.00999999999999801 (1238, 1200) 129.004775 140.994775 -38.374825 -26.004825000000004
    /workspace/SA-DATA/aster-silica.nc in 0.0226438045501709
    spacing x 0.010000000000019327 y 0.00999999999999801 (1238, 1200) 129.00452499999997 140.99452499999998 -38.37423888888889 -26.004238888888892
    /workspace/SA-DATA/sa-base-elev.nc in 0.01804804801940918
    spacing x 0.010000000000019327 y 0.00999999999999801 (1208, 1201) 129.005 141.005 -38.065 -25.994999999999997
    /workspace/SA-DATA/sa-dem.nc in 0.012423038482666016
    spacing x 0.010000000000019327 y 0.00999999999999801 (1208, 1201) 129.005 141.005 -38.065 -25.994999999999997
    /workspace/SA-DATA/sa-base-dtb.nc in 0.0037720203399658203
    spacing x 0.010000000000019327 y 0.00999999999999801 (1208, 1201) 129.005 141.005 -38.065 -25.994999999999997
    /workspace/SA-DATA/sa-mag-2vd.nc in 0.013543128967285156
    spacing x 0.010000000000019327 y 0.00999999999999801 (1208, 1201) 129.005 141.005 -38.065 -25.994999999999997
    /workspace/SA-DATA/sa-mag-rtp.nc in 0.012284994125366211
    spacing x 0.010000000000019327 y 0.00999999999999801 (1208, 1201) 129.005 141.005 -38.065 -25.994999999999997
    /workspace/SA-DATA/sa-mag-tmi.nc in 0.022088289260864258
    spacing x 0.010000000000019327 y 0.00999999999999801 (1208, 1201) 129.005 141.005 -38.065 -25.994999999999997
    /workspace/SA-DATA/sa-rad-dose.nc in 0.01431417465209961
    spacing x 0.010000000000019327 y 0.00999999999999801 (1208, 1201) 129.005 141.005 -38.065 -25.994999999999997
    /workspace/SA-DATA/sa-rad-k.nc in 0.0068814754486083984
    spacing x 0.010000000000019327 y 0.00999999999999801 (1208, 1201) 129.005 141.005 -38.065 -25.994999999999997
    /workspace/SA-DATA/sa-rad-th.nc in 0.20752930641174316
    spacing x 0.010000000000019327 y 0.00999999999999801 (1208, 1201) 129.005 141.005 -38.065 -25.994999999999997
    /workspace/SA-DATA/sa-rad-u.nc in 0.22171664237976074
    spacing x 0.010000000000019327 y 0.00999999999999801 (1208, 1201) 129.005 141.005 -38.065 -25.994999999999997
    /workspace/SA-DATA/sa-grav.nc in 0.2518177032470703
    spacing x 0.010000000000019327 y 0.00999999999999801 (1208, 1201) 129.005 141.005 -38.065 -25.994999999999997


### Categorical Geology in vector polygons


```python
#Surface Geology, have converted unique map units to intger
geolshape=shapefile.Reader("SA-DATA/7MGeology/geology_simp.shp")

recsGeol    = geolshape.records()
shapesGeol  = geolshape.shapes()

#Archean basement geology
geolshape=shapefile.Reader("SA-DATA/Archaean_Early_Mesoprterzoic_polygons_shp/Archaean - Early Mesoproterozoic polygons.shp")

recsArch   = geolshape.records()
shapesArch  = geolshape.shapes()
```

# Part 2 - Spatial data mining of datasets

### Select the commodity and geophysical features to use 
Edit *commname* above and turn these labels on/off as required. 
Generally run data mining with all labels. 
Then can turn these features on/off before running ML if needed.


```python
lons=['lon','lat']
reslabels = [     
'res-25',
'res-77',
'res-136',
'res-201',
'res-273',
'res-353',
'res-442',
'res-541',
'res-650',  
'res-772',
'res-907',
'res-1056',
'res-1223',
'res-1407',
'res-1612',
'res-1839',
'res-2092',
'res-2372',
'res-2683',
'res-3028',
'res-3411',
'res-3837',    
'res-4309',
'res-4833',
'res-5414',
'res-6060',
'res-6776',
'res-7572',
'res-8455',
'res-9435',
'res-10523',
'res-11730',
'res-13071',
'res-14559',
'res-16210',
'res-18043',   
'res-20078',
'res-22337',
'res-24844',
'res-27627',
'res-30716',
'res-34145',
'res-37951',
'res-42175',
'res-46865',
'res-52070',
'res-57847',
'res-64261',
'res-71379',
'res-79281',
'res-88052',
'res-97788',
'res-108595',
'res-120590',
'res-133905',
'res-148685',
'res-165090',
'res-183300',
'res-203513',
'res-225950',
'res-250854',
'res-278498',
'res-309183'
]
  
faultlabels=[
    "neoFaults",
    "archFaults",
    "gairFaults"
]

numerical_features=reslabels+faultlabels+[
"aster1-AlOH-cont",
"aster2-AlOH",
"aster3-FeOH-cont",
"aster4-Ferric-cont",
"aster5-Ferrous-cont",
"aster6-Ferrous-index",
"aster7-MgOH-comp",
"aster8-MgOH-cont",
"aster9-green",
"aster10-kaolin",
"aster11-opaque",
"aster12-quartz",
"aster13-regolith-b3",
"aster14-regolith-b4",
"aster15-silica",
"base16",
"dem17",
"dtb18",
"mag19-2vd",
"mag20-rtp",
"mag21-tmi",
"rad22-dose",
"rad23-k",
"rad24-th",
"rad25-u",
"grav26"
]

categorical_features=[
'archean27',
'geol28',
'random'
]
```


```python
print("Number of geophysical layers: ", len(numerical_features))
```

    Number of geophysical layers:  92


### Generate the non-deposit dataset

This step is important. There are numerous ways to generate our non-deposit set, each with different benefits and trade-offs.
The randomisation of points throughout *some* domain appears to be robust. But you must think, is this domain a reasonable estimation of "background" geophysics/geology? Why are you picking these locations as non-deposits? Will they be over/under-representing actual deposits? Will they be over/under-representing actual non-deposits?

Change the lows, highs, and sizes as desired. And enforce the points are with some confinement area if needed.
A good place to start is within the spatial extent of the known deposits/commodity.


```python
#Generate "non-deposit" points within the same spatial domains as deposits (e.g. on land, or in the gawler, or in SA).
#We may want to train and test just over the regions that the grids are valid.
#So we can crop the known deposits to the extent of the grids.

polgonshape=shapefile.Reader("SA-DATA/SA/SA_STATE_POLYGON_shp.shp")
#polgonshape=shapefile.Reader("SA-DATA/GCAS_Boundary/GCAS_Boundary.shp)
shapesPoly  = polgonshape.shapes()

#Now make a set of "non-deposits" using a random location within our exploration area
lats_rand=np.random.uniform(low=min(df.LATITUDE), high=max(df.LATITUDE), size=len(comm.LATITUDE))
lons_rand=np.random.uniform(low=min(df.LONGITUDE), high=max(df.LONGITUDE), size=len(comm.LONGITUDE))

#And enforce the random points are within our the shapefile boudary
#Probably more efficent ways to do this for larger datasets. Fine for now.
boundary=shapesPoly[1]
for i,_ in enumerate(lats_rand):
    while not Point((lons_rand[i],lats_rand[i])).within(shape(boundary)):
            lats_rand[i]=random.uniform(min(df.LATITUDE), max(df.LATITUDE))
            lons_rand[i]=random.uniform(min(df.LONGITUDE), max(df.LONGITUDE))
            
print("Produced", len(lats_rand),len(lons_rand), "latitude-longitude pairs for non-deposits.")
```

    Produced 115 115 latitude-longitude pairs for non-deposits.



```python
#Save the SA polygon for plotting
xvalsa = [x[0] for x in shapesPoly[1].points]
yvalsa = [x[1] for x in shapesPoly[1].points]
```


```python
#Quick plot of where commodity deposit data is and generated non-deposit data
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.margins(0.05) # 5% padding to the map boundary so we can see the true extent nicely

ax.plot(comm.LONGITUDE, comm.LATITUDE, marker='o', linestyle='', color='y')
ax.plot(lons_rand,lats_rand,marker='.',linestyle='',color='k')
plt.plot(xval,yval,label='Gawler')

plt.show()
```


    
![png](01-build_the_data_files/01-build_the_data_30_0.png)
    


### Define function which performs coregistering/data-mining


```python
def coregLoop(sampleData):
    '''
    Define a function to coregister the grids and perform the spatial data mining.
    
    Requires list of lat and lon, will return the value at that point for all the hardcoded grids
    sampleData=[lat,lon]
    
    Returns array of parameters in the form:
    [lat, lon, param1, param2,...., param92,randomValue]
    
    TODO: Hadrcoded grids currently defined globally. Apply function to pass in grids for finer control.
    '''
    
    lat=sampleData[0]
    lon=sampleData[1]
    #Set the search space over which to sample the geophysical grid
    #Units are in same unit as source grids
    region=1 
    #Set the search sapce to sample the resitivity layers
    region2=100

    #Get the closest Resitivity indexes to the point
    idx,dist=coregPoint([lon,lat],lonlatres,region2)
    
    #Get the distance to the faults from the point
    _,dist=coregPoint([lon,lat],faultsNeo,region2)
    _,dist2=coregPoint([lon,lat],faultsArch,region2)
    _,dist3=coregPoint([lon,lat],faultsGair,region2)

    #Get the Numerical data indexes of the geophys at the point
    xloc1=(np.abs(np.array(x1) - lon).argmin())
    yloc1=(np.abs(np.array(y1) - lat).argmin())
    xloc2=(np.abs(np.array(x2) - lon).argmin())
    yloc2=(np.abs(np.array(y2) - lat).argmin())
    xloc3=(np.abs(np.array(x3) - lon).argmin())
    yloc3=(np.abs(np.array(y3) - lat).argmin())
    xloc4=(np.abs(np.array(x4) - lon).argmin())
    yloc4=(np.abs(np.array(y4) - lat).argmin())
    xloc5=(np.abs(np.array(x5) - lon).argmin())
    yloc5=(np.abs(np.array(y5) - lat).argmin())
    xloc6=(np.abs(np.array(x6) - lon).argmin())
    yloc6=(np.abs(np.array(y6) - lat).argmin())
    xloc7=(np.abs(np.array(x7) - lon).argmin())
    yloc7=(np.abs(np.array(y7) - lat).argmin())
    xloc8=(np.abs(np.array(x8) - lon).argmin())
    yloc8=(np.abs(np.array(y8) - lat).argmin())
    xloc9=(np.abs(np.array(x9) - lon).argmin())
    yloc9=(np.abs(np.array(y9) - lat).argmin())
    xloc10=(np.abs(np.array(x10) - lon).argmin())
    yloc10=(np.abs(np.array(y10) - lat).argmin())
    xloc11=(np.abs(np.array(x11) - lon).argmin())
    yloc11=(np.abs(np.array(y11) - lat).argmin())
    xloc12=(np.abs(np.array(x12) - lon).argmin())
    yloc12=(np.abs(np.array(y12) - lat).argmin())
    xloc13=(np.abs(np.array(x13) - lon).argmin())
    yloc13=(np.abs(np.array(y13) - lat).argmin())
    xloc14=(np.abs(np.array(x14) - lon).argmin())
    yloc14=(np.abs(np.array(y14) - lat).argmin())
    xloc15=(np.abs(np.array(x15) - lon).argmin())
    yloc15=(np.abs(np.array(y15) - lat).argmin())
    xloc16=(np.abs(np.array(x16) - lon).argmin())
    yloc16=(np.abs(np.array(y16) - lat).argmin())
    xloc17=(np.abs(np.array(x17) - lon).argmin())
    yloc17=(np.abs(np.array(y17) - lat).argmin())
    xloc18=(np.abs(np.array(x18) - lon).argmin())
    yloc18=(np.abs(np.array(y18) - lat).argmin())
    xloc19=(np.abs(np.array(x19) - lon).argmin())
    yloc19=(np.abs(np.array(y19) - lat).argmin())
    xloc20=(np.abs(np.array(x20) - lon).argmin())
    yloc20=(np.abs(np.array(y20) - lat).argmin())
    xloc21=(np.abs(np.array(x21) - lon).argmin())
    yloc21=(np.abs(np.array(y21) - lat).argmin())
    xloc22=(np.abs(np.array(x22) - lon).argmin())
    yloc22=(np.abs(np.array(y22) - lat).argmin())
    xloc23=(np.abs(np.array(x23) - lon).argmin())
    yloc23=(np.abs(np.array(y23) - lat).argmin())
    xloc24=(np.abs(np.array(x24) - lon).argmin())
    yloc24=(np.abs(np.array(y24) - lat).argmin())
    xloc25=(np.abs(np.array(x25) - lon).argmin())
    yloc25=(np.abs(np.array(y25) - lat).argmin())
    xloc26=(np.abs(np.array(x26) - lon).argmin())
    yloc26=(np.abs(np.array(y26) - lat).argmin())

    
    #Numerical data values
    z1val=coregRaster([xloc1,yloc1],z1,region)
    z2val=coregRaster([xloc2,yloc2],z2,region)
    z3val=coregRaster([xloc3,yloc3],z3,region)
    z4val=coregRaster([xloc4,yloc4],z4,region)
    z5val=coregRaster([xloc5,yloc5],z5,region)
    z6val=coregRaster([xloc6,yloc6],z6,region)
    z7val=coregRaster([xloc7,yloc7],z7,region)
    z8val=coregRaster([xloc8,yloc8],z8,region)
    z9val=coregRaster([xloc9,yloc9],z9,region)
    z10val=coregRaster([xloc10,yloc10],z10,region)
    z11val=coregRaster([xloc11,yloc11],z11,region)
    z12val=coregRaster([xloc12,yloc12],z12,region)
    z13val=coregRaster([xloc13,yloc13],z13,region)
    z14val=coregRaster([xloc14,yloc14],z14,region)
    z15val=coregRaster([xloc15,yloc15],z15,region)
    z16val=coregRaster([xloc16,yloc16],z16,region)
    z17val=coregRaster([xloc17,yloc17],z17,region)
    z18val=coregRaster([xloc18,yloc18],z18,region)
    z19val=coregRaster([xloc19,yloc19],z19,region)
    z20val=coregRaster([xloc20,yloc20],z20,region)
    z21val=coregRaster([xloc21,yloc21],z21,region)
    z22val=coregRaster([xloc22,yloc22],z22,region)
    z23val=coregRaster([xloc23,yloc23],z23,region)
    z24val=coregRaster([xloc24,yloc24],z24,region)
    z25val=coregRaster([xloc25,yloc25],z25,region)
    z26val=coregRaster([xloc26,yloc26],z26,region)
    
    #Append all the values to an array to return
    #Return dummys for categorical data for now
    vals=np.array([lon,lat])
    vals=np.append(vals,f[:,idx,3])
    vals=np.append(vals,
                   [
                    dist,dist2,dist3,
                    z1val,z2val,z3val,
                    z4val,z5val,z6val,
                    z7val,z8val,z9val,
                        z10val,z11val,z12val,
                    z13val,z14val,z15val,
                    z16val,z17val,z18val,
                    z19val,z20val,z21val,
                    z22val,z23val,z24val,
                    z25val,z26val,
                    -9999.,-9999.
                   ])
    #Append a random choice of 999 or -999 to benchmark ML
    coregMap=np.append(vals,[random.choice([-999, 999])])
    
    #Return the data
    return(coregMap)
    

```

## Run spatial mining of known deposits and "non-deposits"
Must be re-run on each commodity change. Can be saved and just loaded in if data has already been generated.


```python
#Load in co-registerd training data
training_data=pd.read_csv("ML-DATA/training_data-"+commname+".csv",header=0)

#Or if that does not exist run next two cells....
```


```python
#Interrogate the data associated with deposits

#Coregloop takes about 0.07s per call
#shapeExplore takes about 1.7s per call

tic=time.time()
deps1=[]
for row in comm.itertuples():
    lazy_result = coregLoop([row.LATITUDE,row.LONGITUDE])
    #lazy_result = delayed(coregLoop)([row.LATITUDE,row.LONGITUDE])
    deps1.append(lazy_result)
    
vec1=pd.DataFrame(np.squeeze(deps1),columns=lons+numerical_features+categorical_features)
vec1['deposit'] = 1 #Add the "depoist category flag"

toc=time.time()
print("Time deposits:", toc-tic, " seconds")
tic=time.time()

#Interrogate the data associated with randomly smapled points to use as counter-examples
deps0=[]
for lat,lon in zip(lats_rand,lons_rand):
    lazy_result = coregLoop([lat,lon])
    #lazy_result = delayed(coregLoop)([lat,lon])
    deps0.append(lazy_result)
    
vec2=pd.DataFrame(np.squeeze(deps0),columns=lons+numerical_features+categorical_features)
vec2['deposit'] = 0 #Add the "non-deposit category flag"

toc=time.time()
print("Time non-deposits:", toc-tic, " seconds")


#Combine the datasets
training_data = pd.concat([vec1, vec2], ignore_index=True)

tic=time.time()

#Add the categorical shapefile data
training_data['geol28']=training_data.apply(shapeExplore, args=(shapesGeol,recsGeol,1), axis=1)
training_data['archean27']=training_data.apply(shapeExplore, args=(shapesArch,recsArch,-1), axis=1)

toc=time.time()
print("Time geology:", toc-tic, " seconds")
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:54: RuntimeWarning: Mean of empty slice


    Time deposits: 15.536046981811523  seconds
    Time non-deposits: 15.219344139099121  seconds
    Time geology: 40.391884565353394  seconds



```python
#And save the training data out to a file
training_data.to_csv("ML-DATA/training_data-"+commname+".csv")
```


```python
#Use this to clean rows from ML if particular data does not have exisitng data and would dilute the models too mcuh
# training_data['badsum']=(training_data == -9999.).astype(int).sum(axis=1)
# (training_data == -9999).astype(int).sum(axis=1).value_counts()
# #If many of the points have no data, drop them
# indexNames = training_data[ training_data['badsum'] > 10 ].index
# training_data.drop(indexNames, inplace=True)
# #training_data.drop(columns=['badsum'], inplace=True)
# #indexNames = training_data[ training_data['17dem'] == 0 ].index
# #training_data.drop(indexNames , inplace=True)
# #Save number of deps/non-deps
# lennon=len(training_data.deposit[training_data.deposit==0])
# lendep=len(training_data.deposit[training_data.deposit==1])

# training_data

```


```python
#Save number of deps/non-deps, to be used for counting later
lennon=len(training_data.deposit[training_data.deposit==0])
lendep=len(training_data.deposit[training_data.deposit==1])
print(lennon,lendep)

#And look at the coregistered training data we have data-mined!
training_data
```

    115 115





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
      <th>res-136</th>
      <th>res-201</th>
      <th>res-273</th>
      <th>res-353</th>
      <th>res-442</th>
      <th>res-541</th>
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
      <td>139.179436</td>
      <td>-29.877637</td>
      <td>2.2135</td>
      <td>2.2916</td>
      <td>2.3263</td>
      <td>2.3432</td>
      <td>2.3547</td>
      <td>2.3644</td>
      <td>2.3725</td>
      <td>2.3779</td>
      <td>...</td>
      <td>-103.815964</td>
      <td>53.314350</td>
      <td>1.326607</td>
      <td>13.133085</td>
      <td>77.592484</td>
      <td>1.656689</td>
      <td>19214</td>
      <td>8602</td>
      <td>999.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>138.808767</td>
      <td>-30.086296</td>
      <td>2.3643</td>
      <td>2.4819</td>
      <td>2.5282</td>
      <td>2.5482</td>
      <td>2.5608</td>
      <td>2.5713</td>
      <td>2.5803</td>
      <td>2.5875</td>
      <td>...</td>
      <td>-203.493454</td>
      <td>57.424988</td>
      <td>1.505081</td>
      <td>10.301062</td>
      <td>65.972046</td>
      <td>-12.404812</td>
      <td>19218</td>
      <td>6658</td>
      <td>999.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>138.752281</td>
      <td>-30.445684</td>
      <td>2.1141</td>
      <td>2.1535</td>
      <td>2.1710</td>
      <td>2.1804</td>
      <td>2.1879</td>
      <td>2.1954</td>
      <td>2.2035</td>
      <td>2.2124</td>
      <td>...</td>
      <td>-167.319275</td>
      <td>94.660133</td>
      <td>2.568490</td>
      <td>20.143124</td>
      <td>85.115166</td>
      <td>-6.649049</td>
      <td>19218</td>
      <td>15524</td>
      <td>999.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>138.530506</td>
      <td>-30.533225</td>
      <td>2.2234</td>
      <td>2.3151</td>
      <td>2.3644</td>
      <td>2.3946</td>
      <td>2.4182</td>
      <td>2.4402</td>
      <td>2.4621</td>
      <td>2.4846</td>
      <td>...</td>
      <td>-131.731750</td>
      <td>73.580902</td>
      <td>2.218921</td>
      <td>14.252978</td>
      <td>68.107780</td>
      <td>-11.395281</td>
      <td>19218</td>
      <td>15524</td>
      <td>-999.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>138.887019</td>
      <td>-30.565479</td>
      <td>2.1982</td>
      <td>2.2647</td>
      <td>2.2899</td>
      <td>2.2987</td>
      <td>2.3022</td>
      <td>2.3038</td>
      <td>2.3039</td>
      <td>2.3022</td>
      <td>...</td>
      <td>-194.104736</td>
      <td>70.939156</td>
      <td>1.907848</td>
      <td>10.630255</td>
      <td>73.486282</td>
      <td>-1.273882</td>
      <td>19218</td>
      <td>13222</td>
      <td>-999.0</td>
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
      <th>225</th>
      <td>140.950913</td>
      <td>-36.818736</td>
      <td>2.0303</td>
      <td>2.0433</td>
      <td>2.0518</td>
      <td>2.0586</td>
      <td>2.0650</td>
      <td>2.0716</td>
      <td>2.0787</td>
      <td>2.0865</td>
      <td>...</td>
      <td>64.354500</td>
      <td>31.400856</td>
      <td>0.577390</td>
      <td>6.478987</td>
      <td>53.581688</td>
      <td>-12.427198</td>
      <td>19222</td>
      <td>5028</td>
      <td>-999.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>226</th>
      <td>139.584019</td>
      <td>-33.121195</td>
      <td>2.0452</td>
      <td>2.0671</td>
      <td>2.0715</td>
      <td>2.0635</td>
      <td>2.0485</td>
      <td>2.0289</td>
      <td>2.0052</td>
      <td>1.9775</td>
      <td>...</td>
      <td>-613.181885</td>
      <td>52.888771</td>
      <td>1.724605</td>
      <td>9.917326</td>
      <td>48.059807</td>
      <td>-11.312914</td>
      <td>19218</td>
      <td>15524</td>
      <td>-999.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>227</th>
      <td>134.221487</td>
      <td>-28.522637</td>
      <td>2.0157</td>
      <td>2.0031</td>
      <td>1.9811</td>
      <td>1.9584</td>
      <td>1.9370</td>
      <td>1.9161</td>
      <td>1.8942</td>
      <td>1.8699</td>
      <td>...</td>
      <td>123.653641</td>
      <td>55.559757</td>
      <td>0.828879</td>
      <td>15.572678</td>
      <td>62.832291</td>
      <td>-24.688011</td>
      <td>18446</td>
      <td>6658</td>
      <td>999.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>228</th>
      <td>131.427746</td>
      <td>-30.738339</td>
      <td>1.9768</td>
      <td>1.9636</td>
      <td>1.9494</td>
      <td>1.9326</td>
      <td>1.9126</td>
      <td>1.8891</td>
      <td>1.8617</td>
      <td>1.8299</td>
      <td>...</td>
      <td>114.935860</td>
      <td>34.257481</td>
      <td>0.830566</td>
      <td>6.841074</td>
      <td>50.776741</td>
      <td>-15.348145</td>
      <td>18012</td>
      <td>6684</td>
      <td>-999.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>229</th>
      <td>133.551967</td>
      <td>-31.226474</td>
      <td>2.2125</td>
      <td>2.2646</td>
      <td>2.2777</td>
      <td>2.2819</td>
      <td>2.2856</td>
      <td>2.2897</td>
      <td>2.2929</td>
      <td>2.2932</td>
      <td>...</td>
      <td>22.312160</td>
      <td>10.491249</td>
      <td>0.123158</td>
      <td>2.786360</td>
      <td>35.205452</td>
      <td>-31.615749</td>
      <td>18294</td>
      <td>16926</td>
      <td>999.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>230 rows × 98 columns</p>
</div>



## Run spatial mining of gawler target data
Only needs to be done once. Each commodity uses this same dataset for targetting. The values of the grid are used to predict whatever commodity is run. Depending on target resolution and whether using parallel versions, can take a good amount of time. 


```python
#Load in target data
target_data=pd.read_csv("ML-DATA/target_data.csv",header=0)

#OR run the next 5 cells....
```


```python
################ RUN FROM HERE ONCE (or use the HPC versions for high-res) ##########################
#Make a regularly spaced grid here for use in making a probablilty map later
lats_reg=np.linspace(min(yval),max(yval),10)
lons_reg=np.linspace(min(xval),max(xval),10)
#lats_reg=np.arange(min(yval),max(yval)+0.0100,0.0100)
#lons_reg=np.arange(min(xval),max(xval)+0.0100,0.0100)

sampleData=[]
for lat in lats_reg:
    for lon in lons_reg:
            sampleData.append([lat, lon])
            
print(np.size(sampleData))
```


```python
#Run the data-mining/coregistration
gridgawler=[]
tic=time.time()
for geophysparams in sampleData:
    #lazy_result = delayed(coregLoop)(geophysparams)
    lazy_result = coregLoop(geophysparams)
    gridgawler.append(lazy_result)
print("appended, now running...")

#c=compute(gridgawler)
toc=time.time()

print("Time taken:", toc-tic, " seconds")
```


```python
#Clean up the output file
target_data=pd.DataFrame(np.squeeze(gridgawler),columns=lons+numerical_features+categorical_features)
```


```python
#Add the categorical shapefile data
target_data['geol28']=target_data.apply(shapeExplore, args=(shapesGeol,recsGeol,1), axis=1)
target_data['archean27']=target_data.apply(shapeExplore, args=(shapesArch,recsArch,-1), axis=1)
```


```python
#Save out the data, and no need to run the co-registration again.
target_data.to_csv("target_data.csv",index=False)

################## RUN TO HERE ONCE #########################
```


```python
#Look at the target data
#Should be in the same form as the training data WITHOUT the information of whether it is a deposit or non-deposit.
target_data
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
      <th>res-136</th>
      <th>res-201</th>
      <th>res-273</th>
      <th>res-353</th>
      <th>res-442</th>
      <th>res-541</th>
      <th>...</th>
      <th>mag20-rtp</th>
      <th>mag21-tmi</th>
      <th>rad22-dose</th>
      <th>rad23-k</th>
      <th>rad24-th</th>
      <th>rad25-u</th>
      <th>grav26</th>
      <th>archean27</th>
      <th>geol28</th>
      <th>random</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>131.000009</td>
      <td>-32.664987</td>
      <td>-0.8814</td>
      <td>-0.8562</td>
      <td>-0.7231</td>
      <td>-0.5389</td>
      <td>-0.3426</td>
      <td>-0.1523</td>
      <td>0.0268</td>
      <td>0.1949</td>
      <td>...</td>
      <td>-9999.000000</td>
      <td>-9999.000000</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.000000</td>
      <td>14536</td>
      <td>1002</td>
      <td>999.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>131.010009</td>
      <td>-32.664987</td>
      <td>-0.8814</td>
      <td>-0.8562</td>
      <td>-0.7231</td>
      <td>-0.5389</td>
      <td>-0.3426</td>
      <td>-0.1523</td>
      <td>0.0268</td>
      <td>0.1949</td>
      <td>...</td>
      <td>-9999.000000</td>
      <td>-9999.000000</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.000000</td>
      <td>14536</td>
      <td>1002</td>
      <td>999.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>131.020009</td>
      <td>-32.664987</td>
      <td>-0.8814</td>
      <td>-0.8562</td>
      <td>-0.7231</td>
      <td>-0.5389</td>
      <td>-0.3426</td>
      <td>-0.1523</td>
      <td>0.0268</td>
      <td>0.1949</td>
      <td>...</td>
      <td>-9999.000000</td>
      <td>-9999.000000</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.000000</td>
      <td>14536</td>
      <td>1002</td>
      <td>-999.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>131.030009</td>
      <td>-32.664987</td>
      <td>-0.8814</td>
      <td>-0.8562</td>
      <td>-0.7231</td>
      <td>-0.5389</td>
      <td>-0.3426</td>
      <td>-0.1523</td>
      <td>0.0268</td>
      <td>0.1949</td>
      <td>...</td>
      <td>-9999.000000</td>
      <td>-9999.000000</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.000000</td>
      <td>14536</td>
      <td>1002</td>
      <td>999.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>131.040009</td>
      <td>-32.664987</td>
      <td>-0.8814</td>
      <td>-0.8562</td>
      <td>-0.7231</td>
      <td>-0.5389</td>
      <td>-0.3426</td>
      <td>-0.1523</td>
      <td>0.0268</td>
      <td>0.1949</td>
      <td>...</td>
      <td>-9999.000000</td>
      <td>-9999.000000</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.000000</td>
      <td>14536</td>
      <td>1002</td>
      <td>999.0</td>
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
      <th>374161</th>
      <td>137.970009</td>
      <td>-27.344987</td>
      <td>1.9917</td>
      <td>1.9870</td>
      <td>1.9831</td>
      <td>1.9793</td>
      <td>1.9749</td>
      <td>1.9697</td>
      <td>1.9647</td>
      <td>1.9629</td>
      <td>...</td>
      <td>60.714424</td>
      <td>77.071075</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-36.425686</td>
      <td>14548</td>
      <td>2030</td>
      <td>-999.0</td>
    </tr>
    <tr>
      <th>374162</th>
      <td>137.980009</td>
      <td>-27.344987</td>
      <td>1.9917</td>
      <td>1.9870</td>
      <td>1.9831</td>
      <td>1.9793</td>
      <td>1.9749</td>
      <td>1.9697</td>
      <td>1.9647</td>
      <td>1.9629</td>
      <td>...</td>
      <td>58.152508</td>
      <td>72.153748</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-35.383732</td>
      <td>14548</td>
      <td>2030</td>
      <td>-999.0</td>
    </tr>
    <tr>
      <th>374163</th>
      <td>137.990009</td>
      <td>-27.344987</td>
      <td>1.9924</td>
      <td>1.9882</td>
      <td>1.9849</td>
      <td>1.9819</td>
      <td>1.9785</td>
      <td>1.9746</td>
      <td>1.9707</td>
      <td>1.9681</td>
      <td>...</td>
      <td>56.024078</td>
      <td>66.960632</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-34.756775</td>
      <td>14548</td>
      <td>2030</td>
      <td>999.0</td>
    </tr>
    <tr>
      <th>374164</th>
      <td>138.000009</td>
      <td>-27.344987</td>
      <td>1.9924</td>
      <td>1.9882</td>
      <td>1.9849</td>
      <td>1.9819</td>
      <td>1.9785</td>
      <td>1.9746</td>
      <td>1.9707</td>
      <td>1.9681</td>
      <td>...</td>
      <td>53.722004</td>
      <td>61.110107</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-34.204567</td>
      <td>14548</td>
      <td>2030</td>
      <td>999.0</td>
    </tr>
    <tr>
      <th>374165</th>
      <td>138.010009</td>
      <td>-27.344987</td>
      <td>1.9924</td>
      <td>1.9882</td>
      <td>1.9849</td>
      <td>1.9819</td>
      <td>1.9785</td>
      <td>1.9746</td>
      <td>1.9707</td>
      <td>1.9681</td>
      <td>...</td>
      <td>51.256954</td>
      <td>54.969093</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-33.568863</td>
      <td>14548</td>
      <td>2030</td>
      <td>999.0</td>
    </tr>
  </tbody>
</table>
<p>374166 rows × 97 columns</p>
</div>



# Part 3 - Machine learning model

Now we have a fully data-mined, coregistered set of geophysical features. Both for training with known information about our deposits classification labels, and also a target set, we can build and apply the Machine Learning model classifier.


```python
#First, we can check some details about the data. 
#Simple check whether at least MOST of the geophysical parameters have a reasonable value associated with them.
[print(training_data.columns[i],j) for i,j in enumerate(training_data.median())]
#If any of these score -9999.0 it is recommended to remove from that column from analysis
#You can do this, by now "commenting out" the layer in cell 27.
```

    lon 138.24976824878848
    lat -31.4234139
    res-25 2.0562500000000004
    res-77 2.07275
    res-136 2.0737
    res-201 2.0729
    res-273 2.0686
    res-353 2.0707
    res-442 2.06045
    res-541 2.0584
    res-650 2.0572999999999997
    res-772 2.0549999999999997
    res-907 2.0544000000000002
    res-1056 2.0513
    res-1223 2.05735
    res-1407 2.0737
    res-1612 2.0917000000000003
    res-1839 2.0865
    res-2092 2.0808999999999997
    res-2372 2.0644
    res-2683 2.01925
    res-3028 2.0089
    res-3411 2.0092999999999996
    res-3837 1.9667
    res-4309 1.98135
    res-4833 1.9901499999999999
    res-5414 2.0079000000000002
    res-6060 2.1113999999999997
    res-6776 2.1632
    res-7572 2.252
    res-8455 2.3027499999999996
    res-9435 2.35195
    res-10523 2.4196
    res-11730 2.4535
    res-13071 2.4711499999999997
    res-14559 2.5052
    res-16210 2.4838
    res-18043 2.4543999999999997
    res-20078 2.41535
    res-22337 2.3192500000000003
    res-24844 2.3199
    res-27627 2.33305
    res-30716 2.2951
    res-34145 2.3369999999999997
    res-37951 2.372
    res-42175 2.3846499999999997
    res-46865 2.4406499999999998
    res-52070 2.4587
    res-57847 2.45555
    res-64261 2.51445
    res-71379 2.6096500000000002
    res-79281 2.6209
    res-88052 2.66555
    res-97788 2.6927000000000003
    res-108595 2.7203999999999997
    res-120590 2.68045
    res-133905 2.6466000000000003
    res-148685 2.5894500000000003
    res-165090 2.5436
    res-183300 2.4987500000000002
    res-203513 2.4674500000000004
    res-225950 2.3987
    res-250854 2.3303000000000003
    res-278498 2.2708
    res-309183 2.1997999999999998
    neoFaults 0.0900854008002272
    archFaults 0.29892370502523613
    gairFaults 1.2047528939475927
    aster1-AlOH-cont 1.9861243963241577
    aster2-AlOH 1.0791621804237366
    aster3-FeOH-cont 1.9664206504821777
    aster4-Ferric-cont 1.3717382550239563
    aster5-Ferrous-cont 0.788004457950592
    aster6-Ferrous-index 0.8526997566223145
    aster7-MgOH-comp 0.9859207570552826
    aster8-MgOH-cont 1.0112017393112183
    aster9-green 1.4036157727241516
    aster10-kaolin 0.9091927707195282
    aster11-opaque -9999.0
    aster12-quartz 0.5054315328598022
    aster13-regolith-b3 0.9444536566734314
    aster14-regolith-b4 1.2997069358825684
    aster15-silica 1.066483736038208
    base16 158.21937561035156
    dem17 218.34683990478516
    dtb18 -9999.0
    mag19-2vd -1.4414607676371816e-06
    mag20-rtp -111.75115966796875
    mag21-tmi -117.75409317016602
    rad22-dose 40.579397201538086
    rad23-k 0.9994525611400604
    rad24-th 8.156786918640137
    rad25-u 51.71901512145996
    grav26 -17.332341194152832
    archean27 19208.0
    geol28 15514.0
    random 999.0
    deposit 0.5





    [None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None]



### ML Classification
This is where the ML classifier is defined. We can substitue our favourite ML technique here, and tune model variables as desired. The default choices are recommended for the Gawler region.


```python
#Create the 'feature vector' and a 'target classification vector'
features=training_data[numerical_features+categorical_features]
targets=training_data.deposit

#Create the ML classifier with numerical and categorical data
#Scale, and replace missing values
numeric_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(missing_values=-9999., strategy='median')),
    ('scaler', StandardScaler())])

#Encode categorical data and fill missing values with default 0
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

#Combine numerical and categorical data
preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
rf = Pipeline(steps=[('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(random_state=1))])

```


```python
#You can apply weighting to the model here. 
#We find manual selction (i.e with sound geological reasoning) of deposits is probably more robust 
#than applying arbitrary weighting of class labels. 
#Nevertheless, we can do this if desired by uncommenting and tweaking  the following.

weights=np.ones(len(training_data))
weightcount=0

#Algorithm for setting weight values, point by point
# for i,row in enumerate(training_data[training_data.deposit==1].itertuples()):
#     xloc1=(np.abs(np.array(comm.LONGITUDE) - row.lon).argmin())
#     if comm.loc[xloc1].SIZE_VAL=="Locally Significant":
#         weights[i]=2
#     elif comm.loc[xloc1].SIZE_VAL=="Significant to SA":
#         weights[i]=4
#     elif comm.loc[xloc1].SIZE_VAL=="Significant to Australia":
#         weights[i]=8
#     elif comm.loc[xloc1].SIZE_VAL=="World-wide Significance":
#         weights[i]=16
#     else:
#         #Else keep the weight at 1
#         weightcount+=1
#         weights[i]=0
#         continue
        

    
```


```python
print('Tranining the Clasifier...')
rf.fit(features,targets,**{'classifier__sample_weight': weights})

print("Done RF. Now scoring...")
scores = cross_val_score(rf, features,targets, cv=10)

print("RF 10-fold cross validation Scores:", scores)
print("SCORE Mean: %.2f" % np.mean(scores), "STD: %.2f" % np.std(scores), "\n")

plt.plot(targets.values,'b-',label='Target (expected)')
plt.plot(rf.predict(features),'rx',label='Prediction')
plt.xlabel("Feature set")
plt.ylabel("Target/Prediction")
plt.legend(loc=7)
```

    Tranining the Clasifier...
    Done RF. Now scoring...
    RF 10-fold cross validation Scores: [0.95652174 0.86956522 1.         0.86956522 0.91304348 0.91304348
     0.7826087  0.86956522 0.82608696 0.69565217]
    SCORE Mean: 0.87 STD: 0.08 
    





    <matplotlib.legend.Legend at 0x7ff04942bd30>




    
![png](01-build_the_data_files/01-build_the_data_52_2.png)
    



```python
#Make a plot out the feature scores. 
#These are the important parameters that are correlated with the deposits.

ft_idx=[]
ft_lab=[]
all_idx=[]
all_lab=[]
all_dat=[]
#Just print the significant features above some threshold
for i,lab in enumerate(np.append(numerical_features,rf['preprocessor'].transformers_[1][1]['onehot'].get_feature_names(categorical_features))):
    all_dat.append([i,lab,rf.steps[1][1].feature_importances_[i]])
    all_lab.append(lab)
    all_idx.append(i)
    if rf.steps[1][1].feature_importances_[i] >1*np.median(rf.steps[1][1].feature_importances_): 
        ft_idx.append(i)
        ft_lab.append(lab)
        
```


```python
#And plot all the feature importances
#plt.plot(rf.steps[1][1].feature_importances_)

fig, ax = plt.subplots(figsize=(5,30))

ft_imps=rf.steps[1][1].feature_importances_
y_pos=np.arange(len(ft_imps))
ax.barh(y_pos,ft_imps,align='center')

ax.set_yticks(all_idx)
ax.set_yticklabels(all_lab)
ax.yaxis.label.set_color('red')
for i in ft_idx:
    ax.get_yticklabels()[i].set_color("red")

ax.set_xlabel('Feature Importance')

plt.show()

#plt.xticks([0,1,2,3,4,5,7,81,82,83,84,85,86])
```


    
![png](01-build_the_data_files/01-build_the_data_54_0.png)
    



```python
#Chec the probabilities at each of the deposit/non-deposit points
print('RF...')
pRF=np.array(rf.predict_proba(features))
print("Done RF")

plt.plot(pRF[:,1])
```

    RF...
    Done RF





    [<matplotlib.lines.Line2D at 0x7ff04905b0b8>]




    
![png](01-build_the_data_files/01-build_the_data_55_2.png)
    


## Finally, apply the model to the grid


```python
#Apply the trained ML to our gridded data to determine the probabilities at each of the points
print('RF...')
pRF_map=np.array(rf.predict_proba(target_data[numerical_features+categorical_features]))
print("Done RF")
```

    RF...
    Done RF



```python
#Create a meshgrid from our xyz list of points
gridX,gridY,gridZ=grid(target_data.lon, target_data.lat,pRF_map[:,1])
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:67: MatplotlibDeprecationWarning: The griddata function was deprecated in Matplotlib 2.2 and will be removed in 3.1. Use scipy.interpolate.griddata instead.



```python
#Save the csv grid of targets
targetCu = {'Longitude': target_data.lon, 'Latitude': target_data.lat, 'Prediction': pRF_map[:,1]}
targetCu=pd.DataFrame(targetCu)
targetCu.to_csv('Targets-'+commname+'.csv',header=0,index=False)
```


```python
#Plot the final target map
fig = plt.figure(figsize=(10,10),dpi=300)

#Make a map projection to plot on.
ax = plt.axes(projection=ccrs.LambertAzimuthalEqualArea(central_longitude=135.0, central_latitude=-31.0))
       
#Set the extent of interest
img_extent = [min(df.LONGITUDE)+1.5,  max(df.LONGITUDE)-3.0, min(df.LATITUDE)+5,max(df.LATITUDE)-1]
ax.set_extent(img_extent)

#Put down a base map
ax.coastlines(resolution='10m', color='gray',)
tiler = Stamen('terrain-background')
mercator = tiler.crs
ax.add_image(tiler, 6)

#Make the gridlines
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.3, color='gray', alpha=0.5, linestyle='-')
gl.top_labels = False
gl.bottom_labels = True
gl.right_labels = False
gl.left_labels = True
#gl.xlines = False
gl.xlocator = mticker.FixedLocator(list(np.linspace(np.floor(min(df.LONGITUDE))+1,np.ceil(max(df.LONGITUDE))-1,num=5)))
gl.ylocator = mticker.FixedLocator(list(np.linspace(np.floor(min(df.LATITUDE))+1,np.ceil(max(df.LATITUDE))-1,num=5)))
gl.xlocator = mticker.FixedLocator([141,138,135,132,129])
gl.ylocator = mticker.FixedLocator([-38,-34,-31,-28,-26])
#gl.ylocator = mticker.FixedLocator(list(np.linspace(-28,-35,num=3)))
gl.xlabel_style = {'size': 10, 'color': 'gray'}
gl.ylabel_style = {'size': 10, 'color': 'gray'}
#gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

#Create a patch of the gawler region where the data is
path=Path(list(zip(xval, yval)))
patch = PathPatch(path, facecolor='none',transform = ccrs.PlateCarree(),linestyle='--',linewidth=0.5)
plt.gca().add_patch(patch)

#Plot the main map
im=ax.contourf(gridX,gridY,gridZ,cmap=plt.cm.coolwarm,transform = ccrs.PlateCarree(),vmin=0,vmax=1)
#im = ax.imshow(gridZ, interpolation='bicubic', cmap=plt.cm.bwr,
#                origin='lower', extent=[np.min(gridX),np.max(gridX),np.min(gridY),np.max(gridY)],
#                clip_path=patch, clip_on=True,zorder=1,transform = ccrs.PlateCarree())
for c in im.collections:
    c.set_clip_path(patch)
    
# l5=ax.scatter(commall.LONGITUDE, commall.LATITUDE, 
#               edgecolor='k',s=10,marker='d', linewidths=0.5,label="",
#               c='r',cmap=plt.cm.bwr,vmin=0,vmax=1,zorder=2,transform = ccrs.PlateCarree())

#Add the deposits coloured by their classification score
l4=ax.scatter(training_data.lon[training_data.deposit==0], training_data.lat[training_data.deposit==0],
               edgecolor='k',s=20,marker='s', linewidths=1,label="",
               c=pRF[lendep:,1],cmap=plt.cm.bwr,vmin=0,vmax=1,zorder=3,transform = ccrs.PlateCarree())

l3=ax.scatter(training_data.lon[training_data.deposit==1], training_data.lat[training_data.deposit==1], 
              edgecolor='k',s=20,marker='o', linewidths=1,label="",
              c=pRF[:lendep,1],cmap=plt.cm.bwr,vmin=0,vmax=1,zorder=2,transform = ccrs.PlateCarree())

#Plot the outline of the Gawler region
ax.plot(xval,yval,'k--',label='Gawler Target Region',linewidth=0.2)
ax.plot(0,0,'r.',label='Known '+commname+' deposits for training',zorder=3,transform = ccrs.PlateCarree())
ax.plot(0,0,'b.',label='Non-Deposits for training',zorder=3,transform = ccrs.PlateCarree())
#ax.plot(0,0,'rd',label='All other Au deposits (not used for training)',zorder=3,transform = ccrs.PlateCarree())

# ax.plot(xlons,xlats,'y-',label='Central Gawler Au Province',zorder=3,transform = ccrs.PlateCarree())
# ax.plot(xlons2,xlats2,'g-',label='Olympic IOCG Province',zorder=3,transform = ccrs.PlateCarree())

# ax.plot(xval,yval,'k--',label='Gawler Target Region',linewidth=0.5,zorder=2,transform = ccrs.PlateCarree())

# Add a map title, legend, colorbar
#plt.title('Known deposits and predictive map for Gawler region, SA')
ax.legend(loc=2,fontsize=12)
#plt.xlabel('Longitude')
#plt.ylabel('Latitude')

#Make a Colorbar
# cbaxes = fig.add_axes([0.16, 0.27, 0.25, 0.015])
# cbar = plt.colorbar(l3, cax = cbaxes,orientation="horizontal")
# cbar.set_label(commname+' prediction')

cbaxes = fig.add_axes([0.20, 0.22, 0.1, 0.015])
cbar = plt.colorbar(im, cax = cbaxes,orientation="horizontal", ticks=[0.0,0.5,1])
# # #cbar.ax.set_xticklabels(['Medium','High'],fontsize=8)
cbar.set_label(commname+' Prediction Score', labelpad=10,fontsize=12)
cbar.ax.xaxis.set_label_position('top')

plt.show()
```

    <urlopen error [Errno -5] No address associated with hostname>
    <urlopen error [Errno -5] No address associated with hostname><urlopen error [Errno -5] No address associated with hostname>
    <urlopen error [Errno -5] No address associated with hostname>
    <urlopen error [Errno -5] No address associated with hostname>
    <urlopen error [Errno -5] No address associated with hostname>
    



    
![png](01-build_the_data_files/01-build_the_data_60_1.png)
    



```python

```