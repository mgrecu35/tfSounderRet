import os
import datetime
import h5py as h5
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.colors as col
from mpl_toolkits.basemap import Basemap
from  netCDF4 import Dataset
ds=datetime.datetime(2014,7,4)
hist2d=np.zeros((50,50),float)
hist2dc=np.zeros((50,50),float)
diffHist=np.zeros((50,176,8),float)


def readGPROF(fnamein):
    f=h5.File(fnamein,'r')
    lat=f['S1/Latitude'][:,:]
    lon=f['S1/Longitude'][:,:]
    stype=f['S1/surfaceTypeIndex'][:,:]
    sfPrecip=f['S1/surfacePrecipitation'][:,:]
    f.close()
    return stype,sfPrecip,lat,lon

