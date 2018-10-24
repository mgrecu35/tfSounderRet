from os import walk
from readGPROFSub import *
mypath='/itedata/ITE057/2015/09'
mypath='/PANFS/prod/data/product/regularGPM/'
f1 = []
f2 = []
f3 = []
f4 = []
import glob
import datetime
s=datetime.datetime(2016,6,1)
import h5py as h5
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.colors as col
from mpl_toolkits.basemap import Basemap
from keras.models import load_model
model=load_model("keras_NPP_ATMS_pred_Ocean.h5")
model_Land=load_model("keras_NPP_ATMS_pred_Land.h5")

from  netCDF4 import Dataset
ifig=0
from pyresample import bilinear, geometry
import pickle

s=datetime.datetime(2016,6,1)
mypath='/gpmdata/'
dbL1=[]
from pyresample import kd_tree
from numpy import *
dbL1=[]

fName='/gpmdata/2014/06/10/gprof/2A-CLIM.NPP.ATMS.GPROF2017v2.20140610-S043048-E061216.013560.V05C.HDF5'

fName='/gpmdata/2014/06/10/gprof/2A.NPP.ATMS.GPROF2017v2.20140610-S162117-E180246.013567.V05C.HDF5'


orb=fName.split('.')[-3]
wCName='/gpmdata/2014/06/10/1C/1C.NPP*'+orb+'*HDF5'
fName1C=glob.glob(wCName)[0]

fgmi=Dataset(fName1C,'r')
 
stype,sfPrecip,lat,lon=readGPROF(fName)
ns,nr=lat.shape
a=np.nonzero((lon[:,nr/2]-110)*(lon[:,nr/2]-140)<0)
b=np.nonzero((lat[:,nr/2][a]-10)*(lat[:,nr/2][a]-55)<0)
#a=[arange(600,900)]
#b=[arange(300)]

cc=np.nonzero(sfPrecip[a][b]>-1)
cc1=np.nonzero(sfPrecip[a][b]>0)
dd=np.nonzero(stype[a][b][cc1]==1)
print sfPrecip[a][b][cc].sum()
m =\
    Basemap(llcrnrlon=110,llcrnrlat=20,urcrnrlon=140,\
            urcrnrlat=50,projection='mill', resolution='i')

target_def = geometry.SwathDefinition(lons=lon[a[0][b],9:-11], \
                                      lats=lat[a[0][b],9:-11])

tc2=fgmi['S2']['Tc'][a[0][b],:,:]
lat2=fgmi['S2']['Latitude'][a[0][b],:]
lon2=fgmi['S2']['Longitude'][a[0][b],:]
source_def = geometry.SwathDefinition(lons=lon2, lats=lat2)
wf = lambda r: 1 - r/100000.0
tc11 = [kd_tree.resample_gauss(source_def,tc2[:,:,0], \
                               target_def,\
                               radius_of_influence=\
                               15e3, \
                               sigmas=10e3)]

tc3=fgmi['S3']['Tc'][a[0][b],:,:]
lat3=fgmi['S3']['Latitude'][a[0][b],:]
lon3=fgmi['S3']['Longitude'][a[0][b],:]
source_def = geometry.SwathDefinition(lons=lon3, lats=lat3)
wf = lambda r: 1 - r/100000.0
tc31 = [kd_tree.resample_gauss(source_def,tc3[:,:,0], \
                               target_def,\
                               radius_of_influence=\
                               15e3, \
                               sigmas=10e3)]

tc22=fgmi['S1']['Tc'][a[0][b],:,:]
lat22=fgmi['S1']['Latitude'][a[0][b],:]
lon22=fgmi['S1']['Longitude'][a[0][b],:]
source_def = geometry.SwathDefinition(lons=lon22, lats=lat22)
wf = lambda r: 1 - r/100000.0
tc221 = [kd_tree.resample_gauss(source_def,tc22[:,:,0], \
                               target_def,\
                               radius_of_influence=\
                               15e3, \
                               sigmas=10e3)]


tc4=fgmi['S4']['Tc'][a[0][b],:,:]
lat4=fgmi['S4']['Latitude'][a[0][b],:]
lon4=fgmi['S4']['Longitude'][a[0][b],:]
source_def = geometry.SwathDefinition(lons=lon4, lats=lat4)
tc41 = [kd_tree.resample_gauss(source_def,tc4[:,:,k], \
                               target_def,\
                               radius_of_influence=\
                               15e3, \
                               sigmas=10e3) for k in range(6)]
tc41=array(tc41)
tc11=array(tc11)
r1L=[]
r2L=[]
sfcPrecip2=sfPrecip*0
tc31=array(tc31)
tc221=array(tc221)
for i in a[0][b]:
    for j in range(9,85):
        if stype[i,j]>=1 and sfPrecip[i,j]>=-0.03 and \
           i-a[0][b][0] < tc11.shape[1]-1:
            i0=i-a[0][b][0]
            if tc11[:,i0,j-9].min()>0 and tc41[:,i0,j-9].min()>0\
               and \
               sfPrecip[i,j]>-0.05:
                tcL=[tc11[0,i0,j-9]]
                tcL.append(tc31[0,i0,j-9])
                tcL.extend(tc41[:,i0,j-9])
                tcL.append(j)
                tcL.append(stype[i,j])
                tcL.append(sfPrecip[i,j])
                xp=array([array(tcL[:-1])])
                if stype[i,j]==1:
                    sfRainE=model.predict(xp)[0][0]
                    if sfRainE<-0.0504:
                        sfRainE=0.
                else:
                    sfRainE=model_Land.predict(xp)[0][0]
                    if sfRainE<-0.0504:
                        sfRainE=0.
                sfcPrecip2[i,j]=sfRainE
ymin=0.0504
for i1 in a[0][b][1:-1]:
    for j in range(1,95):
        if sfcPrecip2[i1,j]<ymin and sfcPrecip2[i1-1:i1+2,j-1:j+2].max()<ymin:
            sfcPrecip2[i1,j]=0.
        if stype[i1,j]==13:
            if sfcPrecip2[i1,j]<ymin and sfcPrecip2[i1-1:i1+2,j-1:j+2].max()<0.75:
                sfcPrecip2[i1,j]=0.
        if stype[i1,j]==14:
            if sfcPrecip2[i1,j]<ymin and sfcPrecip2[i1-1:i1+2,j-1:j+2].max()<0.75:
                sfcPrecip2[i1,j]=0.

t_bias=1.2                
sfcPrecip2*=t_bias
lons=lon[a[0][b],9:-11]
lats=lat[a[0][b],9:-11]
ny=lons.shape[1]
x,y=m(lons,lats)
m.drawcoastlines()
m.drawparallels(10+np.arange(8)*5,labels=[True,False,False,False])
m.drawmeridians(110+np.arange(7)*5,labels=[True,True,False,True])
plt.pcolormesh(x[:,:],y[:,:],sfcPrecip2[a[0][b],9:-11],norm=col.LogNorm(),vmax=30.,cmap='jet')
plt.title('ATMS Retrieval, 10 June 2014') 
cbar=plt.colorbar()
cbar.ax.set_title('mm/h')
plt.savefig('sfcPrecipATMS_2014_06_10.13567.png')
plt.figure()
m.drawcoastlines()
m.drawparallels(10+np.arange(8)*5,labels=[True,False,False,False])
m.drawmeridians(110+np.arange(7)*5,labels=[True,True,False,True])
for i in range(4):
    plt.subplot(2,2,i+1)
    m.drawcoastlines()
    plt.pcolormesh(x[:,:],y[:,:],tc41[i,:,:],cmap='jet')


plt.figure()
m.drawcoastlines()
for i in range(1):
    plt.pcolormesh(x[:,:],y[:,:],tc31[i,:,:],cmap='jet')

plt.figure()
m.drawcoastlines()
for i in range(1):
    plt.pcolormesh(x[:,:],y[:,:],tc221[i,:,:],cmap='jet')
