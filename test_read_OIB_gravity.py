# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, date, time

### setup params
# if os.path.isdir('/Volumes/C/'): basedir = '/Volumes/C/data/Antarctic/OIB/GRAVITY/'
# else: basedir = '/Volumes/BOOTCAMP/data/Antarctic/OIB/GRAVITY/'

basedir = '/Users/dporter/Documents/data_local/Antarctica/OIB/'
datadir = 'IGGRV1B'
timedir = '2009.10.31'
suffix = '.txt'
infile = 'IGGRV1B_20091031_11020500_V016'

### Read ascii file as csv
headers = ('LAT','LONG','DATE','DOY','TIME','FLT','PSX','PSY','WGSHGT','FX','FY','FZ','EOTGRAV','FACOR','INTCOR','FAG070','FAG100','FAG140','FLTENVIRO')
# df = pd.read_csv(os.path.join(basedir, datadir, timedir, infile + suffix),
#                  delimiter=r"\s+", skiprows=69)
df = pd.read_csv(os.path.join(basedir, datadir, timedir, infile + suffix), delimiter=r"\s+", skiprows=70, header=None, names=headers)
# df = pd.read_csv(os.path.join(basedir, datadir, timedir, infile + suffix),
#                  delimiter=",", na_values='-9999.00')
subframe = 2009103102021


### Read ascii file as csv
#metadata ends on line 69, column names on line 70
# #headers = ('LAT','LONG','DATE','DOY','TIME','FLT','PSX','PSY','WGSHGT','FX','FY','FZ','EOTGRAV','FACOR','INTCOR','FAG070','FAG100','FAG140','FLTENVIRO')
# df = pd.read_csv(os.path.join(basedir,infile+suffix),delimiter=r"\s+",skiprows=69)
# #headers = [(df.columns[1:df.shape[1]]),'temp']
# headers = df.columns[1:df.shape[1]]
# print df.columns
# print headers
# df.rename(columns=dict(zip(df.columns,headers)), inplace=True)
# #df['ENVIRO'] = df.columns[19]
# print df.columns
# df.drop(df.columns[19],axis=1,inplace=True,errors='ignore')
# print df.columns

### do some DATETIME operations
df['DATETIME'] = (df.DATE*1e5)+df.TIME
df['DATE'] = pd.to_datetime(df['DATE'],format='%Y%m%d')
df['UNIX']=df['DATE'].astype(np.int64) // 10**9
df['UNIX']=df['UNIX']+df['TIME']
df['index'] = pd.to_datetime(df['UNIX'],unit='s')
df = df.set_index('index')

### some flight diagnostics
#print "%0.1f hour flight" % ((max(df.DATETIME)-min(df.DATETIME))/3600)
print df.DATE[0]
print "%0.1f hour flight" % ((df.UNIX[-1:]-df.UNIX[0])/3600)

#max(df.DATETIME[1:len(df)-1]-df.DATETIME[0:len(df)-2])
t0 = str(df.DATETIME[0])
#te = df.TIME[int(len(df))-1]
#te = df.TIME[len(df.axes[0])-1]
#te = df.TIME[df.shape[0]-1]

### rolling mean
windowminutes = 15
ma = df.rolling(window=2*60*windowminutes)
r = df.resample('1min').mean()

#########################################
# Create a figure of size (i.e. pretty big)
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
fig = plt.figure(figsize=(8,8))
#create basemap
m = Basemap(width=700000,height=1100000,
            resolution='h',projection='stere',area_thresh = 100000.0,
            lat_ts=-55,lat_0=-68,lon_0=-65.)
# Draw the coastlines on the map
m.drawcoastlines()
# Draw country borders on the map
m.drawcountries()
# Fill the land with grey
#m.fillcontinents(color = 'gainsboro')
#m.drawmapboundary(fill_color='steelblue')
#m.bluemarble()
#m.etopo()
m.shadedrelief()
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,5.),labels=[1,0,0,0])
m.drawmeridians(np.arange(-180.,181.,5.),labels=[0,0,0,1])
# Define our longitude and latitude points
# We have to use .values because of a wierd bug when passing pandas data
# to basemap.
x,y = m(r['LONG'].values, r['LAT'].values)
# Plot them using round markers of size 6
#m.plot(x, y, 'ro', markersize=2)
m.scatter(x,y,c=r['FX'],marker="o",cmap=cm.RdYlBu,s=40, edgecolors='none')#,vmin=-1000,vmax=1000
c = plt.colorbar(orientation='vertical', shrink = 0.5)
c.set_label("mGal")
#plt.show()
#plt.tight_layout()
plt.suptitle('FX 1 min', y=1.02)
plt.savefig(infile+'_FX_1m_mapplot.png',bbox_inches='tight')   # save the figure to file
#plt.close(m)  

#########################################
# Create a figure of size (i.e. pretty big)
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
fig = plt.figure(figsize=(8,8))
#create basemap
m = Basemap(width=700000,height=1100000,
            resolution='h',projection='stere',area_thresh = 100000.0,
            lat_ts=-55,lat_0=-68,lon_0=-65.)
# Draw the coastlines on the map
m.drawcoastlines()
# Draw country borders on the map
m.drawcountries()
# Fill the land with grey
#m.fillcontinents(color = 'gainsboro')
#m.drawmapboundary(fill_color='steelblue')
#m.bluemarble()
#m.etopo()
m.shadedrelief()
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,5.),labels=[1,0,0,0])
m.drawmeridians(np.arange(-180.,181.,5.),labels=[0,0,0,1])
# Define our longitude and latitude points
# We have to use .values because of a wierd bug when passing pandas data
# to basemap.
x,y = m(r['LONG'].values, r['LAT'].values)
# Plot them using round markers of size 6
#m.plot(x, y, 'ro', markersize=2)
m.scatter(x,y,c=r['FAG070'],marker="o",cmap=cm.jet,s=40, edgecolors='none',vmin=-100,vmax=200)#,vmin=-100,vmax=100
c = plt.colorbar(orientation='vertical', shrink = 0.5)
c.set_label("mGal")
#plt.show()
#plt.tight_layout()
plt.suptitle('FAG070 1 min', y=1.02)
plt.savefig(infile+'_FAG070_1m_mapplot.png',bbox_inches='tight')   # save the figure to file
#plt.close(m)  

#########################################
### FULL FAG070
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
fig = plt.figure(figsize=(8,8))
#create basemap
m = Basemap(width=700000,height=1100000,
            resolution='h',projection='stere',area_thresh = 100000.0,
            lat_ts=-55,lat_0=-68,lon_0=-65.)
# Draw the coastlines on the map
m.drawcoastlines()
# Draw country borders on the map
m.drawcountries()
# Fill the land with grey
#m.fillcontinents(color = 'gainsboro')
#m.drawmapboundary(fill_color='steelblue')
#m.bluemarble()
#m.etopo()
m.shadedrelief()
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,5.),labels=[1,0,0,0])
m.drawmeridians(np.arange(-180.,181.,5.),labels=[0,0,0,1])
# Define our longitude and latitude points
# We have to use .values because of a wierd bug when passing pandas data
# to basemap.
x,y = m(df['LONG'].values, df['LAT'].values)
# Plot them using round markers of size 6
#m.plot(x, y, 'ro', markersize=2)
m.scatter(x,y,c=df['FAG070'],marker="o",cmap=cm.jet,s=40, edgecolors='none',vmin=-100,vmax=200)#,vmin=-150,vmax=50
c = plt.colorbar(orientation='vertical', shrink = 0.5)
c.set_label("mGal")
#plt.show()
#plt.tight_layout()
plt.suptitle('FAG070 2 Hz', y=1.02)
plt.savefig(infile+'_FAG070_2Hz_mapplot.png',bbox_inches='tight')   # save the figure to file
#plt.close(m)  
