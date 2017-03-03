# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option("display.max_rows",30)
pd.set_option("precision",13)
pd.set_option('expand_frame_repr', False)
from datetime import datetime, date, time

### setup params
basedir = '/Users/dporter/Documents/data_local/Antarctica/OIB/'
datadir = 'IRMCR2'
timedir = '2009.10.31'
# infile = '2009_Antarctica_DC8'
infile = 'IRMCR2_20091031_01'
suffix = '.csv'
### Read ascii file as csv
df = pd.read_csv(os.path.join(basedir, datadir, timedir, infile + suffix),
                 delimiter=",", na_values='-9999.00')
#df['DATE'] = pd.DataFrame(list(df.FRAMESTR.str[:8]))
df['FRAMESTR'] = df['FRAME'].apply(str)
df['DATE'] = pd.to_datetime(list(df.FRAMESTR.str[:8]),format='%Y%m%d')
del df['FRAMESTR']
df['UNIX']=df['DATE'].astype(np.int64) // 10**9
df['UNIX']=df['UNIX']+df['TIME'].astype(np.float64)
df['iunix'] = pd.to_datetime(df['UNIX'],unit='s')
df = df.set_index('iunix')
df.index.astype(np.int64)[0:7] // 10 ** 9

# #########################################################
# GRAVITY
datadir = 'IGGRV1B'
timedir = '2009.10.31'
suffix = '.txt'
infile = 'IGGRV1B_20091031_11020500_V016'
headers = ('LAT','LONG','DATE','DOY','TIME','FLT','PSX','PSY','WGSHGT','FX','FY','FZ','EOTGRAV','FACOR','INTCOR','FAG070','FAG100','FAG140','FLTENVIRO')
grv = pd.read_csv(os.path.join(basedir, datadir, timedir, infile + suffix), delimiter=r"\s+", skiprows=70, header=None, names=headers)
### do some DATETIME operations
grv['DATETIME'] = (grv.DATE*1e5)+grv.TIME
grv['DATE'] = pd.to_datetime(grv['DATE'],format='%Y%m%d')
grv['UNIX']=grv['DATE'].astype(np.int64) // 10**9
grv['UNIX']=grv['UNIX']+grv['TIME']
grv['iunix'] = pd.to_datetime(grv['UNIX'],unit='s')
grv = grv.set_index('iunix')
grv.index.astype(np.int64)[0:7] // 10 ** 9


# #########################################################
# print "******************"
# print "Original Radar Data"
# print "******************"
# print(df.columns)
# print(df.dtypes)
#
# print(df.shape)
# print(df.head(5))
print(df[['TIME','UNIX','ELEVATION']].head(5))

# ###Subsample all to 2 Hz
rad2hz = {}
rad2hz = df.resample('500L').mean().bfill()   #mean,median,mode???
# atm2hz = atm.resample('500L').mean()
rad2hz.rename(columns={'SURFACE': 'SURFACE_radar'}, inplace=True)
# print "******************"
# print "2 Hz Radar Data"
# print "******************"
# print(rad2hz.columns)
# print(rad2hz.shape)
# print(rad2hz.head(5))
print(rad2hz[['TIME','UNIX','ELEVATION']].head(5))


### some flight diagnostics
#print "%0.1f hour flight" % ((max(df.DATETIME)-min(df.DATETIME))/3600)
print df.DATE[0]
print "%0.1f hour flight" % ((df.UNIX[-1:]-df.UNIX[0])/3600)

#max(df.DATETIME[1:len(df)-1]-df.DATETIME[0:len(df)-2])
t0 = str(df.DATE[0])
#te = df.TIME[int(len(df))-1]
#te = df.TIME[len(df.axes[0])-1]
#te = df.TIME[df.shape[0]-1]

### rolling mean
windowminutes = 15
ma = df.rolling(window=2*60*windowminutes)
r = df.resample('1min').mean()

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
x,y = m(df['LON'].values, df['LAT'].values)
# Plot them using round markers of size 6
#m.plot(x, y, 'ro', markersize=2)
m.scatter(x,y,c=df['THICK'],marker="o",cmap=cm.terrain,s=20, edgecolors='none',vmin=0,vmax=1500)#,vmin=-150,vmax=50
c = plt.colorbar(orientation='vertical', shrink = 0.5)
c.set_label("m")
#plt.show()
#plt.tight_layout()
plt.suptitle('THICK Full', y=1.02)
plt.savefig(infile+'_THICK_full_mapplot.png',bbox_inches='tight')   # save the figure to file
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
x,y = m(r['LON'].values, r['LAT'].values)
# Plot them using round markers of size 6
#m.plot(x, y, 'ro', markersize=2)
m.scatter(x,y,c=r['THICK'],marker="o",cmap=cm.terrain,s=20, edgecolors='none',vmin=0,vmax=1500)#,vmin=-150,vmax=50
c = plt.colorbar(orientation='vertical', shrink = 0.5)
c.set_label("m")
#plt.show()
#plt.tight_layout()
plt.suptitle('THICK 1min', y=1.02)
plt.savefig(infile+'_THICK_1min_mapplot.png',bbox_inches='tight')   # save the figure to file
#plt.close(m)    