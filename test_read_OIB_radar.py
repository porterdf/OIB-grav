# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, date, time

### setup params
if os.path.isdir('/Volumes/C/'): basedir = '/Volumes/C/data/Antarctic/OIB/RADAR/csv'
else: basedir = '/Volumes/BOOTCAMP/data/Antarctic/OIB/RADAR/csv'
infile = '2009_Antarctica_DC8'
suffix = '.csv'

### Read ascii file as csv
#headers = ('LAT','LONG','DATE','DOY','TIME','FLT','PSX','PSY','WGSHGT','FX','FY','FZ','EOTGRAV','FACOR','INTCOR','FAG070','FAG100','FAG140','FLTENVIRO')
df = pd.read_csv(os.path.join(basedir,infile+suffix),delimiter=",", na_values='-9999.00')
#df.replace('-9999.00', np.nan)

### do some DATETIME operations
#df['DATE'] = str(df['FRAME'][0])[:8]

#df['DATE'] = pd.DataFrame(list(df.FRAMESTR.str[:8]))
df['FRAMESTR'] = df['FRAME'].apply(str)
df['DATE'] = pd.to_datetime(list(df.FRAMESTR.str[:8]),format='%Y%m%d')
del df['FRAMESTR']
df['UNIX']=df['DATE'].astype(np.int64) // 10**9
df['UNIX']=df['UNIX']+df['TIME']
df['iunix'] = pd.to_datetime(df['UNIX'],unit='s')
df = df.set_index('iunix')

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