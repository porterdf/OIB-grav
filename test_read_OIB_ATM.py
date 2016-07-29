# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mapplotOIB
from datetime import datetime, date, time

### setup params
if os.path.isdir('/Volumes/C/'): basedir = '/Volumes/C/data/Antarctic/OIB/ATM/2009_AN_NASA_ATM'
else: basedir = '/Volumes/BOOTCAMP/data/Antarctic/OIB/ATM/2009_AN_NASA_ATM'
infile = '2009_AN_NASA_ATM_all'
suffix = '.txt'

### Read ascii file as csv
headers = ('DATE','TIME','TIME2','LAT','LON','SURFACE','SLOPESN','SLOPEWE','RMS','NUMUSED','NUMOMIT','DISTRIGHT','TRACKID')
df = pd.read_csv(os.path.join(basedir,infile+suffix),delimiter=r"\s+",header=None)
df.rename(columns=dict(zip(df.columns,headers)), inplace=True)
del df['TIME2']

### do some DATETIME operations
df['DATETIME'] = (df.DATE*1e5)+df.TIME
df['DATE'] = pd.to_datetime(df['DATE'],format='%Y%m%d')
df['UNIX']=df['DATE'].astype(np.int64) // 10**9
df['UNIX']=df['UNIX']+df['TIME']
df['iunix'] = pd.to_datetime(df['UNIX'],unit='s')
df = df.set_index('iunix')

### some flight diagnostics
#print "%0.1f hour flight" % ((max(df.DATETIME)-min(df.DATETIME))/3600)
print df.DATE[0]
print "%0.1f hour flight" % ((df.UNIX[-1:]-df.UNIX[0])/3600)
df.groupby(df.index.day).median().head(10)

### Bin
numbins = abs(int(np.min(r.DATETIME/1e5)-int(np.max(r.DATETIME/1e5))))
numbins = abs(int(np.min(r.UNIX)-int(np.max(r.UNIX)))/86400)
bins = np.linspace(int(np.min(r.DATETIME/1e5)), int(np.max(r.DATETIME/1e5)), numbins)
r['DAY'] = np.digitize(r.DATETIME/1e5, bins) - 1


#max(df.DATETIME[1:len(df)-1]-df.DATETIME[0:len(df)-2])
t0 = str(df.DATETIME[0])
#te = df.TIME[int(len(df))-1]
#te = df.TIME[len(df.axes[0])-1]
#te = df.TIME[df.shape[0]-1]

### rolling mean
windowminutes = 15
ma = df.rolling(window=2*60*windowminutes)
r = df.resample('500L').mean()

def mapplotOIBlocal (x,y,z,title,units=[int(np.min(x)),[int(np.max(x))],range,colormap):
    import matplotlib.pyplot as plt
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
    x1,y1 = m(x.values,y.values)
    # Plot them using round markers of size 6
    #m.plot(x, y, 'ro', markersize=2)
    m.scatter(x1,y1,c=z,marker="o",cmap=colormap,s=40, edgecolors='none',vmin=range[0],vmax=range[1])#,vmin=-150,vmax=50
    c = plt.colorbar(orientation='vertical', shrink = 0.5)
    c.set_label(units)
    #plt.show()
    #plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.savefig(infile+'_'+title+'_mapplot.png',bbox_inches='tight')   # save the figure to file
    #plt.close(m)  
    return

mapplotOIBlocal(df['LON'], df['LAT'], df['SURFACE'], 'SURFACE','meters',[0,2000],cm.terrain)
mapplotOIBlocal(r['LON'], r['LAT'], r['SURFACE'], 'SURFACE_1min','meters',[0,2000],cm.terrain)
mapplotOIBlocal(r['LON'], r['LAT'], r['DATETIME']/1e5, 'DATETIME','s',[int(np.min(r.DATETIME/1e5)),int(np.max(r.DATETIME/1e5))],cm.hsv)
mapplotOIBlocal(r['LON'], r['LAT'], r['DAY'], 'Date','Date',,cm.hsv)
mapplotOIBlocal(r['LON'], r['LAT'], r['TRACKID'], 'Track ID','ID',[int(np.min(r.DAY)),int(np.max(r.DAY))],cm.hsv)

