# -*- coding: utf-8 -*-
%load_ext autoreload
%autoreload 2

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
from my_OIB_functions import importOIBgrav,importOIBatm,importOIBrad,mapplotOIBlocal
#from datetime import datetime, date, time

#########################################################
### Read in data sets
# point to data directories
basedir = '/Users/dporter/Documents/data_local/Antarctica/OIB/'
# if os.path.isdir('/Volumes/C/'):
#     basedir = '/Volumes/C/data/Antarctic/OIB/ATM/2009_AN_NASA_ATM'
# else:
#     basedir = '/Volumes/BOOTCAMP/data/Antarctic/OIB/ATM/2009_AN_NASA_ATM'

# Get date of gravity flight TODO make this user specified or for all flights
# infile = 'IGGRV1B_20091104_13100500_V016'
# subframe = 2009110401047

timedir = '2009.10.31'
subframe = 2009103102018
subframe = 2009103102021

#infile = 'IGGRV1B_20091116_15124500_V016'
#subframe = 2009111601039

#########################################################
### Run functions to read in each data set
infile = 'IGGRV1B_20091031_11020500_V016'
grv = {}
grv = importOIBgrav(basedir,timedir,infile)

infile = 'IRMCR2_20091031_01'
rad = {}
rad = importOIBrad(basedir)

infile = 'IGGRV1B_20091031_11020500_V016'
atm = {}
atm = importOIBatm(basedir)

#########################################################
###Subsample all to 2 Hz
rad2hz = rad.resample('500L').mean()    #mean,median,mode???
atm2hz = atm.resample('500L').mean()
rad2hz.rename(columns={'SURFACE': 'SURFACE_radar'}, inplace=True)

#########################################################
### append!
df = {}
df = pd.concat([grv, rad2hz[['THICK','ELEVATION','FRAME','SURFACE_radar','BOTTOM','QUALITY']], atm2hz[['SURFACE_atm']]], axis=1,join_axes=[grv.index])
df['DAY'] = df.index.day
df['HOUR'] = df.index.hour
df['ICEBASE'] = df['ELEVATION']-df['BOTTOM']
df['TOPOGRAPHY'] = df['ELEVATION']-df['SURFACE_radar']
#df.drop['TIME2']
### Some plots
#df['FAG070'].plot
#fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(16, 12))
#df['FAG070'].plot(ax=axes[0],legend=True);#axes[0,0].set_title('A');
#df['ICEBASE'].plot(ax=axes[1],legend=True);#axes[0,0].set_title('A');

#########################################################
### Make some plots
### mapplotOIBlocal(x,y,z,title,units,range,colormap)
#mapplotOIBlocal(df['LON'], df['LAT'], df['FAG070'], 'FAG070','mGal',[-80,100],cm.jet)
#mapplotOIBlocal(atm2hz['LON'], atm2hz['LAT'], atm2hz['SURFACE'], 'SURFACE_2hz','meters',[0,2000],cm.terrain)
#mapplotOIBlocal(rad2hz['LON'], rad2hz['LAT'], rad2hz['THICK'], 'THICK_2hz','meters',[100,1200],cm.jet)
#mapplotOIBlocal(grv['LON'], grv['LAT'], grv['FX'], 'FAG070_2hz_20091104','mGal',[-50,150],cm.RdBu)
#mapplotOIBlocal(df10s['LON'], df10s['LAT'], df10s['HOUR'], 'TIME','Hour',[13,24],cm.jet)
#plt.figure(); df.ix['2009-10-31 15:30:00':'2009-10-31 15:50:00'].plot(subplots=True,layout=(5,6),figsize=(11, 8));

#########################################################
### subset one glacier profile by hand
#dfsub = df['2009-10-31 15:25:00':'2009-10-31 15:50:00']
#
# Larsen C
#dfsub = df['2009-11-04 23:00:00':'2009-11-04 23:15:00']
#dfsub = df.loc[df['FRAME'] == 2009110401046]
#dfsub = df.loc[df['FRAME'].isin(['2009110401047','2009110401048'])]
#
# THE ONE BELOW WORKS
dfsub = df.query('(FRAME <= @subframe+2) & (FRAME >= @subframe-2)')

### using FX moving average
# window = 240 for plane maneuvers > 2 minutes
grv['FX_MA'] = grv['FX'].rolling(window=240, center=True).mean()
grv_sub = grv.query('(WGSHGT < 3000) & (FX_MA < 70000)')

# plots
grv['FAG070'].where((grv['WGSHGT'] < 3000) & (grv['FX_MA'] > 100000)).plot()
# overplot
plt.figure(facecolor='w')
grv['FX'].where((grv['WGSHGT'] < 3000)).plot()
grv['FX_MA'].where((grv['WGSHGT'] < 3000)).plot(color='red')
# mapplot
mapplotOIBlocal(grv['LON'], grv['LAT'], grv['FAG070'].where((grv['WGSHGT'] < 3000)), 'test','mGal',[-50,150],cm.RdBu)
mapplotOIBlocal(grv['LON'], grv['LAT'], grv['FX_MA'].where((grv['WGSHGT'] < 3000)), 'test','mGal',[-100000,100000],cm.RdBu)
mapplotOIBlocal(grv_sub['LON'], grv_sub['LAT'], grv_sub['FAG070'], 'test','mGal',[-50,150],cm.RdBu)

#########################################################
### LINE plots
#plt.figure(); dfsub.plot(subplots=True,layout=(4,8),figsize=(16, 12));
# more control
fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(12, 8))
dfsub['FAG070'].plot(ax=axes[0],legend=True,style ='b-');#axes[0,0].set_title('A');
dfsub['FAG140'].plot(ax=axes[0],legend=True,style ='r-');#axes[0,0].set_title('A');
dfsub['ELEVATION'].plot(ax=axes[1],legend=True,style ='g');#axes[0,0].set_title('A');
dfsub['ICEBASE'].plot(ax=axes[1],legend=True,style ='k');#axes[0,0].set_title('A');
dfsub['TOPOGRAPHY'].plot(ax=axes[1],legend=True,style ='b');#axes[0,0].set_title('A');
plt.suptitle(str(subframe), y=1.02)
plt.savefig('profile_'+str(subframe)+'.pdf',bbox_inches='tight')   # save the figure to file
    
#########################################################
### Mapplots
mapplotOIBlocal(dfsub['LON'], dfsub['LAT'], dfsub['FAG070'], 'FAG070_2hz_'+str(subframe),'mGal',[-50,150],cm.jet)
mapplotOIBlocal(dfsub['LON'], dfsub['LAT'], dfsub['TOPOGRAPHY'], 'SURFACE_'+str(subframe),'meters',[0,2000],cm.terrain)
mapplotOIBlocal(dfsub['LON'], dfsub['LAT'], dfsub['ICEBASE'], 'ICEBASE_'+str(subframe),'meters',[0,2000],cm.terrain)

#########################################################
### Add Line Channel
df['LINE'] = int(str(dfsub['LAT'][-3:]))

#########################################################
### Save to CSV
#dfout = df[]
dfsub.to_csv('OIB_'+str(subframe)+'.csv')
df.to_csv('OIB_'+str(subframe)[:8]+'.csv')


#df10s = df.resample('10s').mean()
#df10s.to_csv('OIB_2009-11-04_10sec.csv')
#mapplotOIBlocal(df10s['LON'], df10s['LAT'], df10s['FAG070'], 'FAG070','mGal',[-80,100],cm.jet)

    