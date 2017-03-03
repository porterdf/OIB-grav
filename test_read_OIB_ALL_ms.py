# -*- coding: utf-8 -*-
# %load_ext autoreload
# %autoreload 2

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
from my_OIB_functions import *
#from datetime import datetime, date, time

pd.set_option("display.max_rows",20)
# pd.set_option("precision",13)
pd.set_option('expand_frame_repr', False)

#########################################################
### Read in data sets
#########################################################
basedir = '/Users/dporter/Documents/data_local/Antarctica/OIB/'
# if os.path.isdir('/Volumes/C/'):
#     basedir = '/Volumes/C/data/Antarctic/OIB/ATM/2009_AN_NASA_ATM'
# else:
#     basedir = '/Volumes/BOOTCAMP/data/Antarctic/OIB/ATM/2009_AN_NASA_ATM'

### Get date of gravity flight TODO make this user specified or for all flights
timedir = '2009.10.31'
#subframe = 2009103102018
# subframe = 2009103102021
#subframe = 2009111601039
# subframe = 2009110401047

#########################################################
### Run functions to read in each data set
#########################################################
infile = 'IGGRV1B_20091031_11020500_V016'
grv = {}
grv = importOIBgrav(basedir,timedir,infile)
infile = 'IRMCR2_20091031_01'
rad = {}
rad = importOIBrad(basedir,timedir,infile)
infile = 'IGGRV1B_20091031_11020500_V016'
# catATM(basedir,timedir)
atm = {}
atm = importOIBatm(basedir,timedir)

#########################################################
### Subsample all to 2 Hz
#########################################################
rad2hz = {}
rad2hz = rad.resample('500L').first().bfill()   #mean,median,mode???
# atm2hz = atm.resample('500L').mean()
# print "******************"
# print "Original Radar Data"
# print "******************"
# print(rad[['TIME','UNIX','BOTTOM']].head(15))
#  print rad.DATE.dt.strftime('%Y-%m-%d %H:%M:%S.%f').head(5)
# print "******************"
# print "2 Hz Radar Data"
# print "******************"
# print(rad2hz[['TIME','UNIX','BOTTOM']].head(7))
#  print rad2hz.DATE.dt.strftime('%Y-%m-%d %H:%M:%S.%f').head(5)

#  grv2hz = grv.resample('500L').first().bfill()   #mean,median,mode???
# print "******************"
# print "Original Gravity Data"
# print "******************"
# print(grv[['TIME','UNIX','FAG070']].head(15))
# print rad.DATE.dt.strftime('%Y-%m-%d %H:%M:%S.%f').head(5)
#  print "******************"
#  print "2 Hz Gravity Data"
#  print "******************"
#  print(grv2hz[['TIME','UNIX','FAG070']].head(7))
# print rad2hz.DATE.dt.strftime('%Y-%m-%d %H:%M:%S.%f').head(5)
atm2hz = {}
atm2hz = atm.resample('500L').first().bfill()   #mean,median,mode???

#########################################################
### concat?
#########################################################
df = {}
# df = pd.concat([grv, rad2hz[['THICK','ELEVATION','FRAME','SURFACE_radar','BOTTOM','QUALITY']]], axis=1,join_axes=[grv.index])
df = pd.concat([grv, rad2hz[['THICK','ELEVATION','FRAME','SURFACE_radar','BOTTOM','QUALITY']], atm2hz[['SURFACE_atm','NUMUSED']]], axis=1,join_axes=[grv.index])
df['DAY'] = df.index.day
df['HOUR'] = df.index.hour
df['ICEBASE'] = df['ELEVATION']-df['BOTTOM']
df['TOPOGRAPHY_radar'] = df['ELEVATION']-df['SURFACE_radar']

### Some plots
#df['FAG070'].plot
#fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(16, 12))
#df['FAG070'].plot(ax=axes[0],legend=True);#axes[0,0].set_title('A');
#df['ICEBASE'].plot(ax=axes[1],legend=True);#axes[0,0].set_title('A');
# plt.figure(); df.ix['2009-10-31 15:30:00':'2009-10-31 15:50:00'].plot(subplots=True,layout=(5,6),figsize=(11, 8));

#########################################################
### subset one glacier profile by hand
#########################################################
df_sub = df['2009-10-31 14:10:00':'2009-10-31 14:50:00']
print(df_sub[['TIME','UNIX','FAG070','ICEBASE','TOPOGRAPHY_radar']].head(15))
#
# Larsen C
#df_sub = df['2009-11-04 23:00:00':'2009-11-04 23:15:00']
#df_sub = df.loc[df['FRAME'] == 2009110401046]
#df_sub = df.loc[df['FRAME'].isin(['2009110401047','2009110401048'])]
#
# THE ONE BELOW WORKS
# df_sub = df.query('(FRAME <= @subframe+2) & (FRAME >= @subframe-2)')

### using FX moving average
# # window = 240 for plane maneuvers > 2 minutes
# grv['FX_MA'] = grv['FX'].rolling(window=240, center=True).mean()
# grv_sub = grv.query('(WGSHGT < 3000) & (FX_MA < 70000)')

#########################################################
### Add Line Channel
#########################################################
# df_sub['LINE'] = int(str(df_sub['LAT'][0]))[-3:]
df_sub.loc[:,'LINE'] = str(df_sub['FLT'][0])+'.'+str(abs(df_sub['LAT'][0] * 1e3))[:4]

#########################################################
### Split Dataframe using Gravity quality/presence
#########################################################
### create new logistic column for ANY gravity present

### now split on breaks in gravity breaks
dflst = [g for _, g in df_sub.groupby((df.FLTENVIRO.diff() != 0).cumsum())]
# dflst[0]
# [i.index for i in dflst]
# plt.figure(); dflst[1].plot(subplots=True,layout=(4,8),figsize=(16, 12));
### Add line channel
submode = 0
for dnum,dname in enumerate(dflst,start=0):
    dflst[dnum].loc[:, 'LINE'] = str(dflst[dnum]['FLT'][0]) + '.' + str(abs(dflst[dnum]['LAT'][0] * 1e3))[:4]
    print dflst[dnum]['LINE'].head()
    submode = df_sub['TOPOGRAPHY_radar'].mode()[0]
    if submode < 50:
        clevel = submode
        # print 'ji'
    dflst[dnum].loc[:, 'HYDROAPPX'] = (clevel - (dflst[dnum]['TOPOGRAPHY_radar'] - clevel) * 7.759) # or SURFACE_atm

### Merge back together
df_sub = pd.concat(dflst)
# df_sub = pd.DataFrame.from_dict(map(dict, dflst))
df_sub.loc[df_sub['HYDROAPPX'] < -1500, 'HYDROAPPX'] = np.nan

#########################################################
### LINE plots
#########################################################
#plt.figure(); df_sub.plot(subplots=True,layout=(4,8),figsize=(16, 12));
# df_sub['SURFACE_atm'].where((df_sub['NUMUSED'] > 0)).plot(legend=True,label='Disturbed',style ='r-');#axes[0,0].set_title('A');

### more control
fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(12, 8))
df_sub['FAG070'].where((df_sub['FLTENVIRO'] == 1)).plot(ax=axes[0],legend=True,label='Disturbed',style ='r-')  #axes[0,0].set_title('A');
df_sub['FAG070'].where((df_sub['FLTENVIRO'] == 0)).plot(ax=axes[0],legend=True,label='Normal',style ='k-')  #axes[0,0].set_title('A');
df_sub['FAG070'].where((df_sub['FLTENVIRO'] == -1)).plot(ax=axes[0],legend=True,label='Missing',style ='b-');#axes[0,0].set_title('A');
# df_sub['FAG140'].plot(ax=axes[0],legend=True,style ='r-');#axes[0,0].set_title('A');
df_sub['ELEVATION'].plot(ax=axes[1],legend=True,style ='y.');#axes[0,0].set_title('A');
df_sub['TOPOGRAPHY_radar'].plot(ax=axes[1],legend=True,color ='blue');#axes[0,0].set_title('A');
df_sub['HYDROAPPX'].plot(ax=axes[1],legend=True,color ='grey');#axes[0,0].set_title('A');
df_sub['SURFACE_atm'].where((df_sub['NUMUSED'] > 77)).plot(ax=axes[1],legend=True,color ='cyan');#axes[0,0].set_title('A');
df_sub['ICEBASE'].plot(ax=axes[1],legend=True,color ='brown');#axes[0,0].set_title('A');
plt.suptitle(str(df_sub['LINE'][0]), y=1.02)
plt.savefig('./figs/profile_'+str(df_sub['LINE'][0])+'_sub.pdf',bbox_inches='tight')   # save the figure to file

#########################################################
### Save to CSV
#########################################################
df_sub.to_csv('OIB_F'+str(df_sub['FLT'][0])+'_d'+str(df_sub['DAY'][0])+'.csv')
# df.to_csv('OIB_'+str(df_sub['LINE'][0])[:8]+'.csv')
