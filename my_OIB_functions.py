import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
from datetime import datetime, date, time

def importOIBrad(basedir):
    datadir = 'IRMCR2'
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
    return df
            
def importOIBatm(basedir):
    datadir = 'ILATM2'
    infile = '2009_AN_NASA_ATM_all'
    suffix = '.txt'
    
    ### Read ascii file as csv
    headers = ('DATE','TIME','TIME2','LAT','LON','SURFACE_atm','SLOPESN','SLOPEWE','RMS','NUMUSED','NUMOMIT','DISTRIGHT','TRACKID')
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
    return df
    
def importOIBgrav(basedir,timedir,infile):
    datadir = 'IGGRV1B'
    #infile = 'IGGRV1B_20091104_13100500_V016'
    #infile = 'IGGRV1B_20091031_11020500_V016'
    #infile = 'IGGRV1B_20091116_15124500_V016'
    
    suffix = '.txt'
    
    ### Read ascii file as csv
    #metadata ends on line 69, column names on line 70
    #headers = ('LAT','LONG','DATE','DOY','TIME','FLT','PSX','PSY','WGSHGT','FX','FY','FZ','EOTGRAV','FACOR','INTCOR','FAG070','FAG100','FAG140','FLTENVIRO')
    print "Reading gravity file: %s" % infile+suffix    
    df = pd.read_csv(os.path.join(basedir, datadir, timedir, infile+suffix),
                     delimiter=r"\s+", skiprows=69)
    headers = df.columns[1:df.shape[1]]
    df.rename(columns=dict(zip(df.columns,headers)), inplace=True)
    df.rename(columns={'LONG': 'LON'}, inplace=True)
    #df['ENVIRO'] = df.columns[[19]]
    #df.drop(df.columns['FLTENVIRO'],axis=1,inplace=True)
    
    ### do some DATETIME operations
    df['DATETIME'] = (df.DATE*1e5)+df.TIME
    df['DATE'] = pd.to_datetime(df['DATE'],format='%Y%m%d')
    df['UNIX']=df['DATE'].astype(np.int64) // 10**9
    df['UNIX']=df['UNIX']+df['TIME']
    df['iunix'] = pd.to_datetime(df['UNIX'],unit='s')
    df = df.set_index('iunix')
    return df
    
def mapplotOIBlocal(x,y,z,title,units,range,colormap):
    #import matplotlib.pyplot as plt
    #########################################
    ### FULL FAG070
    #import matplotlib.cm as cm
    from mpl_toolkits.basemap import Basemap
    fig = plt.figure(figsize=(6.5,8))
    #create basemap
    m = Basemap(width=600000,height=900000,
                resolution='h',projection='stere',area_thresh = 100000.0,
                lat_ts=-58,lat_0=-67,lon_0=-63.)
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
    [x1,y1] = m(x.values,y.values)
    # Plot them using round markers of size 6
    m.scatter(x1,y1,c=z,marker="o",cmap=colormap,s=40, edgecolors='none',vmin=range[0],vmax=range[1])#,vmin=-150,vmax=50
    c = plt.colorbar(orientation='vertical', shrink = 0.7)
    c.set_label(units)
    #plt.show()
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.savefig('mapplot_'+title+'.pdf',bbox_inches='tight')   # save the figure to file
    #plt.close(m)
    return