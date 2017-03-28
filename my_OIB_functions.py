import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
# import seaborn as sns
# from datetime import datetime, date, time

def alphanum_key(s):
    import re
    key = re.split(r"(\d+)", s)
    key[1::2] = map(int, key[1::2])
    return key


def catATM(basedir, timedir):
    datadir = 'ILATM2'
    # infile = '2009_AN_NASA_ATM_all'
    suffix = '.csv'
    import sys, glob, time
    # Get icessn filenames that were passed as arguments
    pattern = os.path.join(basedir, datadir, timedir, '*_smooth_*' + suffix)
    try:
        # filenames = [f for f in infiles if f.__contains__('_smooth_') if f.endswith('_50pt.csv')]
        filenames = sorted(glob.glob(pattern))  # , key=alphanum_key)
        # filenames[0]
        print 'Extracting records from {0}...'.format(filenames[0])
    except:
        print __doc__
    # exit()
    output_filename = os.path.join(basedir, datadir, timedir, 'ILATM2_' + timedir + '_all' + suffix)
    tiles = [0]
    # Open output file
    with open(output_filename, 'w') as f:
        # Loop through filenames
        for filename in filenames:
            # 		print 'Extracting records from {0}...'.format(filename)
            # Get date from filename
            # date = '20' + filename[:6]	# this is Linky's original code
            date = os.path.basename(filename)[7:15]
            prevTime = 0
            # Loop through lines in icessn file
            for line in open(filename):
                # Make sure records have the correct number of words (11)
                if (len(line.split()) == 11) and (int(line.split()[-1]) in tiles):
                    line = line.strip()
                    # gpsTime = float(line.split()[0])
                    gpsTime = line.split()[0]
                    # If seconds of day roll over to next day
                    if gpsTime < prevTime:
                        date = str(int(date) + 1)
                    # Create new data record (with EOL in "line" variable)
                    newline = '{0}, {1}'.format(date, line)
                    # print newline
                    f.write(newline + '\n')


def importOIBrad(basedir, timedir, infile):
    datadir = 'IRMCR2'
    # infile = '2009_Antarctica_DC8'
    suffix = '.csv'

    ### Read ascii file as csv
    # headers = ('LAT','LONG','DATE','DOY','TIME','FLT','PSX','PSY','WGSHGT','FX','FY','FZ','EOTGRAV','FACOR','INTCOR','FAG070','FAG100','FAG140','FLTENVIRO')
    df = pd.read_csv(os.path.join(basedir, datadir, timedir, infile + suffix),
                     delimiter=",", na_values='-9999.00')
    # df.replace('-9999.00', np.nan)
    df.rename(columns={'SURFACE': 'SURFACE_radar'}, inplace=True)

    ### do some DATETIME operations
    # df['DATE'] = str(df['FRAME'][0])[:8]
    df['FRAMESTR'] = df['FRAME'].apply(str)
    df['DATE'] = pd.to_datetime(list(df.FRAMESTR.str[:8]), format='%Y%m%d')
    del df['FRAMESTR']
    df['UNIX'] = df['DATE'].astype(np.int64) // 10 ** 9
    df['UNIX'] = df['UNIX'] + df['TIME']
    df['iunix'] = pd.to_datetime(df['UNIX'] * 10 ** 3, unit='ms')
    df = df.set_index('iunix')
    return df


def importOIBrad_all(basedir, timedir):
    from glob import glob
    basedir = '/Users/dporter/Documents/data_local/Antarctica/OIB/'
    datadir = 'IRMCR2'
    # infile = '2009_Antarctica_DC8'
    suffix = '.csv'
    pattern = os.path.join(basedir, datadir, timedir, 'IRMCR2_*' + suffix)
    filenames = sorted(glob(pattern))  # , key=alphanum_key)
    filecounter = len(filenames)
    df_all = {}
    for fnum, filename in enumerate(filenames, start=0):
        print "Data file %i is %s" % (fnum, filename)
        df = pd.read_csv(filename, delimiter=",", na_values='-9999.00')
        df.rename(columns={'SURFACE': 'SURFACE_radar'}, inplace=True)

        ### do some DATETIME operations
        # df['DATE'] = str(df['FRAME'][0])[:8]
        df['FRAMESTR'] = df['FRAME'].apply(str)
        df['DATE'] = pd.to_datetime(list(df.FRAMESTR.str[:8]), format='%Y%m%d')
        del df['FRAMESTR']
        df['UNIX'] = df['DATE'].astype(np.int64) // 10 ** 9
        df['UNIX'] = df['UNIX'] + df['TIME']
        df['iunix'] = pd.to_datetime(df['UNIX'] * 10 ** 3, unit='ms')
        df = df.set_index('iunix')
        if fnum == 0:
            df_all = df
        else:
            df_all = pd.concat([df_all, df])

    return df_all


def importOIBatm(basedir, timedir):
    datadir = 'ILATM2'
    infile = 'ILATM2_'
    suffix = '.csv'

    ### Read ascii file as csv
    headers = (
        'DATE', 'TIME', 'LAT', 'LON', 'SURFACE_atm', 'SLOPESN', 'SLOPEWE', 'RMS', 'NUMUSED', 'NUMOMIT', 'DISTRIGHT',
        'TRACKID')
    df = pd.read_csv(os.path.join(basedir, datadir, timedir, infile + timedir + '_all' + suffix),
                     header=None)  # delimiter=r"\s+",
    df.rename(columns=dict(zip(df.columns, headers)), inplace=True)
    # del df['TIME2']

    ### do some DATETIME operations
    df['DATETIME'] = (df.DATE * 1e5) + df.TIME
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
    df['UNIX'] = df['DATE'].astype(np.int64) // 10 ** 9
    df['UNIX'] = df['UNIX'] + df['TIME']
    df['iunix'] = pd.to_datetime(df['UNIX'] * 10 ** 3, unit='ms')
    df = df.set_index('iunix')
    return df


def importOIBgrav(basedir, timedir):
    from glob import glob
    datadir = 'IGGRV1B'
    # infile = 'IGGRV1B_20091104_13100500_V016'
    # infile = 'IGGRV1B_20091031_11020500_V016'
    # infile = 'IGGRV1B_20091116_15124500_V016'
    suffix = '.txt'
    pattern = os.path.join(basedir, datadir, timedir, 'IGGRV1B_*' + suffix)
    infile = sorted(glob(pattern))  # , key=alphanum_key)

    ### Read ascii file as csv
    # metadata ends on line 69, column names on line 70
    headers = (
        'LAT', 'LONG', 'DATE', 'DOY', 'TIME', 'FLT', 'PSX', 'PSY', 'WGSHGT', 'FX', 'FY', 'FZ', 'EOTGRAV', 'FACOR',
        'INTCOR',
        'FAG070', 'FAG100', 'FAG140', 'FLTENVIRO')
    print "Reading gravity file: %s" % infile[0] + suffix
    df = pd.read_csv(infile[0], delimiter=r"\s+", header=None, names=headers, skiprows=70)
    # headers = df.columns[1:df.shape[1]]
    # df.rename(columns=dict(zip(df.columns,headers)), inplace=True)
    # df.rename(columns={'LONG': 'LON'}, inplace=True)
    # df['ENVIRO'] = df.columns[[19]]
    # df.drop(df.columns['FLTENVIRO'],axis=1,inplace=True)

    ### do some DATETIME operations
    df['DATETIME'] = (df.DATE * 1e5) + df.TIME
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
    df['UNIX'] = df['DATE'].astype(np.int64) // 10 ** 9
    df['UNIX'] = df['UNIX'] + df['TIME']
    df['iunix'] = pd.to_datetime(df['UNIX'] * 10 ** 3, unit='ms')
    df.drop(['DATETIME'], axis=1, inplace=True)
    df = df.set_index('iunix')
    return df


def mapplotOIBlocal(x, y, z, title, units, range, colormap):
    # import matplotlib.pyplot as plt
    #########################################
    ### FULL FAG070
    # import matplotlib.cm as cm
    from mpl_toolkits.basemap import Basemap
    fig = plt.figure(figsize=(6.5, 8))
    # create basemap
    m = Basemap(width=600000, height=900000,
                resolution='h', projection='stere', area_thresh=100000.0,
                lat_ts=-58, lat_0=-67, lon_0=-63.)
    # Draw the coastlines on the map
    m.drawcoastlines()
    # Draw country borders on the map
    m.drawcountries()
    # Fill the land with grey
    # m.fillcontinents(color = 'gainsboro')
    # m.drawmapboundary(fill_color='steelblue')
    # m.bluemarble()
    # m.etopo()
    m.shadedrelief()
    m.drawcoastlines()
    m.drawparallels(np.arange(-80., 81., 5.), labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(-180., 181., 5.), labels=[0, 0, 0, 1])
    # Define our longitude and latitude points
    # We have to use .values because of a wierd bug when passing pandas data
    # to basemap.
    [x1, y1] = m(x.values, y.values)
    # Plot them using round markers of size 6
    m.scatter(x1, y1, c=z, marker="o", cmap=colormap, s=40, edgecolors='none', vmin=range[0],
              vmax=range[1])  # ,vmin=-150,vmax=50
    c = plt.colorbar(orientation='vertical', shrink=0.7)
    c.set_label(units)
    # plt.show()
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.savefig('mapplot_' + title + '.pdf', bbox_inches='tight')  # save the figure to file
    # plt.close(m)
    return


def oib_lineplot(data, ptitle='test_lineplot', pname='test_lineplot'):
    """
    :param data:
    :param ptitle:
    :param pname:
    :return:
    """
    import matplotlib.pyplot as plt
    data.loc[data['HYDROAPPX'] < -1500, 'HYDROAPPX'] = np.nan
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    data['FAG070'].where((data['FLTENVIRO'] == 1)).plot(ax=axes[0], legend=True, label='Disturbed', style='r-')
    data['FAG070'].where((data['FLTENVIRO'] == 0)).plot(ax=axes[0], legend=True, label='Normal', style='k-')
    # data['FAG070'].where((data['FLTENVIRO'] == -1)).plot(ax=axes[0], legend=True, label='Missing', style='b-')
    data['ELEVATION'].plot(ax=axes[1], legend=True, style='y.')
    data['TOPOGRAPHY_radar'].plot(ax=axes[1], legend=True, color='blue')
    data['HYDROAPPX'].plot(ax=axes[1], legend=True, color='grey')
    data['SURFACE_atm'].where((data['NUMUSED'] > 77)).plot(ax=axes[1], legend=True, color='cyan')
    data['ICEBASE'].plot(ax=axes[1], legend=True, color='brown')
    plt.suptitle(ptitle, y=0.98)
    plt.savefig(pname, bbox_inches='tight')   # save the figure to file
    plt.close(fig)
    return


def oib_mapplot(lon, lat, field, units='', ptitle='test_map', pfile='test_map'):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    # from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    # import cartopy.feature
    try:
        ax = plt.axes(projection=ccrs.PlateCarree())
        plt.scatter(lon, lat, c=field, cmap=cm.jet, s=10, transform=ccrs.PlateCarree())
        # ax.set_extent([-50, -80, -80, -60])
        ax.coastlines(resolution='10m')
        # ax.add_feature(cartopy.feature.LAND)
        # ax.add_feature(cartopy.feature.OCEAN)
        ax.gridlines(draw_labels=True, alpha=0.3, color='grey')
        ax.plot(lon[0], lat[0], 'bo', markersize=7, transform=ccrs.PlateCarree())
        ax.xlabels_top = ax.ylabels_right = False
        # plt.xformatter = LONGITUDE_FORMATTER
        # ax.yformatter = LATITUDE_FORMATTER
        c = plt.colorbar(orientation='vertical', shrink=0.8, pad=0.10)
        c.ax.set_yticklabels(c.ax.get_yticklabels(), rotation=90)
        c.set_label(units)
        # plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.suptitle(ptitle, y=0.98)
        plt.savefig(pfile, bbox_inches='tight')  # save the figure to file
        # plt.show()
        plt.close()
    except IndexError:
        print "Couldn't make Map Plot."
    return


def oib_mapplot_hilite(lon, lat, field, data, units='', ptitle='test_map', pfile='test_map'):
    # import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    # from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    # import cartopy.feature
    try:
        plt.figure(figsize=(8, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        plt.scatter(data['LONG'], data['LAT'], c='black', s=1, transform=ccrs.PlateCarree())
        plt.scatter(lon, lat, c=field, cmap=cm.jet, s=15, transform=ccrs.PlateCarree())
        ax.plot(lon[0], lat[0], 'k*', markersize=7, transform=ccrs.PlateCarree())
        # ax.set_extent([-50, -80, -80, -60])
        ax.coastlines(resolution='10m')
        # ax.add_feature(cartopy.feature.LAND)
        # ax.add_feature(cartopy.feature.OCEAN)
        ax.gridlines(draw_labels=True, alpha=0.3, color='grey')
        ax.plot(lon[0], lat[0], 'bo', markersize=7, transform=ccrs.PlateCarree())
        ax.xlabels_top = ax.ylabels_right = False
        # plt.xformatter = LONGITUDE_FORMATTER
        # ax.yformatter = LATITUDE_FORMATTER
        c = plt.colorbar(orientation='vertical', shrink=0.8, pad=0.10)
        c.ax.set_yticklabels(c.ax.get_yticklabels(), rotation=90)
        c.set_label(units)
        # plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.suptitle(ptitle, y=0.98)
        plt.savefig(pfile, bbox_inches='tight')  # save the figure to file
        # plt.show()
        plt.close()
    except IndexError:
        print "Couldn't make Map Plot."
    return
