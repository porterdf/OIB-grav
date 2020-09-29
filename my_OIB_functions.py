import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import matplotlib.dates as mdates
# import seaborn as sns
# from datetime import datetime, date, time



def alphanum_key(s):
    import re
    key = re.split(r"(\d+)", s)
    key[1::2] = list(map(int, key[1::2]))
    return key


def haversine(origin, destination):
    # Source: http://www.platoscave.net/blog/2009/oct/5/calculate-distance-latitude-longitude-python/
    import math
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
                                                  * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(
        dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d


def fill_nan(a):
    from scipy import interpolate
    import numpy.ma as ma
    import numpy as np

    '''
    interpolate to fill nan values
    '''
    b = ma.filled(a, np.nan)
    inds = np.arange(b.shape[0])
    good = np.where(np.isfinite(b))
    f = interpolate.interp1d(inds[good], b[good], bounds_error=False)
    c = np.where(np.isfinite(b), b, f(inds))
    return c


def linearly_interpolate_nans(y):
    import numpy as np
    # Fit a linear regression to the non-nan y values

    # Create X matrix for linreg with an intercept and an index
    X = np.vstack((np.ones(len(y)), np.arange(len(y))))

    # Get the non-NaN values of X and y
    X_fit = X[:, ~np.isnan(y)]
    y_fit = y[~np.isnan(y)].reshape(-1, 1)

    # Estimate the coefficients of the linear regression
    beta = np.linalg.lstsq(X_fit.T, y_fit)[0]

    # Fill in all the nan values using the predicted coefficients
    y.flat[np.isnan(y)] = np.dot(X[:, np.isnan(y)].T, beta)
    return y


def catATM(atmdir, date_flight):
    import sys, glob, time

    suffix = '.csv'

    # Get icessn filenames that were passed as arguments TODO use dates to grab only certain files
    pattern = os.path.join(atmdir, 'ILATM2_' + date_flight + '*_smooth_*' + suffix)
    print(('ATM pattern: {}'.format(pattern)))
    try:
        # filenames = [f for f in infiles if f.__contains__('_smooth_') if f.endswith('_50pt.csv')]
        filenames = sorted(glob.glob(pattern))  # , key=alphanum_key)
        # filenames[0]
        print('First up is {0}'.format(filenames[0]))
    except:
        print(__doc__)
    # exit()
    output_filename = os.path.join(atmdir, 'ILATM2_' + date_flight + '_all' + suffix)
    print(('Extracting records TO {0}'.format(output_filename)))
    tiles = [0]
    # Open output file
    with open(output_filename, 'w') as f:
        # Loop through filenames
        for filename in filenames:
            print(('Now extracting records from {0}...'.format(filename)))
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
                    gpsTime = float(line.split(',')[0])
                    # If seconds of day roll over to next day
                    if gpsTime < prevTime:
                        date = str(int(date) + 1)
                    # Create new data record (with EOL in "line" variable)
                    newline = '{0}, {1}'.format(date, line)
                    # print newline
                    f.write(newline + '\n')


def calc_icebase(sfc, thick):
    """
    :param sfc:
    :param thick:
    :return:
    // C0 = icebase_recalc
    // C1 = surface_recalc
    // C2 = THICK_redo
    @var1 = (C1 == DUMMY) ? (DUMMY): (C1 - C2);
    C0 = (@ var1 >= C1) ? (C1): (@ var1);
    """
    # icebase = sfc - thick
    # icebase = np.where(icebase >= 0, sfc, 0)
    icebase = np.where((sfc - thick) >= 0, sfc, sfc - thick)
    return icebase


def calc_surface(atm, radar):
    """
    :param atm:
    :param radar:
    :return:
    //C0=surface_recalc
    //C1=SURFACE_atm_redo
    //C2=TOPOGRAPHY_radar
    C0 = (C1 != DUMMY) ? (C1) : (C2);
    """
    surface = np.where(np.isnan(radar), atm, radar)
    return surface


def get_closest_cell_xr(ds, lat, lon, lat_key='lat', lon_key='lon'):
    """
    SSIA
    :param file_for_latlon:
    :param lat:
    :param lon:
    :return:
    """
    a = abs(ds[lon_key][:] - lon) + abs(ds[lat_key][:] - lat)
    iii, jjj = np.unravel_index(a.argmin(), a.shape)
    return iii, jjj


def get_closest_cell_llz(df_llz, lat, lon, lat_key='lat', lon_key='lon'):
    """
    SSIA
    :param df_llz:
    :param lat:
    :param lon:
    :return:
    """
    lat_llz = df_llz[lat_key][:]
    lon_llz = df_llz[lon_key][:]
    #     print('lat,lon', lat,lon)
    #     a = abs(LON - lon)
    #     print(a.min())
    #     print('Closest lat,lon', LAT[a.idxmin()], LON[a.idxmin()])

    #     a = abs(LAT - lat)
    #     print(a.min())
    #     print('Closest lat,lon', LAT[a.idxmin()], LON[a.idxmin()])

    a = abs(lon_llz - lon) + abs(lat_llz - lat)
    #     print(a.min())
    #     print('Closest lat,lon', LAT[a.idxmin()], LON[a.idxmin()])
    return a.idxmin()


def importOIBrad(basedir, timedir, infile):
    """
    :param basedir:
    :param timedir:
    :return:
    """
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


def importOIBrad_all(raddir, date_flight):
    """
    :param basedir:
    :param timedir:
    :return:
    """
    from glob import glob
    # basedir = '/Users/dporter/Documents/data_local/OIB/OIB/'
    # datadir = 'IRMCR2'
    # infile = '2009_Antarctica_DC8'
    suffix = '.csv'
    pattern = os.path.join(raddir, '*'+date_flight+'*' + suffix)
    filenames = sorted(glob(pattern))  # , key=alphanum_key)
    filecounter = len(filenames)

    df_all = {}
    for fnum, filename in enumerate(filenames, start=0):
        # print "RADAR data file %i is %s" % (fnum, filename)
        df = pd.read_csv(filename, delimiter=",", na_values='-9999.00')
        df.rename(columns={'SURFACE': 'SURFACE_radar'}, inplace=True)

        ### do some DATETIME operations
        # df['DATE'] = str(df['FRAME'][0])[:8]
        df['FRAMESTR'] = df['FRAME'].apply(str)
        df['DATE'] = pd.to_datetime(list(df.FRAMESTR.str[:8]), format='%Y%m%d')
        del df['FRAMESTR']
        df['UNIX'] = df['DATE'].astype(np.int64) // 10 ** 9
        try:
            df['UNIX'] = df['UNIX'] + df['TIME']  # df['UTCTIMESOD']
        except KeyError:
            df['UNIX'] = df['UNIX'] + df['UTCTIMESOD']  # TODO: columns changed after 2016
        df['iunix'] = pd.to_datetime(df['UNIX'] * 10 ** 3, unit='ms')
        df = df.set_index('iunix')
        if fnum == 0:
            df_all = df
        else:
            df_all = pd.concat([df_all, df])
    return df_all


def importOIBatm(atmdir, date_flight):
    """
    :param basedir:
    :param timedir:
    :return:
    """
    # datadir = 'ILATM2'
    prefix = 'ILATM2_'
    suffix = '.csv'
    infile = os.path.join(atmdir, prefix + date_flight + '_all' + suffix)

    ### Read ascii file as csv
    headers = (
        'DATE', 'TIME', 'LAT', 'LON', 'SURFACE_atm', 'SLOPESN', 'SLOPEWE', 'RMS', 'NUMUSED', 'NUMOMIT', 'DISTRIGHT',
        'TRACKID')
    df = pd.read_csv(infile, header=None)  # delimiter=r"\s+",
    df.rename(columns=dict(list(zip(df.columns, headers))), inplace=True)
    # del df['TIME2']

    ### do some DATETIME operations
    df['DATETIME'] = (df.DATE * 1e5) + df.TIME
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
    df['UNIX'] = df['DATE'].astype(np.int64) // 10 ** 9
    df['UNIX'] = df['UNIX'] + df['TIME']
    df['iunix'] = pd.to_datetime(df['UNIX'] * 10 ** 3, unit='ms')
    df = df.set_index('iunix')
    return df


def importOIBgrav(gravdir, timedir, date_flight):
    """
    :param basedir:
    :param timedir:
    :return:
    """
    from glob import glob
    # datadir = 'IGGRV1B/temp'
    # infile = 'IGGRV1B_20091104_13100500_V016'
    # infile = 'IGGRV1B_20091031_11020500_V016'
    # infile = 'IGGRV1B_20091116_15124500_V016'
    suffix = '.txt'
    pattern = os.path.join(gravdir, timedir, 'IGGRV1B_'+date_flight+'*' + suffix)
    print(('gravity file pattern: {}'.format(pattern)))
    infile = sorted(glob(pattern))  # , key=alphanum_key)

    ### Read ascii file as csv
    # metadata ends on line 69, column names on line 70
    headers = (
        'LAT', 'LONG', 'DATE', 'DOY', 'TIME', 'FLT', 'PSX', 'PSY', 'WGSHGT', 'FX', 'FY', 'FZ', 'EOTGRAV', 'FACOR',
        'INTCOR',
        'FAG070', 'FAG100', 'FAG140', 'FLTENVIRO')
    # print "Reading gravity file: %s" % infile[0] + suffix %TODO why did I think this would be a list?
    print("Reading gravity file: %s" % infile[0] + suffix)
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


def make_outline(top, bottom):
    outline = np.append(top[0], np.concatenate([np.append(top, top[-1:]), np.append(bottom[-1:], bottom[::-1])], axis=0))
    outline = np.append(outline, bottom[0])
    outline = np.append(outline, top[0])
    return outline


def make_outline_dist(x, pad=1e6):
    xs = np.append(-pad, np.concatenate([np.append(x, np.max(x)+pad), np.append(np.max(x)+pad, x[::-1])], axis=0))
    xs = np.append(xs, -pad)
    xs = np.append(xs, -pad)
    return xs


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
    # We have to use .values because of a weird bug when passing pandas data
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


def oib_lineplot_all(data, ptitle='test_lineplot', pname='test_lineplot'):
    """
    :param data:
    :param ptitle:
    :param pname:
    :return:
    """
    plotmax = np.nanmax([np.nanmax(data['surface_recalc'])+1000, 200])
    plotmin = np.nanmin([-200, np.nanmin(data['icebase_recalc']) - 300])
    import matplotlib.pyplot as plt
    data.loc[data['HYDROAPPX'] < -1500, 'HYDROAPPX'] = np.nan
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 6),
                             gridspec_kw={"height_ratios":[2,1,3]})
    # Panel 1
    data['FAG070'].where((data['FLTENVIRO'] == 1)).plot(ax=axes[0], legend=True, label='FLTENVIRO = 1', style='r+')
    data['FAG070'].where((data['FLTENVIRO'] == 0)).plot(ax=axes[0], legend=True, label='FLTENVIRO = 0', style='k+')
    try:
        data['FAA'].plot(ax=axes[0], legend=True, label='ANTGG', style='-b,')
    except KeyError:
        print('No FAA to plot')
    plt.tick_params(labelbottom=False)
    # data['FAG070'].where((data['FLTENVIRO'] == -1)).plot(ax=axes[0], legend=True, label='Missing', style='b-')
    # Panel 2
    try:
        data['ADMAP'].plot(ax=axes[1], legend=True, label='ADMAP', style='r-')
    except KeyError:
        print('No FAA to plot')
    # Panel 3
    data['WGSHGT'].plot(ax=axes[2], legend=True, style='y-')
    data['SURFACE_atm'].where((data['NUMUSED'] > 77)).plot(ax=axes[2], legend=True, color='cyan')
    data['surface_recalc'].plot(ax=axes[2], legend=True, marker=".", color='blue')
    data['icebase_recalc'].plot(ax=axes[2], legend=True, color='brown')
    try:
        data['RTOPO2_bedrock'].plot(ax=axes[2], legend=True, label='RTOPO2', color='black')
    except KeyError:
        print('No RTOPO2 to plot')
    data['HYDROAPPX'].plot(ax=axes[2], legend=True, color='grey')
    axes[1].tick_params(labelbottom=False)
    axes[2].set_ylim([plotmin, plotmax])
    # axes[0].tick_params(
    #     axis='x',  # changes apply to the x-axis
    #     which='both',  # both major and minor ticks are affected
    #     bottom=False,  # ticks along the bottom edge are off
    #     top=False,  # ticks along the top edge are off
    #     labelbottom=False)  # labels along the bottom edge are off
    # axes[2].legend(loc = 'upper center', bbox_to_anchor = (0.5, 0.95),
    #                fancybox = True, shadow = True, ncol = 5)
    lgd = axes[2].legend(bbox_to_anchor=(0.50, -0.05), loc="lower center",
                    bbox_transform=fig.transFigure, ncol=3)
    plt.suptitle(ptitle, y=0.98)
    plt.savefig(pname, bbox_extra_artists=(lgd,), bbox_inches='tight')   # save the figure to file
    plt.close(fig)
    return


def oib_lineplot_derived(df, starttime, endtime, ptitle='test_lineplot', pname='test_lineplot'):
    """

    :param df:
    :param starttime:
    :param endtime:
    :param pname:
    :return:
    """
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 8), sharex=True)
    # ax = fig.add_subplot(111)
    ln1 = axes[0].plot(df[['FAG070']].resample('10S').mean(), label='FAG$_{070}$', c='black')
    ln2 = axes[0].plot(df[['FAG140']].resample('10S').mean(), label='FAG$_{140}$', c='red')
    ln3 = axes[1].plot(df[['FX']].resample('S').mean(), marker=',', label='FX', c='grey')
    ln4 = axes[2].plot(df[['surface_recalc']].resample('S').mean(), label='SFC$_{recalc}$', c='cyan', lw=2.5)
    ln5 = axes[2].plot(df[['icebase_recalc']].resample('S').mean(), label='ICEBASE$_{recalc}$', c='blue', lw=2.5)
    ln5 = axes[2].plot(df[['TOPOGRAPHY_radar']].resample('S').mean(), marker=',', label='TOPO$_{radar}$', c='green', lw=0.5)
    ln52 = axes[2].plot(df[['SURFACE_atm']].resample('S').mean(), marker='+', label='SFC$_{atm}$', c='black', lw=0.5)
    ln6 = axes[2].plot(df[['WGSHGT']].resample('S').mean(), ls='dotted', label='Survey Height', c='black', lw=0.5)
    ln7 = axes[2].plot(df[['HYDROAPPX']].resample('S').mean(), label='Hydrostatic Appx', c='grey', lw=0.5)
    ln8 = axes[3].plot(df[['THICK']].resample('S').mean(), label='THICK', c='black')
    try:
        lnS1 = axes[0].plot(df[['free_air_anomaly']].resample('10S').mean(), label='FAA$_{ANTGG}$', c='purple')
    except KeyError:
        print('No ANTGG')
    axes[0].set_ylabel('mGal')
    axes[0].legend(loc="lower left")
    axes[1].set_ylabel('mGal')
    axes[1].legend(loc="lower right")
    axes[2].set_ylim([-1000, 1800])
    axes[2].set_ylabel('m.a.s.l.')
    axes[2].legend(ncol=6)
    plt.xlim(starttime, endtime)
    fig.autofmt_xdate()
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %H%Mz'))
    plt.suptitle(ptitle, y=0.95)
    plt.tight_layout()
    plt.savefig(pname, bbox_inches='tight')
    plt.close(fig)


def oib_mapplot_flight(lon, lat, field, units='', ptitle='test_map', pfile='test_map'):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    # from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    # import cartopy.feature
    try:
        ax = plt.axes(projection=ccrs.PlateCarree())  #figsize=(12, 8),
        plt.scatter(lon, lat, c=field, cmap=cm.jet, s=2, transform=ccrs.PlateCarree())
        # ax.set_extent([-50, -80, -80, -60])
        ax.set_extent([lon.min()+360-2, lon.max()+360+2, lat.min()-2, lat.max()+2], crs=ccrs.PlateCarree())
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
        print("Couldn't make Map Plot.")
    return


def oib_mapplot_hilite(lon, lat, field, data, units='', ptitle='test_map', pfile='test_map'):
    # import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import matplotlib.cm as cm
    # from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    # import cartopy.feature
    try:
        plt.figure(figsize=(8, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        plt.scatter(data['LONG'], data['LAT'], c='black', s=1, transform=ccrs.PlateCarree())
        plt.scatter(lon, lat, c=field, cmap=cm.jet, s=15, transform=ccrs.PlateCarree())
        ax.plot(lon[0], lat[0], 'k*', markersize=7, transform=ccrs.PlateCarree())
        # ax.set_extent([-50, -80, -80, -60])
        ax.coastlines(resolution='10m')
        # ax.add_feature(cartopy.feature.LAND)
        # ax.add_feature(cartopy.feature.OCEAN)
        ax.gridlines(draw_labels=True, alpha=0.3, color='grey')
        ax.plot(lon[0], lat[0], 'b+', markersize=7, transform=ccrs.PlateCarree())
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
        print("Couldn't make Map Plot.")
    return

def oib_mapplot_zoom(lon, lat, field, units='', ptitle='test_map', pfile='test_map'):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import cartopy.feature
    import shapely.geometry as sgeom
    try:
        # box = sgeom.box(minx=0, maxx=360, miny=-88, maxy=-56)
        # x0, y0, x1, y1 = box.bounds
        myproj = ccrs.SouthPolarStereo(central_longitude=180)

        plt.figure(figsize=(8, 4), facecolor='white', dpi=144)
        ax = plt.axes(projection=myproj)

        #         plt.scatter(data['LONG'], data['LAT'], c='black', s=1, transform=ccrs.PlateCarree())
        plt.scatter(lon, lat, c=field, cmap=cm.jet, s=15, transform=ccrs.PlateCarree())
        c = plt.colorbar(orientation='vertical', shrink=0.4, pad=0.10)
        c.ax.set_yticklabels(c.ax.get_yticklabels(), rotation=0)
        c.set_label(units)
        # transform = ccrs.PlateCarree()._as_mpl_transform(ax)
        # for i, txt in enumerate(field):
        #     # print '%s: %s, %s' % (txt, lon[i], lat[i])
        #     ax.annotate(txt, (lon[i], lat[i]), xycoords=transform,
        #                 ha='right', va='top')

        ax.coastlines(resolution='50m')
        # ax.add_feature(cartopy.feature.LAND)
        # ax.add_feature(cartopy.feature.OCEAN)
        # ax.gridlines(draw_labels=False, alpha=0.3, color='grey')
        # ax.xformatter = LONGITUDE_FORMATTER
        # ax.yformatter = LATITUDE_FORMATTER
        # ax.set_extent([x0, x1, y0, y1], ccrs.PlateCarree())

        plt.tight_layout()
        plt.subplots_adjust(top=1.25)
        plt.suptitle(ptitle, y=0.98)
        plt.savefig(pfile, bbox_inches='tight')  # save the figure to file
        # plt.show()
        # plt.close()
    except IndexError:
        print("Couldn't make Map Plot.")
    return


def talwani_lineplot(x, fag070, gz_adj, polygons, rmse, pmin=-2000, pmax=2000,
                     ptitle='test_talwaniplot', pname='test_talwaniplot'):

    # from fatiando.vis import mpl
    import matplotlib.ticker as ticker
    import matplotlib.pyplot as plt


    fig = plt.figure(num=None, figsize=(8, 5), dpi=80, facecolor='w', edgecolor='k')
    plt.axis('scaled')
    ax1 = plt.subplot(2, 1, 1)
    plt.title(ptitle)
    plt.plot(x, gz_adj, '-r', linewidth=0.7, label='Modeled (WGSHGT)', zorder=1)
    # plt.plot(x, gz-fag070, '-b', linewidth=1)
    plt.plot(x, fag070, color='k', linestyle="None", marker='.', label='FAG070', zorder=0)
    plt.xlim(min(x), max(x))
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / 1e3))
    ax1.xaxis.set_major_formatter(ticks_x)
    plt.ylabel("mGal")
    plt.legend()
    # ax1.annotate('rmse: '+str(int(rmse))+' mGal',
    #              xy=(min(x)*1.1, np.mean(gz)*1.1), xytext=(min(x)*1.1, np.mean(gz)*1.1))
    if not np.isnan(rmse):
        ax1.annotate('rmse: ' + str(int(rmse)) + ' mGal',
                     xy=(x[2], np.mean(gz_adj)), xytext=(x[2], np.mean(gz_adj)))
    ###
    ax2 = plt.subplot(2, 1, 2)
    ax2.fill(polygons[0].vertices[:, 0], polygons[0].vertices[:, 1], edgecolor='black', linewidth=1, alpha=0.5,
             color='cyan')
    ax2.fill(polygons[1].vertices[:, 0], polygons[1].vertices[:, 1], edgecolor='black', linewidth=1, alpha=0.5,
             color='blue')
    ax2.fill(polygons[2].vertices[:, 0], polygons[2].vertices[:, 1], edgecolor='black', linewidth=1, alpha=0.5,
             color='orange')
    plt.xlim(min(x), max(x))
    plt.ylim(pmin, pmax)
    plt.xlabel("Distance [km]")
    plt.ylabel("Ellipsoid Height [m]")
    ax2.xaxis.set_major_formatter(ticks_x)
    # mpl.savefig('figs/' + str(dflst[dnum]['LINE'].values[0]) + '_forward_lineplot.png', bbox_inches='tight')
    # mpl.suptitle(ptitle, y=0.98)
    plt.savefig(pname, bbox_inches='tight')   # save the figure to file
    plt.close(fig)
    return
