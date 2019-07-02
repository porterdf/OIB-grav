# import sys
# import os
# from glob import glob
import time
import re
# import pandas as pd
# import numpy as np
# import xarray as xr
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import seaborn as sns
# from datetime import datetime, date, time
from my_OIB_functions import *


def make_oib_csv(basedir, timedir, date_flight, print_diagnostics=False, make_plots=True, sample_grid_line=False):
    # type: (object, object, object, object, object, object) -> object
    """
    main function that integrates OIB data streams and samples grids
    :param basedir:
    :param timedir:
    :param date_flight:
    :param print_diagnostics:
    :param make_plots:
    :param sample_grid_line:
    """
    # -*- coding: utf-8 -*-
    # %load_ext autoreload
    # %autoreload 2

    pd.options.mode.chained_assignment = None  # None or 'warn' or 'raise'
    pd.set_option("display.max_rows", 20)
    # pd.set_option("precision",13)
    pd.set_option('expand_frame_repr', False)

    '''
    Specify Directories
    '''
    # if os.path.isdir('/Volumes/C/'):
    #     basedir = '/Volumes/C/data/Antarctic/OIB/ATM/2009_AN_NASA_ATM'
    # else:
    #     basedir = '/Volumes/BOOTCAMP/data/Antarctic/OIB/ATM/2009_AN_NASA_ATM'
    # basedir = '/Users/dporter/Documents/data_local/Greenland/OIB/'
    gravdir = os.path.join(basedir, 'IGGRV1B')
    outdir = os.path.join(basedir, 'integrated')

    '''
    Run functions to read in each data set
    '''
    ### Gravity
    grv = importOIBgrav(gravdir, timedir, date_flight)

    ### ATM
    if os.path.exists(os.path.join(basedir, 'ILATM2', timedir)):
        if not os.path.exists(os.path.join(basedir, 'ILATM2', timedir, 'ILATM2_' + timedir + '_all.csv')):
            catATM(os.path.join(basedir, 'ILATM2', timedir), timedir)  # only do this if not already done
        atm = {}
        try:
            atm = importOIBatm(os.path.join(basedir, 'ILATM2', timedir), timedir)
        except AttributeError:
            print('No ATM data for this flight.')
    else:
        print('No SURFACE_atm data for this flight.')
        atm = pd.DataFrame(index=grv.index, columns=['SURFACE_atm', 'NUMUSED'])

    ### RADAR
    # if os.path.exists(os.path.join(basedir, 'IRMCR2', timedir, '**', '*'+date_flight+'*.csv')):
    # if os.path.isdir(os.path.join(basedir, 'IRMCR2', timedir, date_flight + '_*')):
    if np.shape(sorted(glob(os.path.join(basedir, 'IRMCR2', timedir, '*' + date_flight + '*.csv'))))[0] != 0:
        rad = {}
        try:
            print('\nRADAR dir: {}'.format(os.path.join(basedir, 'IRMCR2', timedir)))
            rad = importOIBrad_all(os.path.join(basedir, 'IRMCR2', timedir), date_flight)
            # rad = importOIBrad(basedir, timedir, infile)
        except AttributeError:
            print('No MCoRDS data for this flight.')
    else:
        print('No BOTTOM data for this flight.')
        rad = pd.DataFrame(index=grv.index,
                           columns=['THICK', 'ELEVATION', 'FRAME', 'SURFACE_radar', 'BOTTOM', 'QUALITY'])
        # if not rad:
        # if 'BOTTOM' not in rad:

    '''
    Subsample all to 2 Hz
    '''
    rad2hz = {}
    rad2hz = rad.resample('500L').first().bfill(limit=1)  # mean,median,mode???

    atm2hz = {}
    atm2hz = atm.resample('500L').first().bfill(limit=1)  # mean,median,mode???

    '''
    Concatenate into single dataframe
    '''
    df = {}
    # df = pd.concat([grv, rad2hz[['THICK','ELEVATION','FRAME','SURFACE_radar','BOTTOM','QUALITY']]], axis=1,join_axes=[grv.index])
    df = pd.concat([grv, rad2hz[['THICK', 'ELEVATION', 'FRAME', 'SURFACE_radar', 'BOTTOM', 'QUALITY']],
                    atm2hz[['SURFACE_atm', 'NUMUSED']]], axis=1, join_axes=[grv.index])
    # df['DAY'] = df.index.day
    # df['HOUR'] = df.index.hour
    df['ICEBASE'] = df['ELEVATION'] - df['BOTTOM']
    df['TOPOGRAPHY_radar'] = df['ELEVATION'] - df['SURFACE_radar']

    # # subset one glacier profile by hand #
    # df_sub = df['2009-10-31 14:10:00':'2009-10-31 14:50:00']
    #
    # Larsen C
    # df_sub = df['2009-11-04 23:00:00':'2009-11-04 23:15:00']
    # df_sub = df.loc[df['FRAME'] == 2009110401046]
    # df_sub = df.loc[df['FRAME'].isin(['2009110401047','2009110401048'])]
    #
    # THE ONE BELOW WORKS
    # df_sub = df.query('(FRAME <= @subframe+2) & (FRAME >= @subframe-2)')
    #
    # # # Some plots
    # df['FAG070'].plot
    # fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(16, 12))
    # df['FAG070'].plot(ax=axes[0],legend=True);#axes[0,0].set_title('A');
    # df['ICEBASE'].plot(ax=axes[1],legend=True);#axes[0,0].set_title('A');
    # plt.figure(); df.ix['2009-10-31 15:30:00':'2009-10-31 15:50:00'].plot(subplots=True,layout=(5,6),figsize=(11, 8));

    # # using FX moving average
    # # window = 240 for plane maneuvers > 2 minutes
    # grv['FX_MA'] = grv['FX'].rolling(window=240, center=True).mean()
    # grv_sub = grv.query('(WGSHGT < 3000) & (FX_MA < 70000)')

    if sample_grid_line:
        print("\nReading in some grids to be sampled, LATER")
        # sample_var = ['free_air_anomaly', 'bouguer_anomaly', 'orthometric_height']
        sample_var = ['ADMAP', 'FAA', 'RTOPO2_icemask', 'RTOPO2_bedrock']
        griddir = '/Users/dporter/Documents/data_local/'

        # # ADMAP
        if any(substr in 'ADMAP' for substr in sample_var):
            datadir = 'Antarctica/Geophysical/ADMAP/'
            suffix = '.llz'
            pattern = os.path.join(griddir, datadir, 'ADMAP_ORSTEDcombined' + suffix)
            # pattern = os.path.join(griddir, datadir, 'ant_new' + suffix)

            filenames = sorted(glob(pattern))  # , key=alphanum_key)
            print("Reading {}".format(filenames[0]))
            admap = pd.read_csv(filenames[0], delimiter=r"\s+", names=('lat', 'lon', 's'), header=None)
            # df['ADMAP'] = np.nan

        # # Scheinert_2016
        if any(substr in 'FAA' for substr in sample_var):
            # TODO OR any of the other ANTGG fields we may want to sample
            import xarray as xr
            datadir = 'Antarctica/Geophysical/Scheinert_2016/'
            suffix = '.nc'
            pattern = os.path.join(griddir, datadir, 'antgg*' + suffix)
            filenames = sorted(glob(pattern))  # , key=alphanum_key)
            print("Reading {}".format(filenames[0]))
            antgg = xr.open_dataset(filenames[0])

        # # RTOPO-2
        if any(substr in 'RTOPO2_icemask' for substr in sample_var):
            import xarray as xr
            datadir = 'Antarctica/DEM/RTOPO2'
            suffix = '.nc'
            # pattern = os.path.join(griddir, datadir, 'RTopo-2.0.1_1min_aux*' + suffix)
            pattern = os.path.join(griddir, datadir, 'RTopo-2.0.1_30sec_Antarctica_aux.nc')
            filenames = sorted(glob(pattern))  # , key=alphanum_key)
            print(filenames)
            rtopo2_aux = xr.open_dataset(filenames[0])
            rtopo2_aux.set_index(latdim='lat', inplace=True)
            rtopo2_aux.set_index(londim='lon', inplace=True)

        if any(substr in 'RTOPO2_bedrock' for substr in sample_var):
            import xarray as xr
            datadir = 'Antarctica/DEM/RTOPO2'
            pattern = os.path.join(griddir, datadir, 'RTopo-2.0.1_30sec_Antarctica_data.nc')
            filenames = sorted(glob(pattern))  # , key=alphanum_key)
            print(filenames)
            rtopo2_dat = xr.open_dataset(filenames[0])
            rtopo2_dat.set_index(latdim='lat', inplace=True)
            rtopo2_dat.set_index(londim='lon', inplace=True)


    '''
    Split Dataframe using Gravity quality/presence
    '''
    print("Split Dataframe using Gravity quality/presence")
    df['D_gravmask'] = df['FLTENVIRO']
    df.loc[df['D_gravmask'] == 0, 'D_gravmask'] = 1
    dflst = {}
    dflst = [g for _, g in df.groupby((df.D_gravmask.diff() != 0).cumsum())]
    for dnum, dname in enumerate(dflst, start=0):
        # # Add LINE channel
        # dflst[dnum].loc[:, 'LINE'] = str(dflst[dnum]['FLT'][0]) + '.' + str(abs(dflst[dnum]['LAT'][0] * 1e3))[:4]
        dname.loc[:, 'LINE'] = str(int(dname['UNIX'][0]))
        mode = min(np.mean(dname['SURFACE_atm'][:10]), np.mean(dname['SURFACE_atm'][10:]))
        if 'orthometric_height' in dname.columns:
            # print("Using Scheinert Orthometric Height for Sea level...")
            clevel = dname['orthometric_height'].values
        elif mode < 20:
            # print("Using mode of SURFACE_atm for Sea level...")
            clevel = mode
        else:
            clevel = -10
        if print_diagnostics:
            print('Mode of ATM is %.2f' % mode)
            print('Setting sea-level to %.2f' % clevel)
        # dflst[dnum].loc[:, 'HYDROAPPX'] = (clevel - (dflst[dnum]['SURFACE_atm'] - clevel) * 7.759)  # or SURFACE_atm
        # # dflst[dnum].loc[:, 'HYDROAPPX'] = clevel - ((dflst[dnum]['SURFACE_atm'] - clevel) * 7.759)  # or SURFACE_atm
        # # (clevel - (((dflst[dnum]['SURFACE_atm'] - clevel) * 7.759))
        # try:
        #     dflst[dnum].loc[:, 'HYDROAPPX'] = 0 - (dflst[dnum]['orthometric_height'].values * 7.759)
        # except:
        #     dflst[dnum].loc[:, 'HYDROAPPX'] = (clevel - (dflst[dnum]['SURFACE_atm'].values - clevel) * 7.759)

        '''
        Sample LINES
        '''
        if sample_grid_line:
            start_sample = time.time()
            for v, var in enumerate(sample_var):
                dname[var] = np.nan
            # test = np.full([df.shape[0], ], np.nan)
            # lat_sample = np.full([df.shape[0], ], np.nan)
            # lon_sample = np.full([df.shape[0], ], np.nan)
            for i in range(0, dname.shape[0], 70):
                # print(i)
                # for i in range(0, 30000, 50):
                # ii, jj = get_closest_ANTGG_cell(ds, df['LAT'][i], df['LONG'][i])

                #     print('ii: {}\njj: {}'.format(ii, jj))
                # lat_sample[i / 50] = ds['latitude'][ii, jj].values
                # lon_sample[i / 50] = ds['longitude'][ii, jj].values
                #     print(lat_sample[i])
                #     test[i] = ds['bouguer_anomaly'].isel(x=jj, y=ii).values
                # iii = (abs(admap['lon'][:] - df['LONG'][i]) +
                #        abs(admap['lat'][:] - df['LAT'][i])).argmin()
                if any(substr in 'ADMAP' for substr in sample_var):
                    iii = get_closest_cell_llz(admap, dname['LAT'][i], dname['LONG'][i])
                    dname['ADMAP'].iloc[i] = admap['s'].iloc[iii]

                # # TODO: Sample ANTGG in xarray using coordinate slicing (1D lat/lon an issue? - see notebook)
                # if any(substr in 'FAA' for substr in sample_var):
                #     a = abs(antgg['longitude'][:] - dname['LONG'][i]) + \
                #         abs(antgg['latitude'][:] - dname['LAT'][i])
                #     ii, jj = np.unravel_index(a.argmin(), a.shape)
                #     # # Function
                #     # ii, jj = get_closest_cell_xr(antgg, df['LAT'][i], df['LONG'][i])
                #     dname['FAA'].iloc[i] = antgg['free_air_anomaly'].values[ii, jj]

            if any(substr in 'RTOPO2_icemask' for substr in sample_var):
                dname['RTOPO2_icemask'].iloc[::14] = rtopo2_aux.sel(latdim=dname['LAT'].values[::14],
                                                              londim=dname['LONG'].values[::14],
                                                              method='nearest')['amask'].values.diagonal()

            if any(substr in 'RTOPO2_bedrock' for substr in sample_var):
                dname['RTOPO2_bedrock'].iloc[::14] = rtopo2_dat.sel(latdim=dname['LAT'].values[::14],
                                                              londim=dname['LONG'].values[::14],
                                                              method='nearest')['bedrock_topography'].values.diagonal()

            # # Fill NaNs
            for v, var in enumerate(sample_var):
                try:
                    dname[var].interpolate(method='spline', order=1, s=0., axis=0, limit_area='inside', inplace=True)
                    # # options: method = 'spline', order = 1, limit_area='inside', limir=140
                except:
                    dname[var].interpolate(method='linear', limit=140+1, axis=0, inplace=True)
                # dname[var] = fill_nan(dname[var].values)


            end_sample = time.time()
            print(('Sampling {} for {} took {} sec'.format(sample_var,
                                                            str(dname['LINE'][0]), end_sample - start_sample)))

            # TODO grid noise still in looped samples - BSpline or other smoother?

            '''
            ICEBASE and SURFACE Recalc
            '''
            dname['surface_recalc'] = np.nan
            # df2['surface_recalc'] = df2['SURFACE_atm']
            # df2.loc[df2['surface_recalc'].isnull(), 'surface_recalc'] = df2['TOPOGRAPHY_radar']
            # EXAMPLE df['X'] = np.where(df['Y'] >= 50, 'yes', 'no')
            dname['surface_recalc'] = np.where(dname['SURFACE_atm'].isnull(), dname['TOPOGRAPHY_radar'], dname['SURFACE_atm'])
            # print(('(sfc_atm - recalc) = {}'.format((dname.SURFACE_atm - dname.surface_recalc).max())))
            # print(('(TOPO_radar - recalc) = {}'.format((dname.TOPOGRAPHY_radar - dname.surface_recalc).max())))

            # ICEBASE
            dname['icebase_recalc'] = dname['surface_recalc']
            dname.loc[dname['surface_recalc'] != np.nan, 'icebase_recalc'] = (dname['surface_recalc'] - dname['THICK'])
            dname.loc[dname['surface_recalc'] == dname['icebase_recalc'], 'icebase_recalc'] = (dname['icebase_recalc'] - 1)
            # df2[['TOPOGRAPHY_radar', 'SURFACE_atm', 'surface_recalc', 'THICK', 'ICEBASE', 'icebase_recalc']].loc['2016-10-14T16:50:02':'2016-10-14T16:50:05']

            # # Hydrostatic
            try:
                dname.loc[:, 'HYDROAPPX'] = 0 - (dname['surface_recalc'].values * 7.759)
            except:
                dname.loc[:, 'HYDROAPPX'] = 0 - (dname['orthometric_height'].values * 7.759)
            dname.loc[dname['HYDROAPPX'] < -1500, 'HYDROAPPX'] = np.nan

            '''
            PLOTS
            '''
            if make_plots:
                # individual lines
                pdir = os.path.join(outdir, 'figs', str(dname['DATE'][0])[:10])
                if not os.path.exists(pdir):
                    os.makedirs(pdir)
                if dnum % 2 != 0:
                    print('Plots for {}: '.format(str(dname['LINE'][0])))
                    try:
                        oib_lineplot_all(dname,
                                     str(dname['DATE'][0])[:10] + '_L' + str(dname['LINE'][0]),
                                     os.path.join(pdir, str(dname['LINE'][0]) + '_lineplot.png'))
                    except:
                        print("couldn't lineplot")
                    # try:
                    #     oib_mapplot_hilite(dflst[dnum]['LONG'], dflst[dnum]['LAT'], dflst[dnum]['FAG070'], df2, 'm',
                    #             'FAG070 ' + str(dflst[dnum]['DATE'][0])[:10] + '_L' + str(dflst[dnum]['LINE'][0]),
                    #             os.path.join(pdir, str(dflst[dnum]['LINE'][0])+'_mapplot_FAG070.png'))
                    # except:
                    #     print "couldn't mapplot"


    '''
    Merge back together
    '''
    # print "\n Merging back together"
    df2 = {}
    df2 = pd.concat(dflst)
    # df_sub = pd.DataFrame.from_dict(map(dict, dflst))

    # # Whole Flight Mapplot #
    oib_mapplot_flight(df2['LONG'].where((df2['D_gravmask'] != -1)), df2['LAT'].where((df2['D_gravmask'] != -1)),
                df2['FLTENVIRO'].where((df2['D_gravmask'] != -1)), 'm',
                'FLTENVIRO ' + str(df2['DATE'][0])[:10],
                os.path.join(os.path.join(outdir, 'figs', str(dname['DATE'][0])[:10]),
                             str(df2['DATE'][0])[:10] + '_mapplot_FLTENVIRO_ALL.png'))

    '''
    Save to CSV
    '''
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    df2.to_csv(os.path.join(outdir, 'OIB_' + str(df2['DATE'][0])[:10] + '.csv'))


if __name__ == '__main__':
    basedir = '/Users/dporter/Documents/data_local/Antarctica/OIB/'
    datadir = 'IGGRV1B'
    dirpath = os.path.join(basedir, datadir)

    # # When organized by YEAR folders #
    # start_year = 2016
    # end_year = 2017
    # for y, year in enumerate(range(start_year, end_year+1), 1):
    #     print('Year: {}'.format(year))
    #     pattern = os.path.join(basedir, datadir, str(year), 'IGGRV1B_*V???.txt')
    #     filenames = sorted(glob(pattern))  # , key=alphanum_key)
    #     print(filenames)
    #     for filename in filenames:
    #         start = time.time()
    #         print('\n\n---------------------------------------------------------------')
    #         print('filename: {}'.format(filename))
    #         datestring = filename[-26:-18]
    #         dirpath = os.path.join(basedir, datadir)
    #         print('datestring: {}'.format(datestring))
    #         print('dirpath: {}'.format(dirpath))
    #         make_oib_csv(basedir, str(year), datestring, False, False)
    #         end = time.time()
    #         print('Processing {} took {}'.format(datestring, end - start))

    start_all = time.time()

    # # Custom Dates #
    # directories = ['2016.10.14']
    directories = ['2011.10.24', '2011.11.16', '2011.11.19', '2016.10.14']
    # directories = ['2011.10.20', '2011.10.21', '2011.10.29', '2011.10.30',
    #                '2011.11.07', '2012.10.18', '2012.11.02', '2014.10.25',
    #                '2014.11.08', '2016.10.24', '2016.10.25', '2018.10.10',
    #                '2018.10.11', '2018.10.18', '2018.11.14']

    # directories = next(os.walk(os.path.join(basedir, datadir)))[1][0:]
    # if np.size(directories) == 1:     # Not needed in Python3?
    #     directories = list(directories.split())

    for dnum, dname in enumerate(directories, start=0):
        print('\ndnum: {}, dname: {}'.format(dnum, dname))
        # filetime = str(dname)[0:4] + str(dname)[5:7] + str(dname)[8:10]
        print('\ndname: {}'.format(dname))
        print('dirpath: {}'.format(dirpath))
        print('filetime: {}'.format(re.sub('\.', '', dname)))
        try:
            make_oib_csv(basedir, dname, re.sub('\.', '', dname),
                         print_diagnostics=False, make_plots=True, sample_grid_line=True)
        except IOError:
            print('IOError - Data Not Found')
        except AttributeError:
            print('Attribute Error')

    end_all = time.time()
    print('Processing took {}'.format(end_all - start_all))
