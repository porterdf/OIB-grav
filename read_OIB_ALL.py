import sys
import os
from glob import glob
import time
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import seaborn as sns
# from datetime import datetime, date, time
from my_OIB_functions import *


def make_oib_csv(basedir, timedir, date_flight, print_diagnostics=False, make_plots=True, sample_grid=False):
    # -*- coding: utf-8 -*-
    # %load_ext autoreload
    # %autoreload 2

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

    '''
    Run functions to read in each data set
    '''
    ### Gravity
    grv = importOIBgrav(gravdir, timedir, date_flight)

    ### ATM
    if os.path.exists(os.path.join(basedir, 'ILATM2', timedir)):
        if not os.path.exists(os.path.join(basedir, 'ILATM2', timedir, 'ILATM2_' + date_flight + '_all.csv')):
            catATM(os.path.join(basedir, 'ILATM2', timedir), date_flight)  # only do this if not already done
        atm = {}
        try:
            atm = importOIBatm(os.path.join(basedir, 'ILATM2', timedir), date_flight)
        except AttributeError:
            print('No ATM data for this flight.')
    else:
        print('No SURFACE_atm data for this flight.')
        atm = pd.DataFrame(index=grv.index, columns=['SURFACE_atm', 'NUMUSED'])

    ### RADAR
    # if os.path.exists(os.path.join(basedir, 'IRMCR2', timedir, '**', '*'+date_flight+'*.csv')):
    # if os.path.isdir(os.path.join(basedir, 'IRMCR2', timedir, date_flight + '_*')):
    if np.shape(sorted(glob(os.path.join(basedir, 'IRMCR2', timedir, '**', '*'+date_flight+'*.csv'))))[0] != 0:
        rad = {}
        try:
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

    '''
    subset one glacier profile by hand
    '''
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
    # ### Some plots
    # df['FAG070'].plot
    # fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(16, 12))
    # df['FAG070'].plot(ax=axes[0],legend=True);#axes[0,0].set_title('A');
    # df['ICEBASE'].plot(ax=axes[1],legend=True);#axes[0,0].set_title('A');
    # plt.figure(); df.ix['2009-10-31 15:30:00':'2009-10-31 15:50:00'].plot(subplots=True,layout=(5,6),figsize=(11, 8));

    '''
    using FX moving average
    '''
    # # window = 240 for plane maneuvers > 2 minutes
    # grv['FX_MA'] = grv['FX'].rolling(window=240, center=True).mean()
    # grv_sub = grv.query('(WGSHGT < 3000) & (FX_MA < 70000)')

    '''
    Read in Grids to Sample
    '''
    if sample_grid:
        print("\nReading in some grids to be sampled.")
        griddir = '/Users/dporter/Documents/data_local/'

        # ADMAP
        datadir = 'Antarctica/Geophysical/ADMAP/'

        suffix = '.dat'
        pattern = os.path.join(griddir, datadir, 'ant_new*' + suffix)
        suffix = '.llz'
        pattern = os.path.join(basedir, datadir, 'ADMAP_ORSTEDcombined' + suffix)

        filenames = sorted(glob(pattern))  # , key=alphanum_key)
        print("Reading {}".format(filenames[0]))
        admap = pd.read_csv(filenames[0], delimiter=r"\s+", names=('lat', 'lon', 's'), header=None)
        df['ADMAP'] = np.nan

        # Scheinert_2016
        datadir = 'Antarctica/Geophysical/Scheinert_2016/'
        suffix = '.nc'
        pattern = os.path.join(griddir, datadir, 'antgg*' + suffix)
        filenames = sorted(glob(pattern))  # , key=alphanum_key)
        import xarray as xr
        sample_var = ['free_air_anomaly', 'bouguer_anomaly', 'orthometric_height']
        print("{} from {}".format(sample_var, filenames[0]))
        start_sample = time.time()
        ds = xr.open_dataset(filenames[0])

        for v, var in enumerate(sample_var):
            df[var] = np.nan
        # test = np.full([df.shape[0], ], np.nan)
        # lat_sample = np.full([df.shape[0], ], np.nan)
        # lon_sample = np.full([df.shape[0], ], np.nan)
        for i in range(0, df.shape[0], 50):
            # for i in range(0, 30000, 50):
            ii, jj = get_closest_ANTGG_cell(ds, df['LAT'][i], df['LONG'][i])
            #     print('ii: {}\njj: {}'.format(ii, jj))
            # lat_sample[i / 50] = ds['latitude'][ii, jj].values
            # lon_sample[i / 50] = ds['longitude'][ii, jj].values
            #     print(lat_sample[i])
            #     test[i] = ds['bouguer_anomaly'].isel(x=jj, y=ii).values
            iii = (abs(admap['lon'][:] - df['LONG'][i]) + abs(admap['lat'][:] - df['LAT'][i])).argmin()
            df['ADMAP'].iloc[i] = admap['s'][iii]
            for v, var in enumerate(sample_var):
                df[var].iloc[i] = ds[var][ii, jj].values
        # Fill NaNs
        df['ADMAP'] = fill_nan(df['ADMAP'].values)
        for v, var in enumerate(sample_var):
            df[var] = fill_nan(df[var].values)
        end_sample = time.time()
        print('Sampling {} for {} took {} sec'.format(sample_var, str(date_flight), end_sample - start_sample))

    '''
    Split Dataframe using Gravity quality/presence
    '''
    print("Split Dataframe using Gravity quality/presence")
    df['D_gravmask'] = df['FLTENVIRO']
    df.loc[df['D_gravmask'] == 0, 'D_gravmask'] = 1
    dflist = {}
    dflst = [g for _, g in df.groupby((df.D_gravmask.diff() != 0).cumsum())]
    for dnum, dname in enumerate(dflst, start=0):
        ### Add LINE channel
        # dflst[dnum].loc[:, 'LINE'] = str(dflst[dnum]['FLT'][0]) + '.' + str(abs(dflst[dnum]['LAT'][0] * 1e3))[:4]
        dflst[dnum].loc[:, 'LINE'] = str(int(dflst[dnum]['UNIX'][0]))
        mode = min(np.mean(dflst[dnum]['SURFACE_atm'][:10]), np.mean(dflst[dnum]['SURFACE_atm'][10:]))
        if 'orthometric_height' in dflst[dnum].columns:
            # print("Using Scheinert Orthometric Height for Sea level...")
            clevel = dflst[dnum]['orthometric_height'].values
        elif mode < 20:
            # print("Using mode of SURFACE_atm for Sea level...")
            clevel = mode
        else:
            clevel = -10
        if print_diagnostics:
            print 'Mode of ATM is %.2f' % (mode)
            print 'Setting sea-level to %.2f' % (clevel)
        # dflst[dnum].loc[:, 'HYDROAPPX'] = (clevel - (dflst[dnum]['SURFACE_atm'] - clevel) * 7.759)  # or SURFACE_atm
        # # dflst[dnum].loc[:, 'HYDROAPPX'] = clevel - ((dflst[dnum]['SURFACE_atm'] - clevel) * 7.759)  # or SURFACE_atm
        # # (clevel - (((dflst[dnum]['SURFACE_atm'] - clevel) * 7.759))
        # try:
        #     dflst[dnum].loc[:, 'HYDROAPPX'] = 0 - (dflst[dnum]['orthometric_height'].values * 7.759)
        # except:
        #     dflst[dnum].loc[:, 'HYDROAPPX'] = (clevel - (dflst[dnum]['SURFACE_atm'].values - clevel) * 7.759)

    '''
    Merge back together
    '''
    # print "\n Merging back together"
    df2 = {}
    df2 = pd.concat(dflst)
    # df_sub = pd.DataFrame.from_dict(map(dict, dflst))


    '''
    ICEBASE and SURFACE Recalc
    '''
    df2['surface_recalc'] = np.nan
    # df2['surface_recalc'] = df2['SURFACE_atm']
    # df2.loc[df2['surface_recalc'].isnull(), 'surface_recalc'] = df2['TOPOGRAPHY_radar']
    # EXAMPLE df['X'] = np.where(df['Y'] >= 50, 'yes', 'no')
    df2['surface_recalc'] = np.where(df2['SURFACE_atm'].isnull(), df2['TOPOGRAPHY_radar'], df2['SURFACE_atm'])
    print('(sfc_atm - recalc) = {}'.format((df2.SURFACE_atm - df2.surface_recalc).max()))
    print('(TOPO_radar - recalc) = {}'.format((df2.TOPOGRAPHY_radar - df2.surface_recalc).max()))
    # df2[['TOPOGRAPHY_radar', 'SURFACE_atm', 'surface_recalc']].loc['2016-10-14T16:50:02':'2016-10-14T16:50:05']

    # ICEBASE
    df2['icebase_recalc'] = df2['surface_recalc']
    df2.loc[df2['surface_recalc'] != np.nan, 'icebase_recalc'] = (df2['surface_recalc'] - df2['THICK'])
    df2.loc[df2['surface_recalc'] == df2['icebase_recalc'], 'icebase_recalc'] = (df2['icebase_recalc'] - 1)
    # df2[['TOPOGRAPHY_radar', 'SURFACE_atm', 'surface_recalc', 'THICK', 'ICEBASE', 'icebase_recalc']].loc['2016-10-14T16:50:02':'2016-10-14T16:50:05']

    ### Hydrostatic
    try:
        df2.loc[:, 'HYDROAPPX'] = 0 - (df2['surface_recalc'].values * 7.759)
    except:
        df2.loc[:, 'HYDROAPPX'] = 0 - (df2['orthometric_height'].values * 7.759)
    df2.loc[df2['HYDROAPPX'] < -1500, 'HYDROAPPX'] = np.nan


    '''
    Save to CSV
    '''
    outdir = os.path.join(basedir, 'integrated')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    df2.to_csv(os.path.join(outdir, 'OIB_' + str(df2['DATE'][0])[:10] + '.csv'))


    '''
    PLOTS
    '''
    # plt.figure(); df_sub.plot(subplots=True,layout=(4,8),figsize=(16, 12));
    # df['BOTTOM'].where((df['FLTENVIRO'] == 0)).plot(legend=True,label='Good',style ='r-');#axes[0,0].set_title('A');
    # df2['BOTTOM'].where((df2['NUMUSED'] > 0)).plot(legend=True,label='Sparse',style ='r-');#axes[0,0].set_title('A');
    ### Loop through segments

    print "\nStandard Plots"
    pdir = os.path.join(outdir, 'figs')
    # print "Plotting directory is ", pdir
    if not os.path.exists(pdir):
        os.makedirs(pdir)
    # whole flight
    oib_lineplot_derived(df2,
                        df2.index[0], df2.index[-1],
                        str(df2['DATE'][0])[:10],
                        os.path.join(pdir, str(df2['DATE'][0])[:10] + '_lineplot_Flight.png'))

    dflist = {}
    dflst = [g for _, g in df2.groupby((df2.D_gravmask.diff() != 0).cumsum())]
    dnum = 9
    oib_lineplot_derived(dflst[dnum],
                         dflst[dnum].index[0], dflst[dnum].index[-1],
                         str(dflst[dnum]['DATE'][0])[:10] + '_L' + str(dflst[dnum]['LINE'][0]),
                      os.path.join(pdir, str(dflst[dnum]['LINE'][0]) + '_lineplot_derived.png'))
    oib_lineplot_original(dflst[dnum],
                          str(dflst[dnum]['DATE'][0])[:10] + '_L' + str(dflst[dnum]['LINE'][0]),
                          os.path.join(pdir, str(dflst[dnum]['LINE'][0]) + '_lineplot_original.png'))


    if make_plots:
        # individual lines
        pdir = os.path.join(outdir, 'figs', str(dflst[dnum]['DATE'][0])[:10])
        if not os.path.exists(pdir):
            os.makedirs(pdir)
        for dnum, dname in enumerate(dflst, start=0):
            if dnum % 2 != 0:
                print "Plots for Dnum: ", dnum
                try:
                    oib_lineplot_line(dflst[dnum], str(dflst[dnum]['DATE'][0])[:10] + '_L' + str(dflst[dnum]['LINE'][0]),
                                 os.path.join(pdir, str(dflst[dnum]['LINE'][0]) + '_lineplot.png'))
                except:
                    print "couldn't lineplot"
                # try:
                #     oib_mapplot_hilite(dflst[dnum]['LONG'], dflst[dnum]['LAT'], dflst[dnum]['FAG070'], df2, 'm',
                #             'FAG070 ' + str(dflst[dnum]['DATE'][0])[:10] + '_L' + str(dflst[dnum]['LINE'][0]),
                #             os.path.join(pdir, str(dflst[dnum]['LINE'][0])+'_mapplot_FAG070.png'))
                # except:
                #     print "couldn't mapplot"

        # Whole Flight Mapplot
        oib_mapplot(df2['LONG'].where((df2['D_gravmask'] != -1)), df2['LAT'].where((df2['D_gravmask'] != -1)),
                    df2['FLTENVIRO'].where((df2['D_gravmask'] != -1)), 'm', 'FLTENVIRO '+str(df2['DATE'][0])[:10],
                    os.path.join(pdir, str(df2['DATE'][0])[:10]+'_mapplot_FLTENVIRO_ALL.png'))


if __name__ == '__main__':
    basedir = '/Users/dporter/Documents/data_local/Antarctica/OIB/'
    datadir = 'IGGRV1B'
    #
    # custom_dates = ['20161014', '20161015',
    #                 '20161111', '20161112', '20161114', '20161117', '20161118',
    #                 '20171121']
    custom_dates = ['20161014']
    for filename in custom_dates:
        start_year = time.time()
        make_oib_csv(basedir, filename[:4], filename, False, False, True)
        end_year = time.time()
        print('Processing {} took {}'.format(str(filename[:4]), end_year - start_year))
    # # Run through each directory #
    # start_year = 2016
    # end_year = 2016
    # for y, year in enumerate(range(start_year, end_year+1), 1):
    #     print('Year: {}'.format(year))
    #     start_year = time.time()
    #     pattern = os.path.join(basedir, datadir, str(year), 'IGGRV1B_*V???.txt')
    #     filenames = sorted(glob(pattern))  # , key=alphanum_key)
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
    #     end_year = time.time()
    #     print('Processing {} took {}'.format(str(year), end_year - start_year))



    # # directories = next(os.walk(os.path.join(basedir, datadir)))[1]
    # for dnum, dname in enumerate(directories, start=0):
    #     print('\ndname: {}'.format(dname))
    #     dirpath = os.path.join(basedir, datadir)
    #     print('dirpath: {}'.format(dirpath))
    #     # timedir = '2009.10.31'  # TODO make this user specified or for all flights
    #     # dname = '2010.04.19'
    #     # sys.exit(main(timedir))
    #
    #     # try:
    #     make_oib_csv(dirpath, dname, False, False)
    #     except IOError:
    #         print 'IOError - Data Not Found'
    #     except AttributeError:
    #         print 'Attribute Error'
