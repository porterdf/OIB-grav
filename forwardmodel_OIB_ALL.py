# import sys
# import os
import time
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from datetime import datetime, date, time

from fatiando.gravmag import talwani
from fatiando.mesher import Polygon

from my_OIB_functions import *


def forward_oib(basedir, infile, dropturns=False, make_plots=False, diagnostics=False):
    # type: (object, object, object, object) -> object
    # -*- coding: utf-8 -*-
    # %load_ext autoreload
    # %autoreload 2

    pd.options.mode.chained_assignment = None  # None or 'warn' or 'raise'
    pd.set_option("display.max_rows", 20)
    # pd.set_option("precision",13)
    pd.set_option('expand_frame_repr', False)

    '''
    Read in file
    '''
    # if os.path.isdir('/Volumes/C/'):
    #     basedir = '/Volumes/C/data/Antarctic/OIB/ATM/2009_AN_NASA_ATM'
    # else:
    #     basedir = '/Volumes/BOOTCAMP/data/Antarctic/OIB/ATM/2009_AN_NASA_ATM'
    # basedir = '/Users/dporter/Documents/data_local/Antarctica/OIB/'
    # basedir = 'data'
    outdir = os.path.join(basedir, 'integrated', 'forward')
    df = pd.read_csv(infile)

    '''
    Preliminary Processing
    '''
    # Crossing Horizons
    # df.ix[df['SURFACE_atm'] - df['ICEBASE'] <= 0, 'ICEBASE'] = (df['SURFACE_atm'] - 0.1)
    # df.loc[df['surface_recalc'] - df['ICEBASE'] <= 0, 'ICEBASE'] = (df['surface_recalc'] - 0.1)
    df.loc[df['surface_recalc'] - df['icebase_recalc'] <= 0, 'icebase_recalc'] = (df['surface_recalc'] - 0.1)

    #
    df = df.round({'RTOPO2_icemask': 0, 'D_gravmask': 0})

    # TODO Add water block when HYDROAPPROX close to icebase_recalc OR use RTOPO icemask

    if make_plots:
        pdir = os.path.join('figs', str(df['DATE'].values[0]))
        if not os.path.exists(pdir):
            os.makedirs(pdir)

    # Compute distance along transect
    distance = np.zeros((np.size(df['LONG'])))
    for i in range(2, np.size(df['LONG'])):
        distance[i] = distance[i - 1] + haversine([df['LAT'].values[i - 1], df['LONG'].values[i - 1]],
                                                  [df['LAT'].values[i], df['LONG'].values[i]])
    df['DIST'] = distance

    # Drop where no gravity
    if dropturns:
        df = df.dropna(subset=['FAG070'])

    '''
    Run functions to read in each flight
    '''
    dflist = {}
    dflst = [g for _, g in df.groupby(['LINE'])]
    for dnum, dname in enumerate(dflst, start=0):
        # print 'Mode of ATM is %.2f' % (mode)
        print('L%s' % (str(dname['LINE'].values[0])))
        dist = dname['DIST'][0:].values
        fag070 = dname['FAG070'][0:].values
        print('Distance of line is %i km' % (dist[-1] - dist[0]))
        if np.isfinite(fag070).any() and (dist[-1] - dist[0] < 5e2) and (dname['surface_recalc'].any()):
            if diagnostics:
                print('Processing the line.')
            '''
            Extract data from DataFrame
            '''
            rho_i = 915
            rho_w = 1005
            rho_r = 2670

            ### Interpolate
            if dname['icebase_recalc'].isnull().all():
                print('No Icebase for this flight...')
                dname['icebase_recalc'] = dname['surface_recalc'] - 1

            # if dname['surface_recalc'].all():
            #     print('No surface for this flight...')

            # TODO should these horizons be interpolated in previous script (read_OIB_ALL.py)?
            dname['ICESFC_horizon'] = dname['surface_recalc']
            dname['ICESFC_horizon'].interpolate(method='pad', inplace=True)
            dname['ICEBASE_horizon'] = dname['icebase_recalc']
            dname['ICEBASE_horizon'].interpolate(method='linear', limit_area='inside', axis=0, inplace=True)


            # Find ice front position
            firstmean = np.mean(dname['ICEBASE_horizon'].iloc[:5])  # or 'icebase_recalc', but this breaks when no bed
            lastmean = np.mean(dname['ICEBASE_horizon'].iloc[-5:])
            firsticepoint = dname['ICEBASE_horizon'].first_valid_index() - 1
            lasticepoint = dname['ICEBASE_horizon'].last_valid_index() + 1
            if np.isnan(firstmean):
                if diagnostics:
                    print('Missing data at start of line.')
                # firsticepoint = dname['icebase_recalc'].first_valid_index() - 1
                if dname['ICESFC_horizon'].loc[firsticepoint] >= 120:
                    if diagnostics:
                        print('NOT floating.')
                    dname['ICEBASE_horizon'].interpolate(method='linear', limit_direction='backward', axis=0,
                                                         inplace=True)
                else:
                    # make ice constant thick at end of line
                    if diagnostics:
                        print('LIKELY floating.')
                    dname['ICEBASE_horizon'].loc[firsticepoint] = dname['ICESFC_horizon'].loc[firsticepoint] - 1
                    dname['ICEBASE_horizon'].iloc[0] = dname['ICESFC_horizon'].iloc[0] - 1
            else:
                if diagnostics:
                    print('NO missing data at start of line.')

            # Find ice front position
            # lastmean = np.mean(dname['icebase_recalc'].iloc[-5:])
            if np.isnan(lastmean):
                if diagnostics:
                    print('Missing data at end of line.')
                # lasticepoint = dname['icebase_recalc'].last_valid_index() + 1
                if dname['ICESFC_horizon'].loc[lasticepoint] >= 120:
                    if diagnostics:
                        print('NOT floating.')
                    dname['ICEBASE_horizon'].interpolate(method='linear', limit_direction='forward', axis=0,
                                                         inplace=True)
                else:
                    # make ice constant thick at end of line
                    if diagnostics:
                        print('LIKELY floating.')
                    dname['ICEBASE_horizon'].loc[lasticepoint] = dname['ICESFC_horizon'].loc[lasticepoint] - 1
                    dname['ICEBASE_horizon'].iloc[-1] = dname['ICESFC_horizon'].iloc[-1] - 1
            else:
                if diagnostics:
                    print('NO missing data at end of line.')

            ### Final INNER interpolation
            dname['ICEBASE_horizon'].interpolate(method='linear', limit_area='inside', axis=0, inplace=True)

            # Check for crossing horizons AGAIN?
            #     dname.loc[dname['ICESFC_horizon']-dname['ICEBASE_horizon'] <= 0, 'ICEBASE_horizon'] = (dname['ICESFC_horizon'] - 0.1)

            # Create water base, set it to single depth below ice
            dname['WATERBASE_horizon'] = dname['ICEBASE_horizon'] - 1
            if np.mean(dname['ICESFC_horizon'].iloc[:5]) < 120:
                print('Adding water block to START.')
                dname['WATERBASE_horizon'].loc[dname['WATERBASE_horizon'].index[0]:firsticepoint] = \
                dname['WATERBASE_horizon'].loc[
                    firsticepoint + 1]
            if np.mean(dname['ICESFC_horizon'].iloc[-5:]) < 120:
                print('Adding water block to END.')
                dname['WATERBASE_horizon'].loc[lasticepoint:dname['WATERBASE_horizon'].index[-1]] = \
                dname['WATERBASE_horizon'].loc[
                    lasticepoint - 1]

            # if make_plots:
            #     pdir = os.path.join('figs', str(dname['DATE'].values[0]))
            #     print(pdir)
            #     if not os.path.exists(pdir):
            #         os.makedirs(pdir)
            #     oib_lineplot_all(dname, str(dname['DATE'].values[0])[:10] + '_L' + str(dname['LINE'].values[0]),
            #                     os.path.join(pdir, str(dname['LINE'].values[0])+'_forward_lineplot.png'))

            '''
            Convert OIB data to polygon arrays
            '''
            icesfc = dname['ICESFC_horizon'][0:].values
            icebase = dname['ICEBASE_horizon'][0:].values
            # iceoutline = np.append(np.concatenate([icesfc, icebase[::-1]], axis=0), icesfc[0]) # no extension
            iceoutline = make_outline(icesfc, icebase)
            # plt.figure(facecolor='white'); plt.plot(iceoutline[1180:1220])

            watertop = icebase
            waterbase = dname['WATERBASE_horizon'][0:].values
            wateroutline = make_outline(watertop, waterbase)

            rocktop = waterbase
            rockbase = -30000 * np.ones_like(waterbase)
            # rockoutline = np.append(icebase[0], np.concatenate([icebase, z_r], axis=0), icebase[0], icebase[0])
            rockoutline = make_outline(rocktop, rockbase)

            ### Distances
            x = dist * 1000
            xs = make_outline_dist(x, 1e6)

            ### Heights
            elevation = dname['WGSHGT'][0:].values
            # z = int(np.max(elevation))
            # z = int(np.max(icesfc) + 1) * np.ones_like(x)
            try:
                z = elevation + int(np.max(icesfc) + 1)
            except ValueError:
                z = elevation + 50
            # z = np.max(elevation) * np.ones_like(x)
            # z = elevation + 50

            '''
            Build the Polygon
            '''
            props_i = {'density': rho_i}
            props_r = {'density': rho_r}
            props_w = {'density': rho_w}
            # polygon = Polygon(np.transpose([xs, iceoutline]), props_i)
            # polygons = [Polygon(np.transpose([xs, iceoutline]), props_i),
            #             Polygon(np.transpose([xs, rockoutline]), props_r)
            #             ]
            polygons = [Polygon(np.transpose([xs, iceoutline]), props_i),
                        Polygon(np.transpose([xs, wateroutline]), props_w),
                        Polygon(np.transpose([xs, rockoutline]), props_r)
                        ]

            '''
            Forward Model
            '''
            gz = talwani.gz(x, z, polygons)
            gz_adj = (gz - np.nanmean(gz)) + np.nanmean(fag070)
            n = len(gz_adj)
            rmse = np.linalg.norm(gz_adj - fag070) / np.sqrt(n)
            # print 'modeled'

            '''
            Make new channels
            '''
            # print 'make new channels'
            try:
                dname.loc[:, 'rmse'] = rmse
                dname.loc[:, 'RESIDUAL'] = gz_adj - fag070
                dname.loc[:, 'GZ'] = gz_adj
            except:
                dname.loc[:, 'rmse'] = 0
                dname.loc[:, 'RESIDUAL'] = 0
                dname.loc[:, 'GZ'] = 0

            '''
            Models of Intermediate Complexity
            '''
            # dnum = 7

            # Interpolate across NaNs in compilation bedrock
            dflst[dnum]['RTOPO2_bedrock'].interpolate(method='linear', axis=0, inplace=True)

            # Remove Ice Mass Contribution

            # Remove Water Mass Contribution

            # Bouguer correction rock
            # dflst[dnum]['BOUGUER_CORR_1'] = dflst[dnum]['RTOPO2_bedrock'] * 0.0419088 * (np.abs(rho_w - rho_r)) / 1000
            dflst[dnum]['BOUGUER_CORR'] = np.where(dflst[dnum]['RTOPO2_bedrock'] > 0,
                                                  dflst[dnum]['RTOPO2_bedrock'] * 0.0419088 * (
                                                      np.abs(rho_r)) / 1000,
                                                  dflst[dnum]['RTOPO2_bedrock'] * 0.0419088 * (
                                                      np.abs(rho_w - rho_r)) / 1000)

            # Bouguer Anamoly
            dflst[dnum]['BOUGUER_ANOMALY'] = dflst[dnum]['FAG070'] - dflst[dnum]['BOUGUER_CORR']
            # dflst[dnum][['BOUGUER_ANOMALY', 'BOUGUER_ANOMALY_1']].plot();
            # plt.show()

            # Low-pass filter (fill NaNs first)
            dflst[dnum]['BOUGUER_ANOMALY'].interpolate(method='cubic', axis=0, inplace=True)

            ## Savitzky-Golay
            # from scipy import signal
            # dflst[dnum]['BOUGUER_ANOMALY_LP'] = signal.savgol_filter(dflst[dnum]['BOUGUER_ANOMALY'].values,
            #                                                      window, 2, mode='nearest')
            # dflst[dnum][['BOUGUER_ANOMALY', 'BOUGUER_ANOMALY_LP', 'FAG070']].plot();
            # plt.show()

            ## Butterworth (filtfilt)
            from scipy.signal import butter, filtfilt
            import scipy.signal as signal
            N = 1  # Filter order
            # Wn = 0.4 # Cutoff frequency
            lp_length_wallclock = 70
            Wn = 1 / (lp_length_wallclock / np.nanmean(np.diff(dflst[dnum]['UNIX'])))
            B, A = signal.butter(N, Wn, output='ba')
            dflst[dnum]['BOUGUER_ANOMALY_LP'] = signal.filtfilt(B, A, dflst[dnum]['BOUGUER_ANOMALY'].values,
                                                               method="gust")
            # dflst[dnum][['BOUGUER_ANOMALY', 'BOUGUER_ANOMALY_LP', 'FAG070']].plot();
            # plt.show()

            # Floating Ice
            dflst[dnum]['D_floatice'] = np.where(dflst[dnum]['RTOPO2_icemask'] == 2, 1, np.nan)
            # dflst[dnum][['RTOPO2_icemask', 'D_floatice']].plot();
            # plt.show()

            # Set bed beneath floating ice to -2000 AKA 'bedcomp'
            # TODO samples SID and only do this to UNCONSTRAINED
            dflst[dnum]['BEDCOMP'] = np.where((dflst[dnum]['D_floatice'] != 1),# & (dflst[dnum]['D_floatice'] != 1),
                                              dflst[dnum]['ICEBASE_horizon'],
                                              -2000)
            # dflst[dnum][['RTOPO2_bedrock', 'ICESFC_horizon', 'ICEBASE_horizon', 'BEDCOMP']].plot();
            # plt.show()

            '''
            Plot
            '''
            # diagnostics = True
            if make_plots:
                pdir = os.path.join('figs', str(dname['DATE'].values[0]))
                if not os.path.exists(pdir):
                    os.makedirs(pdir)
                if diagnostics:
                    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
                    axes[0].plot(xs, iceoutline, color='cyan', marker='o')
                    axes[1].plot(xs, wateroutline, color='blue')
                    axes[2].plot(xs, rockoutline, color='orange')
                    plt.savefig(os.path.join(pdir, str(dname['LINE'].values[0]) + '_polygonsplot.pdf'))
                    plt.show()

                    # ### Horizons
                    # plt.figure(num=None, figsize=(8, 5), dpi=80, facecolor='w', edgecolor='k')
                    # plt.plot(xs, iceoutline)
                    # # plt.plot(x, icebase, linewidth=1, color='orange', alpha=0.5)
                    # plt.plot(x, z, '-r', linewidth=2, label='Modeled (constant)')
                    # plt.xlim(min(x), max(x))
                    # # plt.legend()

            ### Plot model results
            if make_plots:
                talwani_lineplot(x, fag070, gz_adj, polygons, rmse, np.nanmin(rocktop) - 300, np.nanmax(icesfc) + 300,
                                 str(dname['DATE'].values[0]) + '_L' + str(dname['LINE'].values[0]),
                                 os.path.join(pdir, str(dname['LINE'].values[0]) + '_Talwani_lineplot.pdf'))
            if make_plots:
                if diagnostics:
                    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
                    axes.fill(polygons[0].vertices[:, 0], polygons[0].vertices[:, 1], edgecolor='black', linewidth=1,
                              color='cyan', alpha=0.5)
                    # axes.fill(polygons[1], '.-k', linewidth=1, color='blue', alpha=0.5)
                    # axes.fill(polygons[2], '.-k', linewidth=1, color='orange', alpha=0.5)
                    plt.savefig(os.path.join(pdir, str(dname['LINE'].values[0]) + '_polygonsplot.pdf'))
                    plt.show()

        elif not np.isfinite(fag070).any():
            if diagnostics:
                print('No FAG070 present: skipping.')
        else:
            if diagnostics:
                print('Line over 1000 km: skipping.')
        print(('\n' * 0))

    '''
    Merge back together
    '''
    dfout = {}
    dfout = pd.concat(dflst, sort=True)

    '''
    Map
    '''
    ### Entire flight
    # if make_plots:
    #     oib_mapplot_flight(dfout['LONG'].where((dfout['D_gravmask'] != -1)),
    #                        dfout['LAT'].where((dfout['D_gravmask'] != -1)),
    #                        dfout['FLTENVIRO'].where((dfout['D_gravmask'] != -1)), '',
    #                        'FLTENVIRO ' + str(dfout['DATE'].values[0])[:10],
    #                        os.path.join(pdir, str(dfout['DATE'].values[0])[:10] + '_mapplot_talwani_FLTENVIRO_ALL.png'))
    #
    # if make_plots:
    #     oib_mapplot(dfout['LONG'], dfout['LAT'], dfout['rmse'], 'm',
    #             'rmse '+str(dfout['DATE'].values[0])[:10],
    #             os.path.join(pdir, str(dfout['DATE'].values[0])+'_mapplot_Talwani_rmse_ALL.pdf'))

    if make_plots:
        try:
            oib_mapplot_zoom(dfout['LONG'].where((dfout['D_gravmask'] != -1)),
                             dfout['LAT'].where((dfout['D_gravmask'] != -1)),
                             dfout['BOUGUER_ANOMALY_LP'].where((dfout['D_gravmask'] != -1)), '', '',
                             os.path.join(pdir, str(dfout['DATE'].values[0])[:10] + '_oceanmapplot_BOUGUER_ALL.png'))
            oib_mapplot_zoom(dfout['LONG'].where((dfout['D_gravmask'] != -1)),
                             dfout['LAT'].where((dfout['D_gravmask'] != -1)),
                             dfout['RESIDUAL'].where((dfout['D_gravmask'] != -1)), '', '',
                             os.path.join(pdir, str(dfout['DATE'].values[0])[:10] + '_oceanmapplot_RESIDUAL_ALL.png'))
        except KeyError:
            print("Can't make mapplot zoom for this flight. Sorry - do it yourself.")

    '''
    Save to CSV
    '''
    # TODO: rename some channels?
    # Trim and rename
    if not diagnostics:
        print("Trimming output CSV for export")
        dfout.drop(['FLT', 'FAG100', 'FAG140', 'FX', 'FY',
                    'BOTTOM', 'ELEVATION', 'NUMUSED', 'EOTGRAV', 'FACOR', 'ICEBASE',
                    'SURFACE_radar', 'TOPOGRAPHY_radar', 'INTCOR', 'QUALITY'],
                   axis=1, inplace=True)
    else:
        print("Exporting it all...")

    # Change Precision of certain fields
    dfout = dfout.round({'GZ': 1, 'DIST': 0, 'RESIDUAL': 2, 'rmse': 2, 'FAA': 1,
                         'BOUGUER_CORR': 1, 'BOUGUER_ANOMALY': 1, 'BOUGUER_ANOMALY_LP': 1,
                         'HYDROAPPX': 0, 'icebase_recalc': 1, 'surface_recalc': 1, 'BEDCOMP': 1,
                         'RTOPO2_icemask': 0, 'RTOPO2_bedrock': 0,
                         'ICESFC_horizon': 1, 'ICEBASE_horizon': 1, 'WATERBASE_horizon': 1})
    # dfout['GZ_test'] = dfout['GZ'].map(lambda x: '%2.1f' % x)

    # Write to CSV
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    dfout.to_csv(os.path.join(outdir, 'OIB_' + str(dfout['DATE'].values[0])[:10] + '_forward.csv'))


if __name__ == '__main__':
    start_all = time.time()

    basedir = '/Users/dporter/Documents/data_local/Antarctica/OIB/'
    datadir = 'integrated'
    suffix = '.csv'
    ### Run through each directory
    pattern = os.path.join(basedir, datadir, 'OIB_*' + suffix)
    # pattern = './data/NetCDF/10103/R10103_003*.nc'
    filenames = sorted(glob(pattern))  # , key=alphanum_key)
    print("Filelist:\n%s" % (filenames))
    filecounter = len(filenames)
    for fnum, filename in enumerate(filenames, start=0):
        print("Data file %i is %s" % (fnum, filename))
        # sys.exit(main(timedir))
        # try:
        forward_oib(basedir, filename, dropturns=True, make_plots=True, diagnostics=False)
        # except IOError:
        #     print 'IOError - Data Not Found'
        # except AttributeError:
        #     print 'Attribute Error'

    end_all = time.time()
    print('Processing took {}'.format(end_all - start_all))
