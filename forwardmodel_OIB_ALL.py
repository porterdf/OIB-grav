# import sys
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from datetime import datetime, date, time

from fatiando.gravmag import talwani
from fatiando.mesher import Polygon

from my_OIB_functions import *

def forward_oib(infile, make_plots=False):
    # -*- coding: utf-8 -*-
    # %load_ext autoreload
    # %autoreload 2

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
    basedir = 'data'
    df = pd.read_csv(infile)

    '''
    Preliminary Processing
    '''
    ### Crossing Horizons
    df.ix[df['SURFACE_atm'] - df['ICEBASE'] <= 0, 'ICEBASE'] = (df['SURFACE_atm'] - 0.1)

    ### Compute distance along transect
    distance = np.zeros((np.size(df['LONG'])))
    for i in range(2, np.size(df['LONG'])):
        distance[i] = distance[i - 1] + haversine([df['LAT'].values[i - 1], df['LONG'].values[i - 1]],
                                          [df['LAT'].values[i], df['LONG'].values[i]])
    df['DIST'] = distance
    '''
    Run functions to read in each flight
    '''
    dflist = {}
    dflst = [g for _, g in df.groupby(['LINE'])]
    for dnum, dname in enumerate(dflst, start=0):
        # print 'Mode of ATM is %.2f' % (mode)
        print 'L%s' % (str(dflst[dnum]['LINE'].values[0]))
        dist = dflst[dnum]['DIST'][0:].values
        fag070 = dflst[dnum]['FAG070'][0:].values
        print 'Distance of line is %i km' % (dist[-1]-dist[0])
        if np.isfinite(fag070).any() and (dist[-1]-dist[0] < 1e3):
            print 'Processing the line.'
            pdir = 'figs'   #os.path.join('figs', str(dflst[dnum]['DATE'][0])[:10])
            # oib_lineplot(dname, str(dflst[dnum]['DATE'].values[0])[:10] + '_L' + str(dflst[dnum]['LINE'].values[0]),
            #                      os.path.join(pdir, str(dflst[dnum]['LINE'].values[0])+'_forward_lineplot.png'))
            '''
            Extract data from DataFrame
            '''
            rho_i = 915
            rho_w = 1005
            rho_r = 2670
            elevation = dflst[dnum]['WGSHGT'][0:].values
            icesfc = dflst[dnum]['SURFACE_atm'][0:].values
            icebase = dflst[dnum]['ICEBASE'][0:].values

            '''
            Convert OIB data to polygon arrays
            '''
            # iceoutline = np.append(np.concatenate([icesfc, icebase[::-1]], axis=0), icesfc[0]) # no extension
            iceoutline = make_outline(icesfc, icebase)
            # plt.figure(facecolor='white'); plt.plot(iceoutline[1180:1220])

            rockbase = -30000 * np.ones_like(icebase)
            # rockoutline = np.append(icebase[0], np.concatenate([icebase, z_r], axis=0), icebase[0], icebase[0])
            rockoutline = make_outline(icebase, rockbase)

            ### Distances
            x = dist * 1000
            xs = make_outline_dist(x, 1e6)

            ### Heights
            # z = int(np.max(elevation))
            # z = int(np.max(icesfc) + 1) * np.ones_like(x)
            z = np.max(elevation) * np.ones_like(x)  # elevation + int(np.max(icesfc)+1)

            '''
            Build the Polygon
            '''
            props = {'density': rho_i}
            props2 = {'density': rho_r}
            polygon = Polygon(np.transpose([xs, iceoutline]), props)
            polygons = [Polygon(np.transpose([xs, iceoutline]), props),
                        Polygon(np.transpose([xs, rockoutline]), props2)
                        ]

            '''
            Forward Model
            '''
            gz = talwani.gz(x, z, polygons)
            gz_adj = (gz - np.mean(gz)) + np.mean(fag070)
            n = len(gz_adj)
            rmse = np.linalg.norm(gz_adj - fag070) / np.sqrt(n)
            # print 'modeled'

            '''
            Make new channels
            '''
            # print 'make new channels'
            try:
                dflst[dnum].loc[:, 'rmse'] = rmse
                dflst[dnum].loc[:, 'RESIDUAL'] = gz_adj - fag070
                dflst[dnum].loc[:, 'GZ'] = gz_adj
            except:
                dflst[dnum].loc[:, 'rmse'] = 0
                dflst[dnum].loc[:, 'RESIDUAL'] = 0
                dflst[dnum].loc[:, 'GZ'] = 0

            '''
            Plot
            '''
            # print 'plotting'
            pdir = os.path.join('figs', str(dflst[dnum]['DATE'].values[0]))

            # ### Horizons
            # plt.figure(num=None, figsize=(8, 5), dpi=80, facecolor='w', edgecolor='k')
            # plt.plot(xs, iceoutline)
            # # plt.plot(x, icebase, linewidth=1, color='orange', alpha=0.5)
            # plt.plot(x, z, '-r', linewidth=2, label='Modeled (constant)')
            # plt.xlim(min(x), max(x))
            # # plt.legend()

            ### Plot model results
            if make_plots:
                # if not os.path.exists(pdir):
                #     os.makedirs(pdir)
                talwani_lineplot(x, fag070, gz_adj, polygons, rmse,
                                 str(dflst[dnum]['DATE'].values[0]) + '_L' + str(dflst[dnum]['LINE'].values[0]),
                                 os.path.join(pdir,str(dflst[dnum]['LINE'].values[0])+'_Talwani_lineplot.pdf'))
        elif not np.isfinite(fag070).any():
            print 'No FAG070 present: skipping.'
        else:
            print 'Line over 1000 km: skipping.'
        print('\n' * 0)

    # ### Loop through segments
    # pdir = os.path.join('figs', str(dflst[dnum]['DATE'][0])[:10])
    # if not os.path.exists(pdir):
    #     os.makedirs(pdir)
    # for dnum, dname in enumerate(dflst, start=0):
    #     if dnum % 2 != 0:
    #         print dnum
    #         if make_plots:
    #             '''
    #             LINE
    #             '''
    #             oib_lineplot(dflst[dnum], str(dflst[dnum]['DATE'][0])[:10] + '_L' + str(dflst[dnum]['LINE'][0]),
    #                          os.path.join(pdir, str(dflst[dnum]['LINE'][0])+'_lineplot.png'))
    #             oib_mapplot_hilite(dflst[dnum]['LONG'], dflst[dnum]['LAT'], dflst[dnum]['FAG070'], df2, 'm',
    #                         'FAG070 ' + str(dflst[dnum]['DATE'][0])[:10] + '_L' + str(dflst[dnum]['LINE'][0]),
    #                         os.path.join(pdir, str(dflst[dnum]['LINE'][0])+'_mapplot_FAG070.png'))
    #

    '''
    Merge back together
    '''
    df2 = {}
    df2 = pd.concat(dflst)

    '''
    Map
    '''
    ### Entire flight
    # oib_mapplot(df2['LONG'].where((df2['D_gravmask'] != -1)), df2['LAT'].where((df2['D_gravmask'] != -1)),
    #             df2['FLTENVIRO'].where((df2['D_gravmask'] != -1)), 'm', 'FLTENVIRO '+str(df2['DATE'].values[0])[:10],
    #             os.path.join(pdir, str(df2['DATE'].values[0])[:10]+'_mapplot_talwani_FLTENVIRO_ALL.png'))
    oib_mapplot(df2['LONG'], df2['LAT'], df2['rmse'], 'm',
                'rmse '+str(df2['DATE'].values[0])[:10],
                os.path.join(pdir, str(df2['DATE'].values[0])+'_mapplot_Talwani_rmse_ALL.pdf'))


    '''
    Save to CSV
    '''
    df2.to_csv('data/forward/OIB_ANT_'+str(df2['DATE'].values[0])[:10]+'_forward.csv')

if __name__ == '__main__':
    basedir = 'data'
    datadir = 'agg2invert'
    suffix = '.csv'
    ### Run through each directory
    pattern = os.path.join(basedir, datadir, 'OIB_ANT_2009-10-16*' + suffix)
    # pattern = './data/NetCDF/10103/R10103_003*.nc'
    filenames = sorted(glob(pattern))  # , key=alphanum_key)
    filecounter = len(filenames)
    for fnum, filename in enumerate(filenames, start=0):
        print "Data file %i is %s" % (fnum, filename)
        # sys.exit(main(timedir))
        try:
            forward_oib(filename, True)
        except IOError:
            print 'IOError - Data Not Found'
        except AttributeError:
            print 'Attribute Error'
