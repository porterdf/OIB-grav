import sys
import os
from glob import glob
import re
import time
import yaml

from my_OIB_functions import *


def make_oib_csv(base_dir, timedir, date_flight, config, print_diagnostics=False, make_plots=True, sample_grid_line=False):
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
    # gravdir = os.path.join(basedir, 'IGGRV1B')
    # outdir = os.path.join(basedir, 'integrated')
    base_dir = config['base_dir']
    grv_dir = config['grv_dir']
    rad_dir = config['rad_dir']
    atm_dir = config['atm_dir']
    prc_dir = config['prc_dir']
    out_dir_base = config['out_dir']
    out_dir = os.path.join(out_dir_base, prc_dir)
    grd_dir = config['grd_dir']

    sample_var = config['sample_var']

    '''
    Run functions to read in each data set
    '''
    ### Gravity
    grv = importOIBgrav(os.path.join(base_dir, grv_dir), timedir, date_flight)

    ### ATM
    if os.path.exists(os.path.join(base_dir, atm_dir, timedir)):
        if not os.path.exists(os.path.join(base_dir, atm_dir, timedir, 'ILATM2_' + date_flight + '_all.csv')):
            print('cat ATM data...meow')
            catATM(os.path.join(base_dir, atm_dir, timedir), date_flight)
        else:
            print('ATM data already catted.')
        atm = {}
        try:
            atm = importOIBatm(os.path.join(base_dir, atm_dir, timedir), date_flight)
        except AttributeError:
            print('No ATM data for this flight.')
    else:
        print('No SURFACE_atm data for this flight.')
        atm = pd.DataFrame(index=grv.index, columns=['SURFACE_atm', 'NUMUSED'])

    ### RADAR
    # if os.path.exists(os.path.join(base_dir, 'IRMCR2', timedir, '**', '*'+date_flight+'*.csv')):
    # if os.path.isdir(os.path.join(base_dir, 'IRMCR2', timedir, date_flight + '_*')):
    if np.shape(sorted(glob(os.path.join(base_dir, rad_dir, timedir, '*' + date_flight + '*.csv'))))[0] != 0:
        rad = {}
        try:
            print('\nRADAR dir: {}'.format(os.path.join(base_dir, rad_dir, timedir)))
            rad = importOIBrad_all(os.path.join(base_dir, rad_dir, timedir), date_flight)
            # rad = importOIBrad(base_dir, timedir, infile)
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
    # df = pd.concat([grv, rad2hz[['THICK', 'ELEVATION', 'FRAME', 'SURFACE_radar', 'BOTTOM', 'QUALITY']],
    #                 atm2hz[['SURFACE_atm', 'NUMUSED']]], axis=1, join_axes=[grv.index])
    df = pd.concat([grv, rad2hz[['THICK', 'ELEVATION', 'FRAME', 'SURFACE_radar', 'BOTTOM', 'QUALITY']],
                    atm2hz[['SURFACE_atm', 'NUMUSED']]], axis=1)  # .reindex(grv.columns)       
    # df['DAY'] = df.index.day
    # df['HOUR'] = df.index.hour
    df['ICEBASE'] = df['ELEVATION'] - df['BOTTOM']
    df['TOPOGRAPHY_radar'] = df['ELEVATION'] - df['SURFACE_radar']

    # # Subset
    # df_sub = df.query('(FRAME <= @subframe+2) & (FRAME >= @subframe-2)')

    # # using FX moving average
    # # window = 240 for plane maneuvers > 2 minutes
    # grv['FX_MA'] = grv['FX'].rolling(window=240, center=True).mean()
    # grv_sub = grv.query('(WGSHGT < 3000) & (FX_MA < 70000)')

    if sample_grid_line:
        import pickle

        print("\nReading in some grids to be sampled, LATER")
        # sample_var = ['free_air_anomaly', 'bouguer_anomaly', 'orthometric_height']
        # sample_var = ['ADMAP', 'FAA', 'RTOPO2_icemask', 'RTOPO2_bedrock']
        # grd_dir = '/Users/dporter/Documents/data_local/'

        # # ADMAP
        if any(substr in 'ADMAP' for substr in sample_var):
            # datadir = 'Antarctica/Geophysical/ADMAP/'
            suffix = '.llz'
            pattern = os.path.join(grd_dir, 'ADMAP_ORSTEDcombined' + suffix)
            # pattern = os.path.join(grd_dir, datadir, 'ant_new' + suffix)

            filenames = sorted(glob(pattern))  # , key=alphanum_key)
            print(filenames)
            print("Reading {}".format(filenames[0]))
            admap = pd.read_csv(filenames[0], delimiter=r"\s+", names=('lat', 'lon', 's'), header=None)

        # # Scheinert_2016
        if any(substr in 'FAA' for substr in sample_var):
            try:
                import xarray as xr
                # datadir = 'Antarctica/Geophysical/Scheinert_2016/'
                suffix = '.nc'
                pattern = os.path.join(grd_dir, 'antgg*' + suffix)
                filenames = sorted(glob(pattern))  # , key=alphanum_key)
                print("Reading {}".format(filenames[0]))
                antgg = xr.open_dataset(filenames[0])  
                # antgg = antgg.set_index(latdim='latitude')
                # antgg = antgg.set_index(londim='longitude')         
            except:    
                # TODO OR any of the other ANTGG fields we may want to sample
                antgg = pickle.load(open(os.path.join(grd_dir,'antgg2015_bouger.p'),'rb'))


        # # RTOPO-2
        if any(substr in 'RTOPO2_icemask' for substr in sample_var):
            import xarray as xr
            # datadir = 'Antarctica/DEM/RTOPO2'
            suffix = '.nc'
            # pattern = os.path.join(grd_dir, datadir, 'RTopo-2.0.1_1min_aux*' + suffix)
            pattern = os.path.join(grd_dir, 'RTopo-2.0.1_30sec_Antarctica_aux.nc')
            filenames = sorted(glob(pattern))  # , key=alphanum_key)
            print(filenames)
            rtopo2_aux = xr.open_dataset(filenames[0])
            rtopo2_aux = rtopo2_aux.set_index(latdim='lat')
            rtopo2_aux = rtopo2_aux.set_index(londim='lon')

        if any(substr in 'RTOPO2_bedrock' for substr in sample_var):
            import xarray as xr
            # datadir = 'Antarctica/DEM/RTOPO2'
            pattern = os.path.join(grd_dir, 'RTopo-2.0.1_30sec_Antarctica_data.nc')
            filenames = sorted(glob(pattern))  # , key=alphanum_key)
            print(filenames)
            rtopo2_dat = xr.open_dataset(filenames[0])
            rtopo2_dat = rtopo2_dat.set_index(latdim='lat')
            rtopo2_dat = rtopo2_dat.set_index(londim='lon')


    '''
    Split Dataframe using Gravity quality/presence
    '''
    df['D_gravmask'] = df['FLTENVIRO']
    df.loc[df['D_gravmask'] == 0, 'D_gravmask'] = 1
    dflst = {}
    dflst = [g for _, g in df.groupby((df.D_gravmask.diff() != 0).cumsum())]
    print("Split Dataframe using Gravity quality/presence")
    for dnum, dname in enumerate(dflst, start=0):
        if print_diagnostics:
            print(f"\nLine {str(int(dname['UNIX'][0]))}")
        # # Add LINE channel
        # dname.loc[:, 'LINE'] = str(dname['FLT'][0]) + '.' + str(abs(dname['LAT'][0] * 1e3))[:4]
        dname.loc[:, 'LINE'] = str(int(dname['UNIX'][0]))

        # # Find Sea level for HYDROAPPROX
        # TODO OR just get Geoid from start, mean, etc of survey area
        mode = min(np.mean(dname['SURFACE_atm'][:10]), np.mean(dname['SURFACE_atm'][10:]))
        if 'orthometric_height' in dname.columns:
            # print("Using Scheinert Orthometric Height for Sea level...")
            clevel = dname['orthometric_height'].values
        elif mode < 40:
            # print("Using mode of SURFACE_atm for Sea level...")
            clevel = mode
        else:
            clevel = 0
        if print_diagnostics:
            print('Mode of ATM is %.2f' % mode)
            print('Setting sea-level to %.2f' % clevel)
        # dname.loc[:, 'HYDROAPPX'] = (clevel - (dname['SURFACE_atm'] - clevel) * 7.759)  # or SURFACE_atm
        # # dname.loc[:, 'HYDROAPPX'] = clevel - ((dname['SURFACE_atm'] - clevel) * 7.759)  # or SURFACE_atm
        # # (clevel - (((dname['SURFACE_atm'] - clevel) * 7.759))
        # try:
        #     dname.loc[:, 'HYDROAPPX'] = 0 - (dname['orthometric_height'].values * 7.759)
        # except:
        #     dname.loc[:, 'HYDROAPPX'] = (clevel - (dname['SURFACE_atm'].values - clevel) * 7.759)

        '''
        Create Output Directories Now
        '''
        if make_plots:
            # individual lines
            pdir = os.path.join(out_dir, 'figs', str(dname['DATE'][0])[:10])
            if not os.path.exists(pdir):
                os.makedirs(pdir)

        '''
        SAMPLE
        '''
        if sample_grid_line:
            # for v, var in enumerate(sample_var):
                # dname[var] = np.nan

            # Determine sampling rate
            rtopo_dec = 14
            antgg_dec = rtopo_dec * 5
            admap_dec = rtopo_dec * 5
            survey_speed_kmh = 350
            time_res_hz = 2
            sample_res_km = (rtopo_dec / time_res_hz) * (survey_speed_kmh / 3600)
            print(f"RTOPO sample_res (35 km/hr): {sample_res_km:.2f} km")

            if any(substr in 'ADMAP' for substr in sample_var):
                # print("Sampling admap")
                # print(f"var is {var}")
                # print(f"sample_var[v] is {sample_var[v]}")

                start_sample = time.time()
                dname['ADMAP'] = np.nan
                for i in range(0, dname.shape[0], admap_dec):
                    iii = get_closest_cell_llz(admap, dname['LAT'][i], dname['LONG'][i])
                    dname['ADMAP'].iloc[i] = admap['s'].iloc[iii]

                end_sample = time.time()
                print(f"Sampling ADMAP for {str(dname['LINE'][0])} took {end_sample - start_sample:.2f} sec")

            if any(substr in 'FAA' for substr in sample_var):
                # print("Sampling antgg")
                # print(f"var is {var}")
                # print(f"sample_var[v] is {sample_var[v]}")
                
                start_sample = time.time()
                dname['FAA'] = np.nan
                for i in range(0, dname.shape[0], antgg_dec):
                    ii, jj = get_closest_cell_xr(antgg, dname['LAT'][i], dname['LONG'][i],
                                                lat_key='latitude', lon_key='longitude')
                    dname['FAA'].iloc[i] = antgg['free_air_anomaly'].values[ii, jj]

                end_sample = time.time()
                print(f"Sampling FAA for {str(dname['LINE'][0])} took {end_sample - start_sample:.2f} sec")
            

            if any(substr in 'RTOPO2_icemask' for substr in sample_var):
                start_sample = time.time()
                dname['RTOPO2_icemask'] = np.nan
                dname['RTOPO2_icemask'].iloc[::rtopo_dec] = rtopo2_aux.sel(latdim=dname['LAT'].values[::rtopo_dec],
                                                            londim=dname['LONG'].values[::rtopo_dec],
                                                            method='nearest')['amask'].values.diagonal()

                end_sample = time.time()
                print(f"Sampling RTOPO2_icemask for {str(dname['LINE'][0])} took {end_sample - start_sample:.2f} sec")

            if any(substr in 'RTOPO2_bedrock' for substr in sample_var):
                # print("Sampling rtopo")
                # print(f"var is {var}")
                # print(f"sample_var[v] is {sample_var[v]}")

                start_sample = time.time()
                dname['RTOPO2_bedrock'] = np.nan
                dname['RTOPO2_bedrock'].iloc[::rtopo_dec] = rtopo2_dat.sel(latdim=dname['LAT'].values[::rtopo_dec],
                                                            londim=dname['LONG'].values[::rtopo_dec],
                                                            method='nearest')['bedrock_topography'].values.diagonal()

                end_sample = time.time()
                print(f"Sampling RTOPO2_bedrock for {str(dname['LINE'][0])} took {end_sample - start_sample:.2f} sec")

            # # Fill NaNs
            start_sample = time.time()
            for v, var in enumerate(sample_var):
                try:
                    dname[var].interpolate(method='spline', order=1, s=0., axis=0, limit_area='inside', inplace=True)
                    # # options: method = 'spline', order = 1, limit_area='inside', limir=140
                except:
                    dname[var].interpolate(method='linear', limit=140+1, axis=0, inplace=True)
                # dname[var] = fill_nan(dname[var].values)

                # TODO grid noise still in looped samples - BSpline or other smoother?
            end_sample = time.time()
            print(f"Interpolating took {end_sample - start_sample:.2f} sec")

        '''
        ICEBASE and SURFACE Recalc
        '''
        dname['surface_recalc'] = np.nan
        # dfout['surface_recalc'] = dfout['SURFACE_atm']
        # dfout.loc[dfout['surface_recalc'].isnull(), 'surface_recalc'] = dfout['TOPOGRAPHY_radar']
        # EXAMPLE df['X'] = np.where(df['Y'] >= 50, 'yes', 'no')
        dname['surface_recalc'] = np.where(dname['SURFACE_atm'].isnull(), dname['TOPOGRAPHY_radar'], dname['SURFACE_atm'])
        # print(('(sfc_atm - recalc) = {}'.format((dname.SURFACE_atm - dname.surface_recalc).max())))
        # print(('(TOPO_radar - recalc) = {}'.format((dname.TOPOGRAPHY_radar - dname.surface_recalc).max())))

        # ICEBASE
        dname['icebase_recalc'] = dname['surface_recalc']
        dname.loc[dname['surface_recalc'] != np.nan, 'icebase_recalc'] = (dname['surface_recalc'] - dname['THICK'])
        dname.loc[dname['surface_recalc'] == dname['icebase_recalc'], 'icebase_recalc'] = (dname['icebase_recalc'] - 1)
        # dfout[['TOPOGRAPHY_radar', 'SURFACE_atm', 'surface_recalc', 'THICK', 'ICEBASE', 'icebase_recalc']].loc['2016-10-14T16:50:02':'2016-10-14T16:50:05']

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
            if dnum % 2 != 0:
                print('Plots for {}: '.format(str(dname['LINE'][0])))
                try:
                    oib_lineplot_all(dname,
                                    str(dname['DATE'][0])[:10] + '_L' + str(dname['LINE'][0]),
                                    os.path.join(pdir, str(dname['LINE'][0]) + '_lineplot.png'))
                except:
                    print("couldn't lineplot")
                try:
                    oib_mapplot_hilite(dname['LONG'], dname['LAT'], dname['FAG070'], dname, 'mGal',
                            'FAG070 ' + str(dname['DATE'][0])[:10] + '_L' + str(dname['LINE'][0]),
                            os.path.join(pdir, str(dname['LINE'][0])+'_mapplot_FAG070.png'))
                except:
                    print("couldn't mapplot")

        # need to update dataframe list with temporary 'dname'
        dflst[dnum] = dname

    '''
    Merge back together
    '''
    dfout = {}
    dfout = pd.concat(dflst, sort=True)

    # dfout = pd.concat(dflst)
    # dfout = pd.concat(dname)
    # df_sub = pd.DataFrame.from_dict(map(dict, dflst))

    # # Whole Flight Mapplot #
    if make_plots:
        # TODO plot other vars
        # TODO only plot survey location, perhaps based on AGL?
        oib_mapplot_flight(dfout['LONG'].where((dfout['D_gravmask'] != -1)), dfout['LAT'].where((dfout['D_gravmask'] != -1)),
                    dfout['FLTENVIRO'].where((dfout['D_gravmask'] != -1)), 'm',
                    'FLTENVIRO ' + str(dfout['DATE'][0])[:10],
                    os.path.join(os.path.join(out_dir, 'figs', str(dname['DATE'][0])[:10]),
                                str(dfout['DATE'][0])[:10] + '_mapplot_FLTENVIRO_ALL.png'))

    '''
    Save to CSV
    '''
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(out_dir, 'IPGG84_' + str(dfout['DATE'][0])[:10] + '.csv')
    print(out_file)
    dfout.to_csv(out_file)


if __name__ == '__main__':
    # Read YAML config file
    try:
        config_file = sys.argv[1]
        print(f"Getting processing parameters from {config_file}")
    except IndexError:
        proj_path = os.path.abspath(os.path.dirname(__file__))
        config_file = os.path.join(proj_path, 'config_runtime.yml')
        print(f"Reading default configuration file from {proj_path}")
    with open(config_file, 'r') as file:  #  TODO write a functino to get parameters from conig YAML all at once
        config = yaml.safe_load(file)
    base_dir = config['base_dir']
    grv_dir = config['grv_dir']
    flights = config['flight_list']
    print(f"Processing flights: {flights}")

    start_all = time.time()

    for dnum, dname in enumerate(flights, start=0):
        print('\ndnum: {}, dname: {}'.format(dnum, dname))
        # filetime = str(dname)[0:4] + str(dname)[5:7] + str(dname)[8:10]
        print('\ndname: {}'.format(dname))
        print('dirpath: {}'.format(os.path.join(base_dir, grv_dir)))
        print('filetime: {}'.format(re.sub('\.', '', dname)))
        # try:
        make_oib_csv(base_dir, dname, re.sub('\.', '', dname), config,
                         print_diagnostics=True, make_plots=True, sample_grid_line=True)
        # except IOError:
        #     print('IOError - Data Not Found')
        # except AttributeError:
        #     print('Attribute Error in __main__')

    end_all = time.time()
    print('Processing took {}'.format(end_all - start_all))
