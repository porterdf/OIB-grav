import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from my_OIB_functions import *
import pandas as pd

def oib_lineplot_cust(data, ptitle='test_lineplot', pname='test_lineplot'):
    import matplotlib.pyplot as plt
    data.loc[data['HYDROAPPX'] < -1500, 'HYDROAPPX'] = np.nan
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), facecolor='white')
    data['FAG070'].where((data['FLTENVIRO'] == 1)).plot(ax=axes[0], legend=True, label='Disturbed', style='r-')
    data['FAG070'].where((data['FLTENVIRO'] == 0)).plot(ax=axes[0], legend=True, label='Normal', style='k-')
    # data['FAG070'].where((data['FLTENVIRO'] == -1)).plot(ax=axes[0], legend=True, label='Missing', style='b-')
#     data['ELEVATION'].plot(ax=axes[1], legend=True, style='y.')
    data['ICEBASE'].plot(ax=axes[1], legend=True, marker=".", linestyle="None", color="brown")
    data['TOPOGRAPHY_radar'].plot(ax=axes[1], legend=True, marker=".", linestyle="None", color="blue")
#     data['HYDROAPPX'].plot(ax=axes[1], legend=True, color='grey')
    data['SURFACE_atm'].where((data['NUMUSED'] > 77)).plot(ax=axes[1], legend=True,
                                                           marker=".", linestyle="None", color='cyan')
    plt.suptitle(ptitle, y=0.98)

    ### Mouse click
    global coords
    coords = []
    fig.canvas.callbacks.connect('button_press_event', on_click)
    print coords
    y = np.median(data['ICEBASE'].values[coords])
    # plt.plot(event.xdata, y, 'rs', ms=10, picker=5, label='cont_pnt')
    # Call click func
    # cid = fig.canvas.mpl_connect('button_press_event', onclick)
    # ch1 = np.where(x == (find_nearest(x, coords[0][0])))
    # ch2 = np.where(x == (find_nearest(x, coords[1][0])))
    # print ''
    # print 'Integral between ' + str(coords[0][0]) + ' & ' + str(coords[1][0])

    plt.show()
    # plt.savefig(pname, bbox_inches='tight')   # save the figure to file
    # plt.close(fig)
    return

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    # print 'x = %d, y = %d'%(
    #     ix, iy)
    # assign global variable to access outside of function
    global coords
    coords.append((ix, iy))
    # Disconnect after 2 clicks
    if len(coords) == 2:
        fig.canvas.mpl_disconnect(cid)
        plt.close(1)
    return

def on_click(event):
    if event.inaxes is not None:
        global ix, iy
        ix, iy = event.xdata, event.ydata
        coords.append((ix, iy))
        print ix, iy
    else:
        print 'Clicked ouside axes bounds but inside plot window'
    return coords



df = pd.read_csv('data/agg2invert/OIB_ANT_2010-05-19.csv')
gb = df.groupby(['LINE'])
segment = 1274277755
lf = gb.get_group(segment)
oib_lineplot_cust(lf)



