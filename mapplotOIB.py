def mapplotOIB (x,y,z,title,units):
    import matplotlib.pyplot as plt
    #########################################
    ### FULL FAG070
    import matplotlib.cm as cm
    from mpl_toolkits.basemap import Basemap
    fig = plt.figure(figsize=(8,8))
    #create basemap
    m = Basemap(width=700000,height=1100000,
                resolution='h',projection='stere',area_thresh = 100000.0,
                lat_ts=-55,lat_0=-68,lon_0=-65.)
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
    x1,y1 = m(x.values,y.values)
    # Plot them using round markers of size 6
    #m.plot(x, y, 'ro', markersize=2)
    m.scatter(x1,y1,c=z,marker="o",cmap=cm.jet,s=40, edgecolors='none',vmin=-100,vmax=200)#,vmin=-150,vmax=50
    c = plt.colorbar(orientation='vertical', shrink = 0.5)
    c.set_label(units)
    #plt.show()
    #plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.savefig(infile+'_'+title+'_mapplot.png',bbox_inches='tight')   # save the figure to file
    #plt.close(m)  
    return