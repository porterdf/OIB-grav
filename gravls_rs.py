#!/usr/bin/python3

## Copyright Hugh Pumphrey 2012/2016

##   This program is free software: you can redistribute it and/or modify
##   it under the terms of the GNU General Public License as published by
##   the Free Software Foundation, either version 3 of the License, or
##   (at your option) any later version.

##   This program is distributed in the hope that it will be useful,
##   but WITHOUT ANY WARRANTY; without even the implied warranty of
##   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##   GNU General Public License for more details.

##    You should have received a copy of the GNU General Public License
##    along with this program.  If not, see
##    <http://www.gnu.org/licenses/>.

## Gravity model for several simple line sources as per page xx fig xx
## of Lowrie. Each source has 3 parameters: depth Z, position x0 and
## radius R.  The density contrast, which we keep the same for all
## sources, is entered by the user: you can really only estimate the
## mass/unit length of the source.

## No version numbers, but changes recorded here with date.

## 23 Nov 2012: Removed some uninformative printout from stdout.
## Added printout of fitted model values

## Oct 2013: Version that also fits regional slope.

## March 2016: translation into python/Tk/numpy/matplotlib
## February 2019: Translation into Python 3


import numpy as np
import matplotlib.pyplot as plt
from tkinter import *


## Utility function to draw a circle. Just for the sake of demonstration
## we make this a global function rather than a method of our main Class.
def drawcircle(xc,yc,r,npts=30,**kwargs):
    """ Draw a circle, given the centre and radius """
    theta = np.linspace(0,2*np.pi,num=npts)
    px =xc + r*np.cos(theta)
    py = yc + r*np.sin(theta)
    plt.fill(px,py,**kwargs)



class Application(Frame):
    """Simple line-source gravity model application.

    Application is done as a single instance of a class. Look at the
    Tkinter docs on line for more details. 

    """
    def formod(self,xs,x,b,doks=False):
        ## Forward function for line sources.  The vector b is things
        ## that the FM needs but which are not to be estimated. In
        ## this case the density contrast is in b because we can not
        ## estimate both it and the radii. The measurement vector is
        ## calculated from the formula on page 92 of Lowrie. The K
        ## (aka G ) matrix is currently found by brute force; it would
        ## be better to do this by differentiating analytically. The
        ## argument doks controls whether we calculate K or not.
        
        ## Force the input model vector x to be an array rather than a matrix
        xs = np.array(xs)
        xs=xs.flatten()

        ## ns is the number of line sources
        ns = np.int(np.int((len(xs)-2)) / 3)
  
        ## Big G, the gravitational constant
        G = 6.67421e-5 ### SI * 1.0e6, so that y comes out in GU
        drho = b
        ## Break the model vector apart into its components
        x0 = xs[0:ns]
        Z = xs[(ns):(2*ns)]
        R = xs[(2*ns):(3*ns)]
        dg0 = xs[3*ns]
        rsl = xs[3*ns+1]
        
        yf = x * 0;

        ## Loop over the ns line sources adding on the contribution from each
        for isource in range(0,ns) :
            thisr = R[isource]
            thisz=Z[isource]
            thisx = x0[isource]
            thisdg = (2*np.pi*G*drho*thisr*np.abs(thisr) * thisz /
                      (thisz*thisz + (thisx-x)*(thisx-x) ) )
            
            yf =  yf + thisdg

  
  
            y =  dg0 + rsl * x + yf

        if(doks):
            k = np.zeros([len(y),len(xs)])
            delt = 0.1
            ## Loop over elements of model vector, calculating row of K for each
            for ipt in range(0,len(xs)):
                xpt = xs*1.0 ## Force copy
                xpt[ipt] =  xpt[ipt] + delt
                ypt = self.formod(xpt,x,b,doks=False)[0]
                k[:,ipt] = (ypt - y) / delt
            
        else :
            k = 0

        retlist= [np.matrix(y),np.matrix(k)]
        return retlist
    




    def nlls(self,xs,x,y,b):
        ## Function to fit a set of observations to the model. We use the
        ## Marquadt-levenberg approach with a simple doubling / halving of the
        ## ML parameter gamma depending on whether the cost function goes down
        ## or up. Just using the simpler inverse Hessian approach only works
        ## if the starting point is _very_ close to the solution.

        ## Force input data (y) and model vector (xs) to be Nx1, Px1 matrices
        y=np.matrix(y).T
        xs=np.matrix(xs).T
        
        xi = 1.0*xs
        oldcost = 1.0e31
        gamma = 1024.0
        crit= 1.0e30
        nx = len(xs)
        D = np.matrix(np.eye(nx))
        gamfac = 2
        convok = False
        ## Main loop. We do at most 200 iterations, breaking out before then
        ## if it converges
        for i in range(0,200):
            fmr = self.formod(xi,x,b,doks=True)
            yc = np.matrix(fmr[0]).T
            k = np.matrix(fmr[1])
            dy = y - yc
            oldcost = dy.T*dy
            oldcost=oldcost[0,0]
            ktk = k.T * k
            xhat=xs * np.NaN

            ## Calculating an inverse matrix is inside this "try" statement
            ## because it might go wrong. The "except" clause is what happens
            ## if it does go wrong.
            try:
                mli=(ktk + gamma * D).I
            except:
                print("Oh dear. Your problem is not well-posed.")
                xhat=xs * np.NaN
                return [xhat,oldcost,crit,gamma,convok]

            delx=mli * k.T * (y - yc)
            xhat = xi + delx
            dif = xhat -xi
            crit = (dif.T*dif)[0]
            print("iteration",i,"conv=",crit,"gamma=",gamma,"cost=",oldcost)

            if crit < 1.e-3 and gamma < 1.0/1024.0 :
                print("finished at iteration " + str(i))
                convok =True
                break
    
            fmr = self.formod(xhat,x,b,doks=False)
            ynew =  np.matrix(fmr[0]).T
            dy = y - ynew
            newcost =  (dy.T * dy)[0]

            ## Here is the Marquadt-Levenberg bit: If the cost
            ## function went up more than a little bit we do that step
            ## again with a larger gamma
            if  newcost < oldcost*1.1 : 
                xi = xhat
                gamma = gamma / gamfac
            else: 
                gamma = gamma * gamfac
    
  
        if not convok:
            print("Failed to converge at iteration",i)
        return [xhat,oldcost,crit,gamma,convok]

    


    def lscalc(self):
        """ Main function that is called when you press "Run" """

        ## Make sure that we have a window. But don't destroy and re-make
        ## if we do have one.
        plt.figure(1)
        plt.clf()
        plt.ion()
        plt.subplot(2, 1, 1)


        ## This is where we  get stuff from the Tk entry boxes
        Z = eval("np.array(["+ self.Zin.get() + "])" )
        drho = eval("np.array(["+ self.drhoin.get() + "])" )
        drho=drho[0]
        x0 = eval("np.array(["+ self.x0in.get() + "])" )
        R = eval("np.array(["+ self.Rin.get() + "])" )
        offset = eval("np.array(["+ self.offsetin.get() + "])" )
        offset = np.array([offset[0]])
        regslope = eval("np.array(["+ self.slopein.get() + "])" )
        regslope = np.array([regslope[0]])
        ## Note we made offset and regslope arrays of length 1 rather than
        ## scalars as we need to concatenate them with some other arrays
  
        ## Guard against the user entering incompatible lengths for Z,x0
        nf = np.min([len(Z), len(x0),len(R)])
        Z = Z[0:nf]
        x0 = x0[0:nf]
        R = R[0:nf]
        ## Control whether we fit the data. This is a relic for
        ## testing purposes only.
        dofit = True
        
        ## These are not model parameters, but they need to come from
        ## somewhere.  they are the limits of the range to be
        ## considered. 
        x1 = eval("np.array(["+ self.x1in.get() + "])" )
        x2 = eval("np.array(["+ self.x2in.get() + "])" )
        x1=x1[0]
        x2=x2[0]
        
        ## Extract text from the main entry box
        qux=self.dat.get("1.0","end")
        ## this gives you a single character string for the whole lot
        ### Now we split it into lines and keep only lines with 2 numbers
        qlines=qux.splitlines()
        maxnl=len(qlines)
        lc=0
        xdat=np.zeros(maxnl)
        gdat=np.zeros(maxnl)
        for iline in range(0,len(qlines)):
            thisline=qlines[iline].split()
            if len(thisline) != 2:
                print ("Dud line", qlines[iline])
            else :
                ##print thisline[0],"---", thisline[1]
                xdat[lc] = np.float(thisline[0])
                gdat[lc] = np.float(thisline[1])
                lc=lc+1
        xdat=xdat[0:lc]
        gdat=gdat[0:lc]
                
  
                

        npp = 201 ## This is just for plotting purposes
        x = np.linspace(x1,x2,num=npp)
        
        xs = np.concatenate((x0,Z,R,offset,regslope))
        dg = self.formod(xs,x,drho)[0]
        dg=np.array(dg).flatten()
        plt.plot(x,dg,'k-')
        plt.xlabel("Distance / m")
        plt.ylabel("Gravity anomaly / GU")

        if len(xdat) > 1:
            plt.plot(xdat,gdat,marker='s',color="blue",label="Data",
                     linestyle="none",markersize=8)
            grc=self.formod(xs,xdat,drho)[0]
            grc=np.array(grc).flatten()
            plt.plot(xdat,grc,color="black",marker="o",linestyle="none",
                     label="entered")
            
        ## fit the model parameters to the data
         
        if dofit and len(gdat) > 3 :
            [xhat,oldcost,crit,gamma,convok]= self.nlls(xs,xdat,gdat,drho)
            xhat = np.array(xhat).flatten()
            if np.isfinite(xhat[0]) :
                if convok:
                    fitcol ="#00ee00"
                    fitlab="fitted (Conv OK)"
                else:
                    fitcol = "red"
                    fitlab="fitted (Conv fail)"

                    
                    
                    ## Fit succeeded. We plot it.
                grc=self.formod(xhat,xdat,drho)[0]
                grc=np.array(grc).flatten()
                plt.plot(xdat,grc,marker='o',color=fitcol,label=fitlab)
                    
                    
                    
            else:
                print("bad xhat")
                
            plt.legend(loc="upper left")
        else:
            ## No data, so estimated model is bad by definition
            xhat=np.array([np.NaN])

        ##print "testing xhat=",xhat
        if np.isfinite(xhat[0]) :
            x0r =  xhat[0:nf]
            Zr= xhat[(nf):(2*nf)]
            Rr= xhat[(2*nf):(3*nf)]
            bot = -(  np.max([Z,Zr]) + np.max([R,Rr])  )

            print ("-----------------------") 
            print ("Original horizontal positions") 
            print (xs[0:nf]) 
            print ("Original depths") 
            print (xs[(nf):(2*nf)] )
            print ("Original radii (-ve ==> -ve density contrast)") 
            print (xs[(2*nf):(3*nf)] )
            print ("Original grav offset =",xs[3*nf],"GU")
            print ("Original reg slope =",xs[3*nf+1],"GU / m")
            
            print ("Fitted horizontal positions") 
            print (xhat[0:nf])
            print ("Fitted depths")
            print (xhat[(nf):(2*nf)] )
            print ("Fitted radii (-ve ==> -ve density contrast)") 
            print (xhat[(2*nf):(3*nf)])
            print ("Fitted grav offset =",xhat[3*nf],"GU")
            print ("Fitted reg slope =",xhat[3*nf+1],"GU / m") 
            
        else:
            print ("Not printing results as xhat bad")
            bot = -(np.max(Z)+np.max(R))
        
        gbot=bot*5

        plt.subplot(2, 1, 2)

        ## Draw gray rectangle to represent ground
        plt.fill([x1,x1,x2,x2,x1],[gbot,0,0,gbot,gbot],color="#dddddd")
        plt.xlabel("Distance / m")
        plt.ylabel("Depth / m")

        for i in range(0,nf):
            ## Add entered source
            if R[i] < 0 :
                fcol = "skyblue"
            else: fcol = "pink"
            drawcircle(x0[i],-Z[i],abs(R[i]),color=fcol,edgecolor="black")

        ## Add estimated source if we have one
        if np.isfinite(xhat[0]) :
            print("Adding found sources")
            for i in range(0,nf) :
                if Rr[i] < 0 :
                    fcol = "blue"
                    print(str(i)+": This one is blue")
                else :
                    fcol = "red"
                    print(str(i)+": This one is red")
                print("Circle params: ",x0r[i],-Zr[i],np.abs(Rr[i]))
                drawcircle(x0r[i],-Zr[i],np.abs(Rr[i]),
                           facecolor="#ffffff00",edgecolor=fcol,hatch="//")
        ## Force axes to be sensible
        plt.axis("equal")
        plt.axis()
        plt.axis([x1,x2,bot,0])
        ## On windows, we seem to need the next line. Without it the
        ## plot window never appears, even though we have done
        ## plt.ion(). The line appears to be harmless but unnecessary
        ## on Linux.
        plt.show()

### --- end of function lscalc that does all the interesting stuff ---



    def set_defaults(self):
        ## Re-set entry boxes to default values
        self.Zin.delete(0,"end")
        self.Zin.insert(0,"200,600")
        self.drhoin.delete(0,"end")
        self.drhoin.insert(0,"400")
        self.offsetin.delete(0,"end")
        self.offsetin.insert(0,"0")
        self.slopein.delete(0,"end")
        self.slopein.insert(0,"0")
        self.x0in.delete(0,"end")
        self.x0in.insert(0,"-900,1100")
        self.Rin.delete(0,"end")
        self.Rin.insert(0,"-70,200")



    def createWidgets(self):
        ## This does all the stuff you need to have buttons
        ## text boxes etc. in the main window

        ## This is how to do some frames to organise the stuff into
        self.datf = Frame(self,relief="groove",borderwidth=3)
        self.contf = Frame(self,relief="groove",borderwidth=3)
        self.datf.pack({"side": "right"})
        self.contf.pack({"side": "left"})

        ## These four lines (plus an extra to set the text colour) are
        ## what you need in order to add a button to the app. This is
        ## the Quit button
        self.QUIT = Button(self.contf)
        self.QUIT["text"] = "QUIT"
        self.QUIT["fg"]   = "red"
        self.QUIT["command"] =  self.quit
        self.QUIT.pack({"side": "top"})

        ## These four lines are what you need in order to add a
        ## button to the app. This is the Run button
        self.run = Button(self.contf,fg="yellow",bg="black")
        self.run["text"] = "Run",
        self.run["command"] = self.lscalc
        self.run.pack({"side": "top"})

        ## The set-to-defaults button
        self.defaults = Button(self.contf)
        self.defaults["text"] = "Defaults",
        self.defaults["command"] = self.set_defaults
        self.defaults.pack({"side": "top"})

        ## This is how you add a single-line text entry box.
        ## The lscalc function extracts a value from it.
        ## The label has to be made separately
        self.Zlab = Label(self.contf,text="Line source depths")
        self.Zlab.pack({"side": "top"})
        self.Zin = Entry(self.contf)
        self.Zin.pack({"side": "top"})

        self.drholab = Label(self.contf,text="Density contrast")
        self.drholab.pack({"side": "top"})
        self.drhoin = Entry(self.contf)
        self.drhoin.pack({"side": "top"})

        self.offsetlab = Label(self.contf,text="Gravity anomaly offset")
        self.offsetlab.pack({"side": "top"})
        self.offsetin = Entry(self.contf)
        self.offsetin.pack({"side": "top"})

        self.slopelab = Label(self.contf,text="Regional Slope")
        self.slopelab.pack({"side": "top"})
        self.slopein = Entry(self.contf)
        self.slopein.pack({"side": "top"})
        
        self.x0lab = Label(self.contf,text="Line source horiz. position")
        self.x0lab.pack({"side": "top"})
        self.x0in = Entry(self.contf)
        self.x0in.pack({"side": "top"})
        
        self.Rlab = Label(self.contf,text="Line source radii")
        self.Rlab.pack({"side": "top"})
        self.Rin = Entry(self.contf)
        self.Rin.pack({"side": "top"})

        self.x1lab = Label(self.contf,text="Left-hand plot limit")
        self.x1lab.pack({"side": "top"})
        self.x1in = Entry(self.contf)
        self.x1in.pack({"side": "top"})

        ### This is how we clear the entry box and enter a starting value
        ## The set to defaults button leaves this alone.
        self.x1in.delete(0,"end")
        self.x1in.insert(0,"-2000")

        self.x2lab = Label(self.contf,text="Right-hand plot limit")
        self.x2lab.pack({"side": "top"})
        self.x2in = Entry(self.contf)
        self.x2in.pack({"side": "top"})

        ### This is how we clear the entry box and enter a starting value
        self.x2in.delete(0,"end")
        self.x2in.insert(0,"3000")

        ### Most of the default values are set at the start by this
        ### function. The user can call it again py pressing the
        ### defaults button.
        self.set_defaults()
        ### Multi-line Text box for entry data
        self.dat = Text(self.datf,width=24)
        self.dat.pack({"side": "right"})
        self.dat.delete(1.0,END)
        self.dat.insert(END,"""-2000 5.3513 
								-1750 5.1155 
								-1500 4.4533 
								-1250 2.4417 
								-1000 -2.3527 
								-750 3.0104 
								-500 5.0561 
								-250 7.2824 
								0 7.8927 
								250 8.9243 
								500 10.829 
								750 14.023 
								1000 19.679 
								1250 29.225 
								1500 34.941 
								1750 28.818 
								2000 20.006 
								2250 14.576 
								2500 11.041 
								2750 9.2719 
								3000 7.4395""")


    def __init__(self, master=None):
        ## This is the magic function that sets up the 
        ## main window and puts the buttons and entry boxes into it.
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

## These few lines are the main program. See the online docs for
## Tkinter for more details.
root = Tk()
app = Application(master=root)
root.wm_title("Simple line-source gravity model")
app.mainloop()
root.destroy()