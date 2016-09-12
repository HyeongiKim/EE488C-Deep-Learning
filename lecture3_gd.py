import matplotlib.pyplot as plt
import numpy as np

r=5.
# function to minimize: r*(x+y)**2+(x-y)**2

xmin=-30.
xmax=30.
ymin=-30.
ymax=30.
fmax=r*(xmax+ymax)**2

plt.figure(1)

# contours
m=100
t=np.arange(0., 1.+1./m, 1./m)
for f in np.arange(fmax/20, fmax, fmax/20):
    dt=1.*f/m
    a=np.sqrt(f)*np.cos(2*np.pi*t)
    b=np.sqrt(f)*np.sin(2*np.pi*t)
    plt.plot((a/np.sqrt(r)+b)/2,(a/np.sqrt(r)-b)/2,'b-')

# show trajectory of gradient descent
n=100
x=np.zeros(n)
y=np.zeros(n)
x[0]=-15
y[0]=25
e=0.08
H=np.matrix([[2*(r+1),2*(r-1)],[2*(r-1),2*(r+1)]])
for k in range(1,n):
    dx=r*2*(x[k-1]+y[k-1])+2*(x[k-1]-y[k-1])
    dy=r*2*(x[k-1]+y[k-1])-2*(x[k-1]-y[k-1])
    d=np.sqrt(dx**2+dy**2)
    # uncomment the following for optimal learning rate
    #e=d**2/(np.matrix([[dx,dy]])*H*np.matrix([[dx],[dy]]))
    x[k],y[k]=x[k-1]-e*dx,y[k-1]-e*dy
    # use the following for Newton's method
    #x[k],y[k]=np.matrix([[x[k-1]],[y[k-1]]])-H**(-1)*np.matrix([[dx],[dy]])
plt.plot(x,y,'or-')

# show normalized gradient vectors
ngd=10
xs=(xmax-xmin)/ngd
ys=(ymax-ymin)/ngd
for xi in np.arange(xmin,xmax+xs,xs):
    for yi in np.arange(ymin,ymax+ys,ys):
        dx=r*2*(xi+yi)+2*(xi-yi)
        dy=r*2*(xi+yi)-2*(xi-yi)
        d=np.sqrt(dx**2+dy**2)
        # uncomment the following for plotting normalized gradient vectors
        #if d!=0:
            #plt.plot([xi-dx/d*xs/3,xi+dx/d*xs/3],[yi-dy/d*ys/3,yi+dy/d*ys/3],'k-')
            #plt.plot(xi,yi,'k.')

plt.grid(True)
plt.axis([xmin,xmax,ymin,ymax])
plt.axes().set_aspect('equal')
# uncomment the following to save the figure
#plt.savefig('lecture3_gd.png',dpi=300,bbox_inches='tight')
plt.show()

