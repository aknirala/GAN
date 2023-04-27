#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Needs 10s of thousands of images: 128x128 is min size, else it gets blurred. dpi = 72
Now variations possible: 12x60 = 720
Radius and ring variance: 5 and 2 = 10 + 5 (5 chose 2) + (5 Chose 1); This is 6 chose 2
Perhaps 3 and 2
It's hard to make it equi-probable so map 1 to 6 to configurations.
#Thickness of each ring could be one of two values: 5 will change to 10; 10 will change to 40 = 50x
3 Choose 2 = 2*3 = 6x
Ticks width and length could be chaged: 4x
Add inner minute tick: 2x
Width and length of needle changed: 4x

= 6 x 4 x 2 x 4 = 196

If I choose 3 out of 5 positions
"""
#import matplotlib.pyplot as plt
from matplotlib import figure
#import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import matplotlib.patches as patches
import io
#from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))
#%config Completer.use_jedi = False


# In[2]:


def addTick(ax, theta, cX = 45, cY = 45, radius = 40, t_angle = 2,
            width=5, hatch = " ", color="black"):
    """
    This function simply add a tick,
    It does so by calculating two points and adding a line between them.
    t_angel will sweep theta-t_angel and theta + t_angel
    """
    #First convert theta to radians.
    t1 = np.pi*(theta - t_angle)/180
    t2 = np.pi*(theta + t_angle)/180
    
    #Now generate Nx2 i.e, 4x2 points for our polygon
    points = np.zeros((4, 2))
    
    for i in range(4):
        if i == 0:
            #inner left point
            r = radius - width
            t = t1
        elif i == 1:
            #inner right
            r = radius - width
            t = t2
        elif i == 2:
            #outer right
            r = radius
            t = t2
        else:
            #outer left
            r = radius
            t = t1
        points[i, 0] = cX + r * np.cos(t)
        points[i, 1] = cY + r * np.sin(t)
    #ax.add_patch(patches.Polygon(points, hatch = hatch)) #, color='black'
    if hatch == " ":
        ax.add_patch(patches.Polygon(points, color=color))
    else:
        ax.add_patch(patches.Polygon(points, hatch=hatch))


# In[3]:


def addNeedle(ax, theta, cX = 45, cY = 45, radius = 35, t_angle = 1,
              hatch=" ", color="black"):
    """
    This will add the needle at the desired theta, and length would be dictated by radius.
    It will add a very soimple square needle.
    It will be 5 times to one side.. so t_angle will be multiplied by 5 in opposite direction.
    """
    t_rad = t_angle*np.pi/180
    #Now generate Nx2 i.e, 4x2 points for our polygon
    points = np.zeros((4, 2))
    
    for i in range(4):
        if i == 0:
            #back left point
            r = -radius/5
            t = theta + t_rad*5
        elif i == 1:
            #back right
            r = -radius/5
            t = theta - t_rad*5
        elif i == 2:
            #front right
            r = radius
            t = theta + t_rad
        else:
            #front left
            r = radius
            t = theta - t_rad
        points[i, 0] = cX + r * np.cos(t)
        points[i, 1] = cY + r * np.sin(t)
    if hatch == " ":
        ax.add_patch(patches.Polygon(points, color=color))
    else:
        ax.add_patch(patches.Polygon(points, hatch=hatch))


# In[8]:


patterns = [" ", "/" , "\\" , "///", "////", "---", "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
def getClock(h=3, m = 10, simp=True, cX = 90, cY = 90, r1 = 62, r2 = 72, 
             lw1 = 3, lw2 = 3,
            tickA = 1, tickW = 4, iTickR = 40,
            hR = 40, hA = 2, mR = 60, mA = 1):
    """
    If simp is false then color etc are generated randomly!!!
    """
    global patterns
    fig = figure.Figure()
    ax = fig.subplots(1)
    fig_size = 2.5
    dpi = 72
    canv_width = int(fig_size*dpi)
    if r1 is not None and r1 != 0:
        radius = r1
    else:
        radius = r2
    fig.set_size_inches(fig_size, fig_size)
    if simp:
        c_canvas =np.ones((canv_width, canv_width, 3), dtype=float)
    else:
        c_canvas =np.random.random((canv_width, canv_width, 3))*0.1 + 0.3
        kp = int(np.random.random()*3)%3
        ch = [0, 1, 2]
        ch.remove(kp)
        c_canvas[:,:,ch] = 0
    ax.imshow(c_canvas)
    ax.set_aspect('equal')
    circ1 = None
    color = np.random.rand(3)
    if simp:
        if r1 is not None and r1 != 0:
            circ1 = Circle((cX, cY), r1, linewidth = lw1, fill = False,
                          color=color)
        color = color*(min(1.0, 0.5 + np.random.rand()/2))
        circ2 = Circle((cX, cY), r2, linewidth = lw2, fill = False,
                      color=color)
    else:
        if r1 is not None and r1 != 0:
            rgb = [ (np.random.random()*0.5 + 0.3) for _ in range(3)]
            rgb.append(1)
            circ1 = Circle((cX, cY), r1, linewidth = lw1, color=rgb)
        rgb = [ (np.random.random()*0.5 + 0.3) for _ in range(3)]
        rgb.append(1)
        circ2 = Circle((cX, cY), r2, linewidth = lw2, fill = False,  color=rgb)
    if circ1 is not None:
        ax.add_patch(circ1)
    ax.add_patch(circ2)
    
    htachIdx = 0
    color = 0.9*np.random.rand(3)
    if not simp:
        htachIdx = int(np.random.random()*len(patterns))%len(patterns)
    
    for i in range(12):
        addTick(ax, -30*i, cX = cX, cY = cY, t_angle = tickA if simp else (2+((i+2)%3)//2), 
                width=tickW, radius = radius, hatch=patterns[htachIdx],
               color = color)
    if iTickR is not None:
        color = 0.9*np.random.rand(3)
        for i in range(60):
            addTick(ax, 6*i, cX = cX, cY = cY, t_angle = 1, 
                width=2, radius = iTickR, hatch=patterns[htachIdx],
                   color = color)
    
    #Now add a time, assume that h and m will always be in correct range.
    h = h%12 #As we are in 12 hour mode
    hTheta = (h + m/60)/12*2*np.pi -np.pi/2
    mTheta = m/60*2*np.pi -np.pi/2
    color = 0.9*np.random.rand(3)
    addNeedle(ax, mTheta, cX = cX, cY = cY, radius = mR, 
              t_angle=mA if simp else 2, 
              hatch=patterns[htachIdx], color=color)
    addNeedle(ax, hTheta, cX = cX, cY = cY, radius = hR, 
              t_angle=hA if simp else 3.5,
              hatch=patterns[htachIdx], color=color)
    #print(type(ax.images[0]), type(ax), dir(ax.images[0]), type(ax.images[0].make_image), dir(ax))
    
    
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=dpi)#DPI)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         #newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
                         newshape=(int(fig_size*dpi), int(fig_size*dpi), -1))
    io_buf.close()
    #fig.close()
    #print(img_arr.shape)
    #plt.imshow(img_arr)
    #plt.show()
    
    mar = 26#8#0.125*64
    delta = 2
    ad_delta = 0
    #print("fig.dpi: ",fig.dpi)
    if fig.dpi != 72:
        ad_delta = 1
    end = int(fig_size*dpi) - mar#64+8 #256//s_down+y_mar
    mar += ad_delta
    end += ad_delta
    ret_img = img_arr[mar:end, mar+delta:end+delta][:,:,:3]
    #del(io_buf)
    #del(img_arr)
    #plt.clf()
    #del(plt)
    #del(ax)
    #plt.close('all')
    #print("Closing all")
    return ret_img


# In[9]:


def getCenter(mv_mar, randSeed = None):
    if mv_mar < 0: mv_mar = 0
    if randSeed is None:
        np.random.seed()
    mv_mar = np.random.randint(mv_mar)
    if randSeed is None:
        np.random.seed()
    if mv_mar > 10:
        if np.random.randint(mv_mar) > 5:
            mv_mar = 5
    if randSeed is None:
        np.random.seed()
    if np.random.randint(2) == 0:
        mv_mar = -1
    return 90 + mv_mar


# In[10]:


def getRandomClock(simp = True, randSeed = None,
                  clkSize=-1, h = -1, m = -1):
    """It will simply generate a random clock.
    Howvere its best to set the seed before this function.
    
    if randSeed is given then it is guranteed that if other 
        parmas are not varied than same clock would be produced.
    
    clkSize: Take 0, 1, 2 (larger values are clipped)
    This is outer ring which is r2, 0, 1, 2: 42-62-82
    
    r1 is smaller radius
    
    """
    org_r_state = None
    if randSeed is not None:
        if randSeed != -1:
            org_r_state = np.random.get_state()
            np.random.seed(randSeed)
    else:
        np.random.seed()
    if clkSize == -1: clkSize = np.random.randint(3)
    if h == -1: h = np.random.randint(12)
    else: np.random.randint(12)  #Hack to get random and not random clocks!!
    if randSeed is None:
        np.random.seed()
    if m == -1: m = np.random.randint(60)
    else: np.random.randint(60)
    #print("Updated code: h: ",h," and m:",m)
    #First r2 can vary from 82 to 52
    r2 = 80
    if clkSize == 1:
        r2 = 70
    elif clkSize < 1:
        r2 = 60
    if randSeed is None:
        np.random.seed()
    r = np.random.randint(18)
    r2 -= r#
    #Now r1 will be 7 to 14 smaller
    if randSeed is None:
        np.random.seed()
    r = np.random.randint(7) + 5
    r1 = r2-r
    #Should r1 (inner radius) be made 0? 
    if randSeed is None:
        np.random.seed()
    if(np.random.randint(3) == 0):
        r1 = 0
    #r1 = 0
    #Now based on r1, one may shift cX and cY
    cX = getCenter(82-r2, randSeed)
    cY = getCenter(82-r2, randSeed)
    #print(82-r2, cX, cY)
    if randSeed is None:
        np.random.seed()
    lw2 = np.random.randint(4 if r2>60 else 2) + 2
    if randSeed is None:
        np.random.seed()
    lw1 = np.random.randint(2) + 1
    
    #if r2 < 60
    
    mR = r2 if r1 == 0 else r1
    #Now lets change tickA (its width) and tickW (it's length)
    if randSeed is None:
        np.random.seed()
    tickA = np.random.randint(3) + 1
    tickW = np.random.randint(5) + 3
    if r1 > 40:
        tickW += (r1-40)//10
    #
    if randSeed is None:
        np.random.seed()
    iTickR = np.random.randint(20)
    if iTickR < 5:
        iTickR = None
    else:
        iTickR = mR - iTickR
    
    #Now width and length of needles...
    if randSeed is None:
        np.random.seed()
    mR = mR -5 -np.random.randint(5)
    hR = mR - 6
    hR -= np.random.randint(5)
    
    if randSeed is None:
        np.random.seed()
    mA = 1 + np.random.randint(2)
    hA = mA +1 + np.random.randint(2)
    #if hA > 4:
    #    hA = 4
    #h=3, m = 10, simp=True, cX = 90, cY = 90, r1 = 62, r2 = 72, 
    #         lw1 = 3, lw2 = 3,
    #        tickA = 1, tickW = 4, iTickR = 40,
    #        hR = 40, hA = 2, mR = 60, mA = 1
    clk, h_m = getClock(h=h, m=m, simp=simp, cX=cX, cY=cY, 
                    r1=r1, r2=r2, lw1=lw1, lw2=lw2,
                    tickA=tickA, tickW=tickW, iTickR=iTickR,
                    hR = hR, hA = hA, mR = mR, mA = mA
                   ), [h, m]
    if org_r_state is not None:
        np.random.set_state(org_r_state)
    return clk, h_m


# In[17]:
"""
from tqdm import tqdm
import pickle
opFolder = "../../../data/clock/"
for i in tqdm(range(1000000)):
    with open(opFolder+str(i)+'.pkl','wb') as f:
        pickle.dump(getRandomClock().transpose(2, 0, 1)/255, f)"""
#clk = getRandomClock()
#import matplotlib.pyplot as plt
#plt.imshow(clk)

