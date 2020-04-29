#Nitrogen-stress

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:10:51 2020

@author: ar54482
"""
###############################################################################
#Perimeter+Area
import os
import cv2
from PIL import Image
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import find_contours
from skimage import io
from skimage.color import convert_colorspace
from scipy.interpolate import splev, splprep
import numpy as np
from skimage.morphology import skeletonize
from sklearn.linear_model import LinearRegression
from frechetdist import frdist
import math
import seaborn as sns
import statistics
from scipy.spatial import distance
from skimage import img_as_bool, io, color, morphology
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
import json

#N-Stress
#CentroidSize

#define distance
def distance(x1,y1,x2,y2):
     d = (x1 - x2)**2 + (y1-y2)**2 
     return d

CSN1=[] #list to store centroid size 
Filename=[] #store filenames in the list


            
img = Image.open(r"NS\101.jpg").convert('L')
#red, green, blue = img.split()
coords = find_contours(img,200)

if len(coords)!=0:
    coords = coords[0]
    

    x=[i[0] for i in coords]  
   
    y=[i[1] for i in coords] 
    
      #centroid
    xc=sum(x)/len(x)
    
    yc=sum(y)/len(y)
    
    #xnew=x-xc
    #ynew=y-yc #make centroid as (0,0)
    
   
    coords_n=[]
    
    for i in range(0,len(coords)):
        
        coords_n.append([x[i],y[i]])  #store the new coordinates in a list
        

#    N=[] #list to store count of each root hair
D=[] #list to store the distances


#centroid length(cl)
for (x,y) in coords_n:

    k=distance(x,y,xc,yc)
    D.append(k)


    
cs=(sum(D))**.5
CSN1.append(cs)
print("Area of the shape")
print(cs)
Filename.append('42.jpg')

#print(len(Filename))
#print(len(CSN1))

data={'ID': Filename, 'Area':CSN1}
NS=pd.DataFrame(data)

#Perimeter



LN=[] #list to store the perimeter

imgcolor = cv2.imread(r"NS\42.jpg")

height, width, channels = imgcolor.shape


gray = cv2.cvtColor(imgcolor, cv2.COLOR_BGR2GRAY)

ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

_, contours, hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


red = 0
## Draw the contour 
for c in contours:
    
    red = red + 10 
#fill the connected contours
    contours_img = cv2.drawContours(imgcolor, [c], -1, (255, 255, red),4)
 
#os.chdir(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\contours")
#cv2.imwrite(filename,contours_img)
#cv2.destroyAllWindows()
   
l=cv2.arcLength(contours[0],closed=True)
LN.append(l)  ######LN!!!
print("Perimeter of shape")
print(l)
NS['Perimeter']=LN

NS.to_csv(r"101_AP.csv", index = False, header = True)
#############################################################################################
#Pre-alignment for ICP

# Read the image you want the connected components extracted from
img = cv2.imread(r"NS\101.jpg")
#img = cv2.imread(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\C\ControlP_shapes\segmented\All shapes\28.jpg")
##img = cv2.imread(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Trial2\2.jpg',0)
ret,thresh_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

cv2.imwrite(r"NS\gray.jpg",thresh_img)

imgdata = Image.open(r"NS\gray.jpg")
   
#imgdata = Image.open(os.path.join(path,filename))
width,height=imgdata.size

coords=[] #store the white coordinates here
#count=0
pix=[]
boundary=[]

for i in range(width):
    for j in range(height):
        coordinate=i,j
        p=imgdata.getpixel((i,j))
        pix.append(p[0])
        

        if p[0]>200:      
            coords.append((i,j)) #
            p1=imgdata.getpixel((i,j-1))
            p2=imgdata.getpixel((i-1,j))
            p3=imgdata.getpixel((i,j+1))
            p4=imgdata.getpixel((i+1,j))

            if p1[0]==0 or p2[0]==0 or p3[0]==0 or p4[0]==0:
                boundary.append((i,j))  
     

x=[i[0] for i in coords]  
y=[i[1] for i in coords] 

xb=[i[0] for i in boundary]  
yb=[i[1] for i in boundary] 



#determine the centroid
xcd=sum(x)/len(x)
ycd=sum(y)/len(y)

##calculate the centroid size
 
coords_n=[]
D=[]   #store the distances of boundary points from the centroid
 
#define distance
def distance(x1,y1,x2,y2):
 d = (x1 - x2)**2 + (y1-y2)**2 
 return d
 
for i in range(0,len(coords)):
    coords_n.append([x[i],y[i]])
 
for (a,b) in coords_n:
    k=distance(a,b,xcd,ycd)
    D.append(k)
 
cs=(sum(D))**.5/len(D)


coords_d=[]
coords_b=[]
        
for i in range(0,len(coords)):
    
    coords_d.append(((x[i]-xcd)/cs,(y[i]-ycd)/cs))

for i in range(0,len(boundary)):
    
    coords_b.append(((xb[i]-xcd)/cs,(yb[i]-ycd)/cs))

x1=[i[0] for i in coords_d]   
y1=[i[1] for i in coords_d] 

x2=[i[0] for i in coords_b]   
y2=[i[1] for i in coords_b]



fig, ax = plt.subplots()
plt.scatter(x1, y1, marker='x',color="white")
ax.set_facecolor('black')
ax.set_ylim(-350,350)
ax.set_xlim(-350,350)
#plt.savefig(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Allfilledshapes\Trial\8a.jpg")

fig, ax = plt.subplots()



plt.scatter(x1, y1, marker='x',color="white")
#plt.scatter(x2, y2, marker='o',color="red")


ax.set_ylim(-350,350)
ax.set_xlim(-350,350)
ax.set_facecolor('black')
#ax.set_axis_off()

   # plt.savefig(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Allfilledshapes\Trial\temp.jpg")



"""
Linear regression model

"""
from sklearn.linear_model import LinearRegression

#convert coordinates into array
X=np.asarray(x1).reshape((-1,1))
Y=np.asarray(y1)

#Build a model for linear regression and fit your data to calculate the weights
model = LinearRegression().fit(X,Y)

r_sq=model.score(X,Y)

#print('coefficient of determination:', r_sq)

#print('intercept:', model.intercept_)

#print('slope:', model.coef_)

a = np.linspace(-350,350,50)
b=model.coef_*(a) + model.intercept_

I=math.degrees(math.atan(model.coef_))
Ir=math.atan(model.coef_)
print('Ir',Ir)
print('model.coef_',model.coef_)

#print(model.coef_)
if model.coef_>0:
    
#Decide on which side of the line the point is

    L=[]
    R=[]
    
    a1=-300
    a2=300
    
    b1=model.coef_*(a1) + model.intercept_
    b2=model.coef_*(a2) + model.intercept_
    
    
    plt.plot(a,b, '-r')
    
    for (a0,b0) in coords_b:
        v=((a0 - a1)*(b2 - b1)) - ((a2 - a1)*(b0 - b1))
        
        if v>0:
            R.append((a0,b0))
        if v<0:
            L.append((a0,b0))
    
    #check (0,0) is on which side
    (a0,b0)=(-100,100)
    v=((a0 - a1)*(b2 - b1)) - ((a2 - a1)*(b0 - b1))
  #  print(v)
    
  #  print(b1,b2)
    
#    if v>0:
#        print("Right")
#    
#    if v<0:
#        print("Left")
    
            
    #find distance of the point from line
    DL=[]   
    XL=[]
    xl=0     
    for (a0,b0) in L:
        dl=(abs((model.coef_*a0)-b0+model.intercept_))/math.sqrt(model.coef_**2+1)
        
        DL.append(dl[0])
        
        xl=xl+1
        XL.append(xl)
    
    
    
    DR=[]
    XR=[]
    xr=0 
    for (a0,b0) in R:
        dr=(abs((model.coef_*a0)-b0+model.intercept_))/math.sqrt(model.coef_**2+1)
        DR.append(dr[0])
        xr=xr+1
        XR.append(xr)
   
    Lm=statistics.mean(DL)
    
    Rm=statistics.mean(DR)
    
    if Lm>Rm:
        H="Left"
    if Rm>Lm:
        H="Right"
        




if model.coef_<0:
    
    L=[]
    R=[]
    
    a1=-200
    a2=200
    
    b1=model.coef_*(a1) + model.intercept_
    b2=model.coef_*(a2) + model.intercept_
    
    plt.plot(a,b, '-r')
    
    for (a0,b0) in coords_b:
        v=((a0 - a1)*(b2 - b1)) - ((a2 - a1)*(b0 - b1))
        
        if v>0:
            L.append((a0,b0))
        if v<0:
            R.append((a0,b0))
    
    #check (0,0) is on which side
    (a0,b0)=(150,150)
    v=((a0 - a1)*(b2 - b1)) - ((a2 - a1)*(b0 - b1))
    #print(v)
    #print(b1,b2)
    
#    if v>0:
#        print("Left")
#    
#    if v<0:
#        print("Right")
    
            
    #find distance of the point from line
    DL=[]   
    XL=[]
    xl=0     
    for (a0,b0) in L:
        dl=(abs((model.coef_*a0)-b0+model.intercept_))/math.sqrt(model.coef_**2+1)
       
        DL.append(dl[0])
        
        xl=xl+1
        XL.append(xl)
    
    
    
    DR=[]
    XR=[]
    xr=0 
    for (a0,b0) in R:
        dr=(abs((model.coef_*a0)-b0+model.intercept_))/math.sqrt(model.coef_**2+1)
        DR.append(dr[0])
        xr=xr+1
        XR.append(xr)
    
    Lm=statistics.mean(DL)
    
    Rm=statistics.mean(DR)
    
    if Lm>Rm:
        H="Left"
       # print("Left")
    if Rm>Lm:
        H="Right"
        #print("Right")
print(H)
        


if H =="Left":
    
         
    flip=[]    
    b=len(coords_d)    
    for (x3,y3) in coords_d:
        xp=width-x3-1
        flip.append((xp,y3))

    
    x3=[i[0] for i in flip]   
    y3=[i[1] for i in flip] 

    x3cd=sum(x3)/len(x3)
    y3cd=sum(y3)/len(y3) 
    
    for i in range(0,len(x3)):
    
        flip.append((x3[i]-x3cd,y3[i]-y3cd))
        
    x4=[i[0] for i in flip]   
    y4=[i[1] for i in flip]
    
    fig, ax = plt.subplots()

    plt.scatter(x4, y4, marker='x',color="white")    
    
    ax.set_ylim(-350,350)
    ax.set_xlim(-350,350)
    ax.set_facecolor('black')
   # ax.set_axis_off()
   #get the flipped slope value
    X4=np.asarray(x4).reshape((-1,1))
    Y4=np.asarray(y4)

    #Build a model for linear regression and fit your data to calculate the weights
    model4 = LinearRegression().fit(X4,Y4)

    r_sq=model.score(X4,Y4)

    #print('coefficient of determination:', r_sq)

    #print('intercept:', model.intercept_)

    #print('slope:', model.coef_)

    a4 = np.linspace(-350,350,50)
    b4=model4.coef_*(a4) + model4.intercept_

    I4=math.degrees(math.atan(model4.coef_))
    Ir4=math.atan(model4.coef_)
    print('Ir4',Ir4)

   
   
#Align its orientation to the mean shape (sd5_20x12L.png44.jpg in the Trial folder)
    
#model slope in degrees
S=math.degrees(math.atan(0.87791686))
Sr=math.atan(0.87791686)
print('S',S)
print('Sr',Sr)
#model slope in radians

#data slope
#D=abs(I)
#print('D',D)    

#For right hooked:
#multiply by rotation matrix with angle as the difference between S and D

coords_d1=[]


if H=="Right":
    
    
    if abs(Ir)>Sr:
        
        a=(Ir-Sr)
        r=[[math.cos(a), math.sin(a)],[(-math.sin(a)), math.cos(a)]]
        mr=np.matrix(r)
    
        
        for x in coords_d:
       
            xx=np.matrix(x)  #(D)
            
            mxt=xx.transpose()          #M=(R*D)
            
            rx=mr*(mxt)
            #convert trx into [x,y]
            rx=rx.tolist()
            #print (txnew)
            coords_d1.append(rx)
            
    
    #print(coords_d1[0])
          
    #print(coords_d1[0][0][0])    
    #print(coords_d1[0][1][0])     
        
        coords_d2=[]  #remove the lists in list
        
        
        for i in range(0,len(coords_d1)):
            
            coords_d2.append((coords_d1[i][0][0],coords_d1[i][1][0]))
        
        
        x5=[i[0] for i in coords_d2]   
        y5=[i[1] for i in coords_d2]
        
           
        fig, ax = plt.subplots()
        
        plt.scatter(x5, y5, marker='x',color="white")   
        ax.set_facecolor('black')
        
        ax.set_ylim(-350,350)
        ax.set_xlim(-350,350)
        #os.chdir(r'NS\42_pre.jpg')
        plt.savefig(r'NS\101_pre.jpg')
        

        
    if abs(Ir)<Sr:
        a=(Ir-Sr)
        r=[[math.cos(a),(-math.sin(a))],[math.sin(a), math.cos(a)]]
        mr=np.matrix(r)
        
        
            
        for x in coords_d:
       
            xx=np.matrix(x)  #(D)
            
            mxt=xx.transpose()          #M=(R*D)
            
            rx=mr*(mxt)
            #convert trx into [x,y]
            rx=rx.tolist()
            #print (txnew)
            coords_d1.append(rx)
            
            
            
        coords_d2=[]  #remove the lists in list
       
        for i in range(0,len(coords_d1)):
            coords_d2.append((coords_d1[i][0][0],coords_d1[i][1][0]))
            
    
            x5=[i[0] for i in coords_d2]   
            y5=[i[1] for i in coords_d2]
            
        
            
       
        fig, ax = plt.subplots()
    
        plt.scatter(x5, y5, marker='x',color="white")   
        ax.set_facecolor('black')
    
        ax.set_ylim(-350,350)
        ax.set_xlim(-350,350)
        #os.chdir(r"NS\42.jpg")
        plt.savefig(r'NS\101_pre.jpg')
        

    
if H=="Left":
    
    if abs(Ir4)>Sr:
        
        a=(Ir4-Sr)
        r=[[math.cos(a), (-math.sin(a))],[math.sin(a), math.cos(a)]]
        mr=np.matrix(r)
        
        for x in flip:
       
            xx=np.matrix(x)  #(D)
            
            mxt=xx.transpose()          #M=(R*D)
            
            rx=mr*(mxt)
            #convert trx into [x,y]
            rx=rx.tolist()
            #print (txnew)
            coords_d1.append(rx)
            
    
    #print(coords_d1[0])
          
    #print(coords_d1[0][0][0])    
    #print(coords_d1[0][1][0])     
        
        coords_d2=[]  #remove the lists in list
        
        
        for i in range(0,len(coords_d1)):
            
            coords_d2.append((coords_d1[i][0][0],coords_d1[i][1][0]))
        
        
        x5=[i[0] for i in coords_d2]   
        y5=[i[1] for i in coords_d2]
        
           
        fig, ax = plt.subplots()
        
        plt.scatter(x5, y5, marker='x',color="white")   
        ax.set_facecolor('black')
        
        ax.set_ylim(-350,350)
        ax.set_xlim(-350,350)
        #os.chdir(r'NS\42.jpg')
        #plt.savefig(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Pre-aligned\1.jpg')
        plt.savefig(r'NS\101_pre.jpg')
        
        
        
   
        


    if abs(Ir4)<Sr:
        
        a=2*(Sr-Ir4)
        
        r=[[math.cos(a),(math.sin(a))],[(-math.sin(a)), math.cos(a)]]
        mr=np.matrix(r)
        
          
                
        for x in flip:
            xx=np.matrix(x)  #(D)
                
            mxt=xx.transpose()          #M=(R*D)
                
            rx=mr*(mxt)
                #convert trx into [x,y]
            rx=rx.tolist()
                #print (txnew)
            coords_d1.append(rx)
                
                
                
        coords_d2=[]  #remove the lists in list
           
        for i in range(0,len(coords_d1)):
            coords_d2.append((coords_d1[i][0][0],coords_d1[i][1][0]))
                
        
        x5=[i[0] for i in coords_d2]   
        y5=[i[1] for i in coords_d2]
                
            
                
           
        fig, ax = plt.subplots()
    
        plt.scatter(x5, y5, marker='x',color="white")   
        ax.set_facecolor('black')
    
        ax.set_ylim(-350,350)
        ax.set_xlim(-350,350)
        #os.chdir(r'NS\42.jpg')
        #plt.savefig(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Pre-aligned\1.jpg')
        plt.savefig(r'NS\101_pre.jpg')
        
##########################################################################################################################          
##crop the pre-aligned figures
image = cv2.imread(r"NS\101_pre.jpg")

y=40
x=54
h=210
w=335
crop = image[y:y+h, x:x+w]
cv2.imshow('Image', crop)
cv2.waitKey(0)
cv2.imwrite(r"NS\101_pre_crop.jpg",crop)
############################################################################################################################    
