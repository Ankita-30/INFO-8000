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


############################################################################################################################    
##ICP followed by calculation of the Hausdorf distance
    
#sys.path.append(r"C:\Users\ar54482\AppData\Local\Programs\Python\Python36-32\Lib\site-packages\ICP" )
#
import ICP
import os

Hausdorf=[]
Filename=[]
#for filename in os.listdir(path):
icp = ICP.ICP( 
               binary_or_color = "binary",
               corners_or_edges = "edges",
               
               
               pixel_correspondence_dist_threshold = 4000,
               auto_select_model_and_data = 1,
               max_num_of_pixels_used_for_icp = 200,
               iterations = 20,
               model_image = r"NS/model_pre_crop.jpg", 
               data_image = r"NS/101_pre_crop.jpg",
               font_file = "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
            )

icp.extract_pixels_from_color_image("model")
icp.extract_pixels_from_color_image("data")



 
with open(r'C:\Users\ar54482\Desktop\WebAPIRH\Project\Data Acquisition\NS\datacoordinates.txt') as f:
     data_coords= json.load(f)

with open(r'C:\Users\ar54482\Desktop\WebAPIRH\Project\Data Acquisition\NS\modelcoordinates.txt') as f:
     model_coords= json.load(f)
     
with open(r'C:\Users\ar54482\Desktop\WebAPIRH\Project\Data Acquisition\NS\superimposedcoordinates.txt') as f:
     ICP_coords= json.load(f)
    
   
x2=[]
y2=[]

  
    

x2=[i[0] for i in ICP_coords]   
y2=[i[1] for i in ICP_coords]  

  
x1=[i[0] for i in model_coords]   
y1=[i[1] for i in model_coords]  
#
x=[i[0] for i in data_coords]   
y=[i[1] for i in data_coords] 


fig, ax = plt.subplots(figsize=(10,10))


ax.scatter(x, y, c='r') #red is imgdata
ax.set_facecolor('black')


plt.savefig(r"NS\red_data.jpg")
plt.show()

fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(x1, y1, c='g') #green is model 
ax.set_facecolor('black')


plt.savefig(r"NS\green_model.jpg")
plt.show()

fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(x2, y2, c='b') #blue is superimposed data
ax.set_facecolor('black')
plt.savefig(r"NS\blue_superimposed.jpg")
plt.show()

#GET THE SPLINE FIT
#Not working!!!
imgdata = Image.open(r"NS\red_data.jpg")
graymodel = imgdata.convert('L')   # 'L' stands for 'luminosity'

graymodel = np.asarray(graymodel)


coords1 = find_contours(graymodel,155)
    
if len(coords1)!=0:
    coords1 = coords1[1]
    
    x3=[i[0] for i in coords1]  
   
    y3=[i[1] for i in coords1] 
    
 


#GET THE SPLINE FIT
#Not working!!!
imgmodel = Image.open(r"NS\green_model.jpg")
graymodel = imgmodel.convert('L')   # 'L' stands for 'luminosity'

graymodel = np.asarray(graymodel)


coords1 = find_contours(graymodel,155)
    
if len(coords1)!=0:
    coords1 = coords1[1]
    
    x4=[i[0] for i in coords1]  
   
    y4=[i[1] for i in coords1] 
    


imgsimp = Image.open(r"NS\blue_superimposed.jpg")
graymodel = imgsimp.convert('L')   # 'L' stands for 'luminosity'

graymodel = np.asarray(graymodel)


coords1 = find_contours(graymodel,155)
    
if len(coords1)!=0:
    coords1 = coords1[1]
    
    x5=[i[0] for i in coords1]  
   
    y5=[i[1] for i in coords1] 
    


    
#Fit a b-spline to model shape
tck, u = splprep([x4,y4], s=10)
new_points = splev(u,tck) #model

#tck, u = splprep([xnew,ynew], s=50)
#new_points = splev(u, tck)

#Fit a b-spline to the superimposed data shape
tck, u = splprep([x5,y5],s=0)
new_points1 = splev(u, tck)

#fig, ax = plt.subplots()
#ax.plot(x3, y3)
#ax.plot(new_points[0], new_points[1], c='g')  #model

#fig, ax = plt.subplots()
#ax.plot(new_points1[0], new_points1[1], c='blue')  #superimposed
#ax.scatter(x, y, c='g')
#plt.show()    

#METHOD1 - HAUSDORFF DISTANCE
#Hausdorff Distance    
data_spline=[]

for i in range(0,len(new_points1[0])):
    x=new_points1[0][i]
    y=new_points1[1][i]
    data_spline.append((x,y))

model_spline=[]

for i in range(0,len(new_points[0])):
    x=new_points[0][i]
    y=new_points[1][i]
    model_spline.append((x,y)) 

u1=np.array(model_spline)
v1=np.array(data_spline)

   
h=directed_hausdorff(u1,v1)
Hausdorf.append(h[0])
print("Hausdorf")
print(Hausdorf)
print ("Len of Hausdorf")
print(len(Hausdorf))
Filename.append("101.jpg")
print(Filename)
data={'Hausdorf':Hausdorf,'ID':Filename}
NS=pd.DataFrame(data)

#METHOD2 - FRECHET DISTANCE
#Frechet Distance
s=u1.shape
v1.resize(s)
Frechet=[]
Filename1=[]
##METHOD1 - FRECHET'S DISTANCE
f=frdist(u1,v1)
print("Frechet's distance")
print(f)
Frechet.append(f)
print("Frechet")
print(Frechet)
print ("Len of Frechet")
print(len(Frechet))
Filename1.append("101.jpg")
print(Filename1)
NS['Frechet']=f
    
#Write out the dataframe to an excel file
NS.to_csv(r"101_HF.csv", index = False, header = True)
    
  
    
