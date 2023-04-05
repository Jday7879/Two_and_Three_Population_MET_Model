# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 17:58:40 2020

@author: Jordan
"""

import numpy as np 
import matplotlib.pyplot as plt

def set_axis_titles(title,xlab,ylab,x_maximum = 0,y_maximum = 0,y_rotation = 90):
    if y_maximum != 0:
        plt.ylim([0,y_maximum])  
        
    if x_maximum != 0:
        plt.xlim([0,x_maximum])  

    plt.xticks(fontsize = 34)
    plt.yticks(fontsize = 34)
    plt.title(title,fontsize = 40)#30)
    plt.xlabel(xlab,fontsize = 40)
    plt.ylabel(ylab,fontsize = 40,rotation = y_rotation)


def save_figure(File_name):
    import os
    from pathlib import Path
    figure_path = Path(os.getcwd(),"Fig")
    plt.savefig(figure_path / File_name)
    
def new_fig(size = (10,10)):
    plt.figure(figsize = size)

def plot_time_series(xdata,ydata,sty='b-',title = '',xlab = '',ylab= '',linelabel = 'Please name data',new_fig = False):
    if new_fig == True:
        plt.figure(figsize = (10,12))
    try:
        for pos in np.arange(int(ydata.shape[0])): 
            plt.plot(xdata,ydata[pos],sty[pos],linewidth = 3,label = linelabel[pos])
    except:
        plt.plot(xdata,ydata,sty,linewidth = 3,label = linelabel)
        
        
    set_axis_titles(title, xlab, ylab)
    plt.legend(fontsize = 18)
    plt.tight_layout()

def plot_positional_data(xdata,combined_data,biofilm_locations,side = 'All',sty='-',title = '',xlab = '',ylab= '',linelabel = 'Please name data',new_fig = False,legend_on = True):
    if new_fig == True:
        plt.figure(figsize = (10,12))
    if side == 'Left' or side == 'left':
        biofilm_side = int(2)
    elif side == 'Right' or side == 'right':
        biofilm_side = int(-2)
    else:
        biofilm_side = int(xdata.size/2)
        
    if combined_data.size != np.sum(biofilm_locations):
        # If combined data is not in the correct vector format (is full array and not vector values) 
        # data is reshaped to be correct
        combined_data = combined_data[biofilm_locations == 1] 

    positional_layout = np.zeros(biofilm_locations.shape)
    positional_layout[biofilm_locations == 1] = combined_data
    y_positions = np.arange(biofilm_locations.shape[1])[biofilm_locations[biofilm_side,:]==1]
    for position in np.arange(y_positions.size):
        plotting_data = positional_layout[:,y_positions[position]] 
        plotting_data[plotting_data == 0] = np.nan # Marking zero data as zero to clean up plotting
        plt.plot(xdata,plotting_data,sty,label = position+1,linewidth = 3)#position+1 linelabel
        
        
    set_axis_titles(title, xlab, ylab, xdata.max())
    if legend_on == True:
        plt.legend(fontsize = 16)
    #plt.tight_layout()


def plot_contour(xmesh,ymesh,contour_data,number_levels = 25,title = '',xlab = 'x',ylab= 'y',new_fig = False,ux_data = 0,uy_data = 0,scatter_positions = 0,scatter_data = 0,stream_density = 3):
    if new_fig == True:
        plt.figure(figsize = (11,14))
        
    levels = np.zeros(number_levels+1)
    for k in np.arange(number_levels+1):
        levels[k] = k*(contour_data.max()/number_levels)
    
    plt.contourf(xmesh,ymesh,contour_data,levels,cmap = 'coolwarm')
    cb = plt.colorbar(fraction=0.063, pad=0.04)
    cb.ax.tick_params(labelsize=34)
    cb.ax.set_ylabel('mg/L',fontsize = 34) #rotation=270
    
    if type(ux_data) == type(contour_data) and type(uy_data) == type(contour_data):
        ux_i = 0.5*(ux_data[0:ux_data.shape[0]-1,:] + ux_data[1:ux_data.shape[0],:])
        uy_i = 0.5*(uy_data[:,0:uy_data.shape[1]-1] + uy_data[:,1:uy_data.shape[1]])
        speed = np.sqrt((ux_i.T)**2 +(uy_i.T)**2 )
        lw = 10*speed / speed.max()            
        start_points = np.array((xmesh[10,:],ymesh[1,:]))
        plt.streamplot(xmesh.T, ymesh.T, ux_i.T, uy_i.T,color = speed,cmap='Greys', density=stream_density,linewidth=1.5,arrowstyle = '-|>')
    else:
        print('Supply both Ux and Uy data to overlay streamlines')
        
    if type(scatter_positions) == type(contour_data) and type(scatter_data) == type(contour_data):
        plt.scatter(xmesh[scatter_positions == 1], ymesh[scatter_positions ==1],c = scatter_data/100,s = scatter_data/20,cmap ='YlGn' ) #'Greens_r'YlGn
    else:
        print('Supply both scatter positions and data to overlay scatter density')

    set_axis_titles(title, xlab, ylab, xmesh.max(), ymesh.max(), y_rotation=0)
    plt.axis('scaled')
    plt.xlim([0, xmesh.max()])
    plt.ylim([0, ymesh.max()])
    plt.xticks(np.arange(0.1, xmesh.max(), 0.1))
    plt.yticks(np.arange(0, ymesh.max(), 0.1))
    plt.tight_layout()
    # set_axis_titles(title, xlab, ylab, xmesh.max(), ymesh.max(),y_rotation = 0)
    # plt.tight_layout()
    
def plot_streamlines(xmesh,ymesh,ux_data,uy_data,title = '',xlab = 'x',ylab= 'y',new_fig = False,stream_density = 3,cmapping = 'Greys'):
    if new_fig == True:
        plt.figure(figsize = (11,14))#figsize = (10,10*ymesh.max()/xmesh.max())) #figsize = (10,12)
    
    ux_i = 0.5*(ux_data[0:ux_data.shape[0]-1,:] + ux_data[1:ux_data.shape[0],:])
    uy_i = 0.5*(uy_data[:,0:uy_data.shape[1]-1] + uy_data[:,1:uy_data.shape[1]])
    speed = np.sqrt((ux_i.T)**2 +(uy_i.T)**2 )
    lw = 10*speed / speed.max()            
    plt.streamplot(xmesh.T, ymesh.T, ux_i.T, uy_i.T,color = speed, density=stream_density,linewidth=1.5,arrowstyle = '-|>')   #,cmap=cmapping
    cb = plt.colorbar(fraction=0.062, pad=0.04)
    cb.ax.tick_params(labelsize=28)
    #cb.ax.set_yticklabels(['$0.5 \cdot 10^{-3}$','$1.0 \cdot 10^{-3}$','$1.5 \cdot 10^{-3}$','$2.0 \cdot 10^{-3}$','$2.5 \cdot 10^{-3}$'])#,'$1.2 \cdot 10^{-3}$'])
    cb.ax.set_ylabel('m/s', rotation=270,fontsize = 28)
    set_axis_titles(title, xlab, ylab, xmesh.max(), ymesh.max(),y_rotation = 0)
    plt.axis('scaled')
    plt.xlim([0,xmesh.max()])
    plt.ylim([0,ymesh.max()])
    plt.xticks(np.arange(0.1, xmesh.max(), 0.1))
    plt.yticks(np.arange(0, ymesh.max(), 0.1))
    plt.tight_layout()#h_pad=1)
    
    
