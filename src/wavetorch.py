# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:54:37 2023

@author: adamq
"""

#Numeric/Computational libraries
import numpy as np
from scipy import signal
import pandas as pd
import time
import torch
import torchaudio

#Write Libraries
from pathlib import Path

#Visualization libraries
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors

#Step function
def step_2d(u, dt, dx, dy, c, u_prev1=0, boundary="None"):
    """
    Calculate the value of wavefunction u at time t+dt, given u at time t

    Args:
        u: 2d array; values of u(x,y) at different values of x and y
        u_prev1: 2d array; values of u(x,y) at different values of x and y; 1 timestep ago;
                Will return an error if not specfied and boundary arg is None
        boundary: String indicating if the time step is calculated at a boundary
        dt: float; time step value
        dx: float; x step value
        dy: float; y step value
        c: float; wave equation constant
    
    Returns:
        u_new: 2d array; values of u(x,y) at next time step
    """
    
    #Given no boundary condition constraints
    if boundary == "None":
        
        #Unsqueeze tensors
        u=u.unsqueeze(dim=2)
        u_prev1 = u_prev1.unsqueeze(dim=2)
        
        #Ensure shape of u is equal to shape of u_prev
        u_new = torch.zeros(size=u.shape)
        
        #Iterate through each x and y coordinate of u to calculate next u step
        #Add u_prev1
        u_new[1:u.shape[0]-1,1:u.shape[1]-1] = u_new[1:u.shape[0]-1,1:u.shape[1]-1].clone().index_add_(dim=0,
            index=torch.arange(0,u.shape[0]-2),
            source=u_prev1[1:u.shape[0]-1, 1:u.shape[1]-1],
            alpha=-1)
                                                                                               
        #Add u multiplied by a factor
        u_new[1:u.shape[0]-1,1:u.shape[1]-1] = u_new[1:u.shape[0]-1,1:u.shape[1]-1].clone().index_add_(dim=0,
                    index=torch.arange(0,u.shape[0]-2),
                    source=u[1:u.shape[0]-1, 1:u.shape[1]-1],
                    alpha=2*(1-(dt**2)*(c**2)/(dx**2)-(dt**2)*(c**2)/(dy**2)))

        #Add u(x-dx) multiplied by dt**2 * c**2 divided by dx**2
        u_new[1:u.shape[0]-1] = u_new[1:u.shape[0]-1].clone().index_add_(dim=0,
                         index=torch.arange(0,u.shape[0]-2),
                         source=u[0:u.shape[0]-2],
                                 alpha = (dt**2)*(c**2)/(dx**2))

        #Add u(x+dx) multiplied by dt**2 * c**2 divided by dx**2
        u_new[1:u.shape[0]-1] = u_new[1:u.shape[0]-1].clone().index_add_(dim=0,
                         index=torch.arange(0,u.shape[0]-2),
                         source=u[2:u.shape[0]],
                                alpha = (dt**2)*(c**2)/(dx**2))

        #Add u(y-dx) multiplied by dt**2 * c**2 divided by dy**2
        u_new[:,1:u.shape[1]-1] = u_new[:,1:u.shape[1]-1].clone().index_add_(dim=1,
                         index=torch.arange(0,u.shape[1]-2),
                         source=u[:,0:u.shape[1]-2],
                                 alpha = (dt**2)*(c**2)/(dy**2))

        #Add u(y+dx) multiplied by dt**2 * c**2 divided by dy**2
        u_new[:,1:u.shape[1]-1] = u_new[:,1:u.shape[1]-1].clone().index_add_(dim=1,
                         index=torch.arange(0,u.shape[1]-2),
                         source=u[:,2:u.shape[1]],
                                 alpha = (dt**2)*(c**2)/(dy**2))
        
        #Calculate u at boundaries x=0, x=L_x, y=0, and y=L_y
        u_new[0:1, :] = 0
        u_new[u.shape[0] - 1:u.shape[0], :] = 0
        u_new[:,0:1] = 0
        u_new[:, u.shape[1] - 1:u.shape[1]] = 0
        #return u at next time step
        return u_new.squeeze(dim=2)
    
    #Given t=0 such that the previous time step is unknown
    elif boundary == "t0":
        
        #Unsqueeze tensors
        u=u.unsqueeze(dim=2)
        
        #Ensure shape of u is equal to shape of u_prev
        u_new = torch.zeros(size=u.shape)
        
        #Add u multiplied by a factor
        u_new[1:u.shape[0]-1,1:u.shape[1]-1] = u_new[1:u.shape[0]-1,1:u.shape[1]-1].clone().index_add_(dim=0,
                    index=torch.arange(0,u.shape[0]-2),
                    source=u[1:u.shape[0]-1, 1:u.shape[1]-1],
                    alpha=1 + (dt**2)*(c**2)/(dx**2) + (dt**2)*(c**2)/(dy**2))

        #Add u(x-dx) multiplied by dt**2 * c**2 divided by dx**2
        u_new[1:u.shape[0]-1] = u_new[1:u.shape[0]-1].clone().index_add_(dim=0,
                         index=torch.arange(0,u.shape[0]-2),
                         source=u[0:u.shape[0]-2],
                                 alpha = 0.5*(dt**2)*(c**2)/(dx**2))

        #Add u(x+dx) multiplied by dt**2 * c**2 divided by dx**2
        u_new[1:u.shape[0]-1] = u_new[1:u.shape[0]-1].clone().index_add_(dim=0,
                         index=torch.arange(0,u.shape[0]-2),
                         source=u[2:u.shape[0]],
                                alpha = 0.5*(dt**2)*(c**2)/(dx**2))

        #Add u(y-dx) multiplied by dt**2 * c**2 divided by dy**2
        u_new[:,1:u.shape[1]-1] = u_new[:,1:u.shape[1]-1].clone().index_add_(dim=1,
                         index=torch.arange(0,u.shape[1]-2),
                         source=u[:,0:u.shape[1]-2],
                                 alpha = 0.5*(dt**2)*(c**2)/(dy**2))

        #Add u(y+dx) multiplied by dt**2 * c**2 divided by dy**2
        u_new[:,1:u.shape[1]-1] = u_new[:,1:u.shape[1]-1].clone().index_add_(dim=1,
                         index=torch.arange(0,u.shape[1]-2),
                         source=u[:,2:u.shape[1]],
                                 alpha = 0.5*(dt**2)*(c**2)/(dy**2))
                
        #Calculate u at boundaries x=0, x=L_x, y=0, and y=L_y
        u_new[0:1, :] = 0
        u_new[u.shape[0] - 1:u.shape[0], :] = 0
        u_new[:,0:1] = 0
        u_new[:, u.shape[1] - 1:u.shape[1]] = 0
        
        #return u at next time step
        return u_new.squeeze(dim=2)
    
#Wave equation function
def wave_eq(u_t0, g_r, wave_meta):
    """
    Calculate wavefunction over time given the initial wave state and wave source functions

    Args:
        u_t0: 2d array; values of u(x,y) at initial time t
        g_r: list containing functions paired with coordinates
                each element is a dictionary of coordinates and functions with keys 'coordinate' and 'function'
        wave_meta: metadata consisting of:
            dx -- x step value
            dy -- y step value
            dt -- time step value
            c -- wavefunction value
            N_t -- count of time steps

    Returns:
        dictionary:
            u_tensor -- torch tensor containing values of u over time
            u_shape -- shape of u_tensor
            wave_meta -- dictionary containing wave simulation metadata
    """
    
    #Get dimensions of u_t0 for storage
    u_shape = u_t0.shape
    
    #Get first values of u at t=0
    u_current = u_t0.clone()
    #Enforce g_r functions at t=0
    for g in g_r:
        #get function at coordinate r = (x,y)
        g_function = g['function']
        #get value of u at r = (x,y) given function
        u_current[g['coordinate'][1]][g['coordinate'][0]] = g_function(0)
    #Initiate output data structure for u (3d array)
    u_tensor = u_current.unsqueeze(dim=0)
    
    #Calculate u at 2nd time step t=dt
    u_next = step_2d(u=u_t0, dt=wave_meta['dt'], dx=wave_meta['dx'], dy=wave_meta['dy'],
                   c=wave_meta['c'], boundary='t0')
    #Enforce g_r functions at t=dt
    for g in g_r:
        #get function at coordinate r = (x,y)
        g_function = g['function']
        #get value of u at r = (x,y) given function
        u_next[g['coordinate'][1]][g['coordinate'][0]] = g_function(wave_meta['dt'])
        
    #Append u at t=dt
    u_tensor = torch.cat((u_tensor.clone(), u_next.unsqueeze(dim=0)))
    #Store values of u at t and t-dt
    u_prev1 = u_current.clone()
    u_current = u_next.clone()
    
    #Time step to append new u values at values of t+1 using for loop
    for n in range(0, wave_meta['N_t']):
        print('Calculating time step {} out of {}'.format(str(n), wave_meta['N_t']))
        #Get next value of u_next
        u_next = step_2d(u=u_current, u_prev1=u_prev1, dt=wave_meta['dt'], dx=wave_meta['dx'], dy=wave_meta['dy'],
                   c=wave_meta['c'], boundary='None')
        #Enforce g_r functions at t=n*dt
        for g in g_r:
            #get function at coordinate r = (x,y)
            g_function = g['function']
            #get value of u at r = (x,y) given function
            u_next[g['coordinate'][1]][g['coordinate'][0]] = g_function(n*wave_meta['dt'])
        #Append u at t
        u_tensor = torch.cat((u_tensor.clone(), u_next.unsqueeze(dim=0)))
        #Store values of u at t and t-dt
        u_prev1 = u_current.clone()
        u_current = u_next.clone()
        
    #Create final dictionary of values and metadata
    output = {'u':u_tensor, 'u shape': u_shape,'metadata': wave_meta}
    #return output
    return output

def get_r_space(N_x, N_y, dx, dy):
    """
    Generate a 2d grid of x and y coordinate values

    Args:
        N_x: int, width of grid, ie shape[0]
        N_y: int, height of grid, ie shape[0]
        dx: float, step size of the x dimension
        dy: float, step size of the y dimension

    Returns:
        r_space: torch tensor
    
    """
    #Generate 2d grid containing x and y coordinates
    xs = torch.linspace(0, (N_x-1)*dx, steps=N_x)
    ys = torch.linspace(0, (N_y-1)*dy, steps=N_y)
    
    #Create meshgrid to calculate wave function
    x_space, y_space = torch.meshgrid(xs, ys, indexing='xy')
    r_space = x_space + y_space

    #Return meshgrid
    return r_space

def generate_labels(u_tensor, source_coordinates, loc_coordinates):
    """
    Generate a pandas dataframe containing flags determining the vicinity of a signal to the different wave sources.
    
    Args:
        u_tensor (torch tensor): 3 dimensional tensor containing 2d wave amplitudes over time
        source_coordinates (list of length 2 tuples): list of coordinates of wave sources
        loc_coordinates (list of length 2 tuples): list of coordinates to be labeled

    Returns:
        df_labels: DataFrame, 
    """
    #define columns using wave source coordinates
    coordinate_columns = ['distance_from_' + str(x) for x in source_coordinates]

    #initialize output dataframe
    df_output = pd.DataFrame(columns = coordinate_columns)

    #populate dataframe with coordinate values
    for loc_coordinate in loc_coordinates:
        
        #Initialize row's data to append
        data_append = {}

        #Add euclidian distances as row's values
        for source_coordinate in source_coordinates:
            #compute euclidian distance
            distance = (abs(source_coordinate[0]-loc_coordinate[0])**2 +  abs(source_coordinate[1]-loc_coordinate[1])**2)**(1/2)
            #add wave source distance to data row dictionary
            data_append['distance_from_' + str(source_coordinate)] = distance

        #Append data row to dataframe
        df_output = pd.concat([df_output, pd.DataFrame(data_append, index=[(loc_coordinate[0], loc_coordinate[1])])])

    #return output dataframe
    return df_output