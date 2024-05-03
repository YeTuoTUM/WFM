#!/usr/bin/env python
# coding: utf-8

# In[ ]:

###plot block
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shapefile
from shapely.geometry import shape
import shapely.geometry as sg
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
from matplotlib.colors import Colormap
from matplotlib import colormaps

def boxplot_byAttr(dataframe, Xdata,Ydata, filename):
    # Assuming you have your DataFrame month_df
    savepath=f'{Xdata}_{Ydata}_boxplot_{filename}.png'
    # Group the DataFrame by 'HRU'
    #grouped = month_df.groupby(groupname)

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(14, 7))    # Adjust the width and height as needed

    sns.boxplot(data=dataframe, linewidth=0.3, 
                showcaps=True, 
                boxprops={"facecolor": (.3, .5, .7, .5)},   
                medianprops={"color": "r", "linewidth": 0.3},
                fliersize=0.2, width=0.4, showfliers=False,x=Xdata, y=Ydata)

    # Set x-axis label
    plt.xlabel(Xdata, fontsize=12)

    # Set y-axis label
    plt.ylabel(Ydata, fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=8)
    # Set the title
    plt.title(f'Boxplots of {Ydata} by {Xdata}_{filename}', fontsize=12)

    # Show legend
    #plt.legend()

    # Save the figure to 300 DPI
    plt.savefig(savepath, dpi=300)

    # Show the plot
    plt.show()
    
    
def timeserie_plot(dataframe,Ydata,filename,luc_filter=None, monthly_filter=False):
    # Group the DataFrame by 'SUB' and 'HRU'
    grouped = dataframe.groupby(['SUB', 'HRU'])

    # Get the unique subbasin IDs
    unique_subbasins = dataframe['SUB'].unique()
    unique_luc = dataframe['LUC'].unique()
    num_luc = len(unique_luc)
    luc_color_palette={unique_luc[i]: sns.color_palette()[i] for i in range(num_luc)}
    line_styles = ['-', '--', '-.', ':', (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (7, 1, 1, 1))]  # You may need to adjust this based on the number of HRUs with the same LUC ID
        
    # Calculate number of rows and columns
    num_subplots = len(unique_subbasins)
    num_cols = 3
    num_rows = -(-num_subplots // num_cols)  # Round up division

    # Set up the figure and axes for subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(30, 7*num_rows), sharex=True)

    # Flatten axes array for easy iteration
    axes = axes.flatten()
    ymax=dataframe[Ydata].max()
    # Iterate over each unique subbasin
    for idx, subbasin_id in enumerate(unique_subbasins):
        # Filter the DataFrame to get data only for the current subbasin
        subbasin_data = pd.concat([grouped.get_group((sub_id, hru_id)) for (sub_id, hru_id) in grouped.groups.keys() if sub_id == subbasin_id])
        #print(subbasin_data)
        if luc_filter:
            subbasin_data = subbasin_data[subbasin_data['LUC'] == luc_filter]
        # Keep track of line styles for different HRUs with the same LUC ID
        hru_line_styles = {}

        # Plot lines for each HRU within the current subbasin on the corresponding subplot
        for hru_id, group in subbasin_data.groupby('HRU'):
            #print(hru_id)
            group=group.reset_index(drop=True)
            xinfor=group.index.values+1
            if monthly_filter:
                xinfor = np.where((group.index.values+1) % 12 == 0, 12, (group.index.values+1) % 12)
            #xinfor = np.where((group.index.values+1) % 12 == 0, 12, (group.index.values+1) % 12)
            #print(group)
            # Get the LUC ID for the current HRU
            luc_id = group['LUC'].iloc[0]  # Assuming 'LUC' ID is the same for all rows within an HRU
            #print(group)
            # Assign line color based on LUC name
            luc_color = luc_color_palette.get(luc_id, 'black')  # Default to black if LUC name not found in palette
            
            # Assign line style based on HRU within the same LUC ID
            if luc_id not in hru_line_styles:
                hru_line_styles[luc_id] = line_styles[0]
                line_styles.append(line_styles.pop(0))  # Rotate line styles for subsequent HRUs with the same LUC ID

            sns.lineplot(data=group, x=xinfor, y=Ydata, ax=axes[idx], label=f'HRU {hru_id}, {luc_id}', color=luc_color, linestyle=hru_line_styles[luc_id])
        
        # Set y-axis label for the subplot
        axes[idx].set_ylabel(Ydata, fontsize=12)

        # Set title for the subplot
        axes[idx].set_title(f'Time Series of {Ydata} for SUB {subbasin_id}', fontsize=14)
        axes[idx].legend(loc='upper center',bbox_to_anchor=(0.5, -0.05),ncol=5)
        axes[idx].set_ylim(0, ymax)
    
    
    # Remove any extra empty subplots
    #for i in range(num_subplots, num_rows * num_cols):
        #fig.delaxes(axes[i])

    # Set x-axis label for the last row of subplots
    for ax in axes[-num_cols:]:
        ax.set_xlabel('Date', fontsize=12)

    # Add legend to the last subplot
   # axes[-1].legend()

    # Adjust layout and spacing
    plt.tight_layout()

    # Save the figure to 300 DPI
    plt.savefig(f'{Ydata}_timeseries_SUB_LUC_{luc_filter}_Monthly_{monthly_filter}_{filename}.png', dpi=300)

    # Show the plot
    plt.show()

    
def boxplot_time_sub(dataframe,Ydata,filename,luc_filter=None):
    # Group the DataFrame by 'SUB' and 'HRU'
    grouped = dataframe.groupby(['SUB', 'HRU'])

    # Get the unique subbasin IDs
    unique_subbasins = dataframe['SUB'].unique()
    
    unique_luc = dataframe['LUC'].unique()
    num_luc = len(unique_luc)
    luc_color_palette={unique_luc[i]: sns.color_palette()[i] for i in range(num_luc)}
    # Calculate number of rows and columns
    num_subplots = len(unique_subbasins)
    
    num_cols = 3
    num_rows = -(-num_subplots // num_cols)  # Round up division

    # Set up the figure and axes for subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(30, 7*num_rows), sharex=True)
    
    # Flatten axes array for easy iteration
    axes = axes.flatten()
    ymax=dataframe[Ydata].max()

    # Iterate over each unique subbasin
    for idx, subbasin_id in enumerate(unique_subbasins):
        # Filter the DataFrame to get data only for the current subbasin
        subbasin_data = pd.concat([grouped.get_group((sub_id, hru_id)).reset_index(drop=True) for (sub_id, hru_id) in grouped.groups.keys() if sub_id == subbasin_id])
        #print(subbasin_data)
        if luc_filter:
            subbasin_data = subbasin_data[subbasin_data['LUC'] == luc_filter]
            
        xinfor = np.where((subbasin_data.index.values+1) % 12 == 0, 12, (subbasin_data.index.values+1) % 12)

        sns.boxplot(data=subbasin_data, x=xinfor, y=Ydata,hue='LUC', ax=axes[idx], palette=luc_color_palette)
        
        # Get unique x values
        unique_x = sorted(np.unique(xinfor))

# Add vertical lines to separate boxplots of each x
        for x_value in unique_x:
            axes[idx].axvline(x_value - 0.5, color='gray', linestyle='--', linewidth=0.5)
    
        # Set y-axis label for the subplot
        axes[idx].set_ylabel(Ydata, fontsize=12)

        # Set title for the subplot
        axes[idx].set_title(f'Boxplot of {Ydata} for SUB {subbasin_id}', fontsize=14)
        axes[idx].legend(loc='upper center',bbox_to_anchor=(0.5, -0.05),ncol=5)
        axes[idx].set_ylim(0, ymax)
    
    # Remove any extra empty subplots
    #for i in range(num_subplots, num_rows * num_cols):
        #fig.delaxes(axes[i])

    # Set x-axis label for the last row of subplots
    for ax in axes[-num_cols:]:
        ax.set_xlabel('Date', fontsize=12)
    
    # Add legend to the last subplot
   # axes[-1].legend()

    # Adjust layout and spacing
    plt.tight_layout()

    # Save the figure to 300 DPI
    plt.savefig(f'{Ydata}_boxplot_SUB_LUC_{luc_filter}_Monthly_{filename}.png', dpi=300)

    # Show the plot
    plt.show()


def boxplot_time_luc(dataframe,Ydata,filename,d_filter=None):
    # Group the DataFrame by 'SUB' and 'HRU'
    grouped = dataframe.groupby(['LUC', 'HRU'])

    # Get the unique subbasin IDs
    unique_subbasins = dataframe['SUB'].unique()
    unique_LUC =dataframe['LUC'].unique()
    
    color_palette={i+1: sns.color_palette("Paired")[i] for i in range(len(unique_subbasins))}
    # Calculate number of rows and columns
    num_subplots = len(unique_LUC)
    num_cols = 1
    num_rows = -(-num_subplots // num_cols)  # Round up division

    # Set up the figure and axes for subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(30, 7*num_rows), sharex=True)

    # Flatten axes array for easy iteration
    axes = axes.flatten()
    
    maxlist=[]
    # Iterate over each unique subbasin
    for idx, subbasin_id in enumerate(unique_LUC):
        # Filter the DataFrame to get data only for the current subbasin
        subbasin_data = pd.concat([grouped.get_group((sub_id, hru_id)).reset_index(drop=True) for (sub_id, hru_id) in grouped.groups.keys() if sub_id == subbasin_id])
        #print(subbasin_data)
        
        ymax=subbasin_data[Ydata].max()
        if d_filter:
            subbasin_data = subbasin_data[subbasin_data['SUB'] == d_filter]
        maxlist.append([subbasin_id,subbasin_data.groupby(['SUB'])[Ydata].max()]) ##return max infor   
        xinfor = np.where((subbasin_data.index.values+1) % 12 == 0, 12, (subbasin_data.index.values+1) % 12)

        sns.boxplot(data=subbasin_data, x=xinfor, y=Ydata,hue='SUB', ax=axes[idx], palette=color_palette)
        
        # Get unique x values
        unique_x = sorted(np.unique(xinfor))

# Add vertical lines to separate boxplots of each x
        for x_value in unique_x:
            axes[idx].axvline(x_value - 0.5, color='gray', linestyle='--', linewidth=0.5)
    
        # Set y-axis label for the subplot
        axes[idx].set_ylabel(Ydata, fontsize=12)

        # Set title for the subplot
        axes[idx].set_title(f'Boxplot of {Ydata} for LUC {subbasin_id}', fontsize=14)
        axes[idx].legend(loc='upper center',bbox_to_anchor=(0.5, -0.05),ncol=12)
        axes[idx].set_ylim(0, ymax)
    
    # Remove any extra empty subplots
    #for i in range(num_subplots, num_rows * num_cols):
        #fig.delaxes(axes[i])

    # Set x-axis label for the last row of subplots
    for ax in axes[-num_cols:]:
        ax.set_xlabel('Date', fontsize=12)

    # Add legend to the last subplot
   # axes[-1].legend()

    # Adjust layout and spacing
    plt.tight_layout()

    # Save the figure to 300 DPI
    plt.savefig(f'{Ydata}_boxplot_LUC_SUB_{d_filter}_Monthly_{filename}.png', dpi=300)

    # Show the plot
    plt.show()
    return maxlist

def plot_polygon(ax, poly, **kwargs):
    path = Path.make_compound_path(
           Path(np.asarray(poly.exterior.coords)[:, :2]),
           *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])
 
    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)
     
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection


def plot_graph(sub_file, user_title, value=None, print_labels=True, cbar_label=None, vmin=None, vmax=None, cmap_str='Spectral', shrink=0.5):
   
    
    polygons, centr, subids = get_polygons_data(sub_file)
    value= value.reindex(subids).values
   # value=value[subids].values
    #print(subids, value)
    centr_x = [p.x for p in centr]
    centr_y = [p.y for p in centr]
    centroids = np.column_stack((centr_x, centr_y))
    
    fig, ax = plt.subplots(figsize=(10,6))

    #if cmap_str is None:
        #cmap_reds = plt.cm.get_cmap('Reds')
        #cmap_blues = plt.cm.get_cmap('Blues_r')
        #cmap = mcolors.ListedColormap([cmap_blues(i) for i in np.linspace(0.0, 0.9, 128)] + [cmap_reds(i) for i in np.linspace(0.1, 1, 128)])
    if isinstance(cmap_str, Colormap):
        cmap = cmap_str
    else:
        cmap = colormaps.get_cmap(cmap_str)
    if vmin is None or vmax is None:
        vmin = min(value)
        vmax = max(value)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=shrink)
    cbar.set_label(cbar_label)
    cbar.ax.tick_params(length=1.5, width=0.5, pad=1.5)
    cbar.outline.set_linewidth(0.3)
    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(["{:.0f}".format(tick) for tick in ticks])
    for i, (polygon, subid) in enumerate(zip(polygons, subids)):
    #for i, polygon in enumerate(polygons):
        #print(i)
        x,y = polygon.exterior.xy
        ax.plot(x, y, linewidth=0.3, color='grey')

        color = cmap(norm(value[i])) 
        subid = f"SUB {subid:.0f}"
        valuelabel = f"{value[i]:.2f}"
    # Check if the polygon has interior boundaries
        if len(polygon.interiors) > 0:
            interior_coords = [interior.coords for interior in polygon.interiors]
            exterior_coords = polygon.exterior.coords
            new_poly = sg.Polygon(shell=exterior_coords, holes=interior_coords)
        else:
            exterior_coords = polygon.exterior.coords
            new_poly = sg.Polygon(shell=exterior_coords)
        plot_polygon(ax, new_poly, facecolor=color)
        if print_labels:
            #print(polygon.centroid.y)
            ax.text(polygon.centroid.x, polygon.centroid.y-800, subid, ha='center', va='center',fontstyle='italic')
        #if print_values:
            ax.text(polygon.centroid.x, polygon.centroid.y+200, valuelabel, ha='center', va='center',fontweight='bold')
        


    ax.set_aspect('equal')
    ax.set_axis_off()
    #ax.set_xlabel('Longitude')
    #ax.set_ylabel('Latitude')
    ax.set_title(user_title)



def get_polygons_data(file):
    shp_reader = shapefile.Reader(file)
    centroids = []
    polygons = []
    shapeID = []
    
    for shape_record in shp_reader.shapeRecords():
        #print(shape_record.record[0])
        attributes = shape_record.record
        shape_geometry = shape_record.shape
        
        if shape_geometry.shapeTypeName == 'POLYGON':
            polygon = shape(shape_geometry)
            centroid = polygon.centroid
            centroids.append(centroid)
            polygons.append(polygon)
            #print(attributes[0])
            shapeID.append(attributes[0])
            #rint(shapeID)
    return polygons, centroids,shapeID
