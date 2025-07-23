import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import geopandas as gpd
import pymannkendall as mk
from functools import reduce
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.colors as mcolors
import geopandas as gpd
# Set default font to Arial and size to 18
#plt.rcParams.update({'font.size': 18, 'font.family': 'Arial', 'font.weight': 'bold'})

# Analysis functions


def avg(df, column_needed, column_name = "100years AI average",sort_col="DATE", start_year= 1924):
    
    df_copy= df.copy()
    dict_trend ={}
    list_trend=[]
    df_copy = df_copy[df_copy["S_year"]<= start_year]
    df_copy = df_copy[df_copy["YEAR"]>= start_year]
    df_copy.dropna(subset= column_needed)
    df_copy_avg = df_copy.groupby(["ID","Lat","Lon"]).agg(column_temporary=(column_needed,"mean")).reset_index()
    df_copy_avg.rename(columns={"column_temporary":column_name}, inplace=True)
    return df_copy_avg

def fulltrend_mmk(df, column_needed, column_name = "100years_spei12_",sort_col="DATE", start_year= 1924):
    import pymannkendall as mk
    df_copy= df.copy()
    dict_trend ={}
    list_trend=[]
    df_copy = df_copy[df_copy["S_year"]<= start_year]
    df_copy = df_copy[df_copy["YEAR"]>= start_year]
    df_copy.dropna(subset= column_needed)
    for ids in df_copy.ID.unique():
        df_each = df_copy[df_copy["ID"]==ids]
        df_each.sort_values(by=sort_col, inplace=True)
        lat =df_each.Lat.unique()[0]
        lon = df_each.Lon.unique()[0]
        trend_series = df_each[column_needed].values
        result= mk.hamed_rao_modification_test(trend_series)
        sen_result = mk.sens_slope(trend_series)
        dict_trend = {"ID": ids, "Lat":lat, "Lon":lon, f"{column_name}trend" :result.trend,f"{column_name}p" :result.p,f"{column_name}slope":sen_result.slope}
        list_trend.append(dict_trend)
    df_trend = pd.DataFrame(list_trend)
    return df_trend
   
def map_plot_numerical(ax, gdf_shape, df3, font=20, column="Value_Column", title="", vmin=None, vmax=None, return_scatter=False):
    """
    Plots a numerical map on a given axis with optional color range while ensuring boundary visibility.

    Parameters:
        ax (AxesSubplot): The axis to plot on.
        gdf_shape (GeoDataFrame): Background shapefile data.
        df3 (DataFrame): Data to plot.
        font (int): Font size for titles.
        column (str): Column to use for plotting.
        title (str): Title of the map.
        vmin (float): Minimum value for color scale.
        vmax (float): Maximum value for color scale.
        return_scatter (bool): Whether to return the scatter object.

    Returns:
        scatter (PathCollection): Scatter object for the plot (if return_scatter=True).
    """
    df_map = df3.copy()

    # Plot the shapefile twice:
    # 1. As a background
    gdf_shape.plot(ax=ax, color="white", edgecolor="black", linewidth=1, zorder=1)

    # 2. Scatter plot
    scatter = ax.scatter(
        df_map.Lon, 
        df_map.Lat, 
        c=df_map[column], 
        cmap='rainbow', 
        s=5, 
        alpha=0.7, 
        vmin=vmin, 
        vmax=vmax, 
        zorder=3  # Ensure scatter plot is above background
    )

    # 3. Redraw the boundaries on top for visibility
    gdf_shape.plot(ax=ax, color="none", edgecolor="black", linewidth=1.2, zorder=4)

    ax.set_title(title, fontsize=font, loc="left")
    ax.set_axis_off()

    if return_scatter:
        return scatter
    return None





def plot_all_maps_numerical_grid(gdf_shape, df_list, column_names, titles, font=20, cbar_label="Values", save_as=None):
    """
    Plots a grid of numerical maps with a single shared colorbar.

    Parameters:
        gdf_shape (GeoDataFrame): Background shapefile data.
        df_list (list of DataFrame): List of DataFrames to plot.
        column_names (list of str): Columns to use for each plot.
        titles (list of str): Titles for each plot.
        font (int): Font size for plot titles.
        cbar_label (str): Label for the colorbar.
        save_as (str, optional): Filepath to save the figure.

    Returns:
        None
    """
    # Determine global vmin and vmax for the color scale
    all_values = []
    for df, column in zip(df_list, column_names):
        all_values.extend(df[column].dropna().values)
    vmin, vmax = min(all_values), max(all_values)

    plt.rcParams.update({"font.size": font, "font.family": "Arial", "font.weight": "regular"})
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    scatter = None

    for i, df in enumerate(df_list):
        if i == len(df_list) - 1:
            scatter = map_plot_numerical(
                axes[i], gdf_shape, df, column=column_names[i], title=titles[i], vmin=vmin, vmax=vmax, return_scatter=True
            )
        else:
            map_plot_numerical(
                axes[i], gdf_shape, df, column=column_names[i], title=titles[i], vmin=vmin, vmax=vmax
            )

    if scatter is not None:
        fig.subplots_adjust(bottom=0.15)
        cax = fig.add_axes([0.2, 0.05, 0.6, 0.02])
        cbar = plt.colorbar(scatter, cax=cax, orientation='horizontal')
        cbar.set_label(cbar_label)

    # Save the figure if a filepath is provided
    if save_as:
        save_as_updated = fr"C:\Users\adeba\OneDrive\Documents\Datascience\Hydrology\Jupyter_lab\US_weather\Figures\{save_as}"
        plt.savefig(save_as_updated, dpi=500, bbox_inches="tight")
        print(f"The picture has been saved as {save_as}")

    plt.show()

    
def map_plot_categorical(ax, gdf_shape, df3, font=20, column="Category_Column", cmap=["green", "grey", "red"], title="", return_scatter=False):
    df_map = df3.copy()
    gdf = gpd.GeoDataFrame(df_map, geometry=gpd.points_from_xy(df_map.Lon, df_map.Lat))
    categories = ["decreasing", "no trend", "increasing"]
    category_dict = {category: idx for idx, category in enumerate(categories)}
    gdf["category_num"] = gdf[column].map(category_dict)
    
    # Create direct color mapping while maintaining original structure
    colors = dict(zip(range(len(categories)), cmap))
    gdf_shape.plot(ax=ax, color="white", edgecolor="black", linewidth= 1.0, zorder=1)
    
    markers = {0: 'v', 1: 'o', 2: '^'}  # Define markers for categories
    for cat in range(len(categories)):
        mask = gdf["category_num"] == cat
        
        # Special handling for "no trend" (category 1)
        if cat == 1:  # "no trend"
            ax.scatter(
                gdf[mask].geometry.x,
                gdf[mask].geometry.y,
                facecolors="none",  # No fill
                edgecolors=colors[cat],  # Outline only
                marker=markers[cat],
                s=5,  # Adjust size if needed
                alpha=0.7,
                label=categories[cat],
                zorder=3
            )
        else:  # Regular handling for other categories
            ax.scatter(
                gdf[mask].geometry.x,
                gdf[mask].geometry.y,
                c=[colors[cat]],
                marker=markers[cat],
                s=20,
                alpha=0.7,
                label=categories[cat],
                zorder=3
            )
    gdf_shape.plot(ax=ax, color="none", edgecolor="black", linewidth= 1.2, zorder=4)
    ax.set_axis_off()
    ax.set_title(title, fontsize=font, loc="left")
    return None

def plot_all_maps_categorical_grid(gdf_shape, df_list, column_names, titles, font=20, cmap=["red", "grey", "green"], cbar_label="AI Categories", save_as=None):
    plt.rcParams.update({"font.size": font, "font.family": "Arial", "font.weight": "regular"})
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (ax, df) in enumerate(zip(axes.flat, df_list)):
        map_plot_categorical(axes[i], gdf_shape, df, font=font, column=column_names[i], title=titles[i], cmap=cmap)
    
    handles, labels = axes[0].get_legend_handles_labels()
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    
    # Add legend at bottom
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.16), ncol=3)
    
    if save_as:
        save_as_updated = fr"C:\Users\adeba\OneDrive\Documents\Datascience\Hydrology\Jupyter_lab\US_weather\Figures\{save_as}"
        plt.savefig(save_as_updated, dpi=500, bbox_inches="tight")
        print(f"The picture has been saved as {save_as}")
    
    plt.show()

def filter_trend(list_dfs, list_columns):
    new_list=[]
    for df, col in zip (list_dfs, list_columns):
        df_new = df[~(df[col]=="no trend")]
        new_list.append(df_new)
    return new_list


def plot_decreasing_trend_percentage(trend_area_data , trend_columns, area_data_list, labels =  ["100 years", "50 years", "30 years", "10 years"], colors = ['#0033A0', '#FFD100', '#C99700', '#F1EB9C'], ylabel= "AI trend",save_as=None):
    """
    Calculates the percentage of area with decreasing trends, plots a bar chart, and optionally saves the figure.

    Parameters:
        trend_area_data (list of DataFrame): List of dataframes with trend data for each time period.
        area_data_list (list of DataFrame): List of dataframes with area data for each time period.
        trend_columns (list of str): List of column names representing trends in the dataframes.
        labels (list of str): Labels for each time period.
        colors (list of str): Colors for the bars in the chart.
        save_as (str, optional): Path to save the figure as a `.png` file. If None, the figure will not be saved.

    Returns:
        list: Percentages of areas with decreasing trends for each time period.
    """
    percentages = []

    for df_trend, area_df, trend_column, label in zip(trend_area_data, area_data_list, trend_columns, labels):
        # Merge trend data with area data
        df_trend_area = df_trend.merge(area_df, on="ID", how="left")

        # Filter for decreasing trend
        df_trend_area_decrease = df_trend_area[df_trend_area[trend_column] == "decreasing"]

        # Calculate the percentage of area with decreasing trend
        percent_decrease = 100 * (df_trend_area_decrease["Area"].sum()) / (area_df["Area"].sum())
        percentages.append(percent_decrease)

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, percentages, color=colors)

    # Annotate bars with integer values positioned above each bar
    for bar, value in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{int(round(value, 0))}%", 
                 ha='center', va='bottom', color='black')

    # Customize plot appearance
    plt.xlabel("Length of record", fontsize=20)
    plt.ylabel(f"Percentage of CONUS\n with Decreasing {ylabel}", fontsize=20)
    plt.ylim(0, max(percentages) + 10)  # Adjust y-limit for better spacing

    # Remove the upper and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Add a legend
    plt.legend(bars, labels, loc="upper right", fontsize=14)

    # Adjust layout
    plt.tight_layout()

    # Save the figure if a file path is provided
    if save_as:
        save_as_updated = fr"C:\Users\adeba\OneDrive\Documents\Datascience\Hydrology\Jupyter_lab\US_weather\Figures\{save_as}"
        plt.savefig(save_as_updated, dpi=500, bbox_inches="tight")
        print(f"The picture has been saved as {save_as}")
        
    # Show the plot
    plt.show()

    return percentages

def plot_increasing_trend_percentage(trend_area_data, trend_columns, area_data_list, labels= ["100 years", "50 years", "30 years", "10 years"], colors = ['#0033A0', '#FFD100', '#C99700', '#F1EB9C'], ylabel="AI trend", save_as=None):
    """
    Calculates the percentage of area with increasing trends, plots a bar chart, and optionally saves the figure.

    Parameters:
        trend_area_data (list of DataFrame): List of dataframes with trend data for each time period.
        area_data_list (list of DataFrame): List of dataframes with area data for each time period.
        trend_columns (list of str): List of column names representing trends in the dataframes.
        labels (list of str): Labels for each time period.
        colors (list of str): Colors for the bars in the chart.
        save_as (str, optional): Path to save the figure as a `.png` file. If None, the figure will not be saved.

    Returns:
        list: Percentages of areas with increasing trends for each time period.
    """
    percentages = []

    for df_trend, area_df, trend_column, label in zip(trend_area_data, area_data_list, trend_columns, labels):
        # Merge trend data with area data
        df_trend_area = df_trend.merge(area_df, on="ID", how="left")

        # Filter for increasing trend
        df_trend_area_increase = df_trend_area[df_trend_area[trend_column] == "increasing"]

        # Calculate the percentage of area with increasing trend
        percent_increase = 100 * (df_trend_area_increase["Area"].sum()) / (area_df["Area"].sum())
        percentages.append(percent_increase)

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, percentages, color=colors)

    # Annotate bars with integer values positioned above each bar
    for bar, value in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{int(round(value, 0))}%", 
                 ha='center', va='bottom', color='black')

    # Customize plot appearance
    plt.xlabel("Length of record", fontsize=20)
    plt.ylabel(f"Percentage of CONUS\n with Increasing {ylabel}", fontsize=20)
    plt.ylim(0, max(percentages) + 10)  # Adjust y-limit for better spacing

    # Remove the upper and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Add a legend
    plt.legend(bars, labels, loc="upper right", fontsize=14)

    # Adjust layout
    plt.tight_layout()

    # Save the figure if a file path is provided
    if save_as:
        save_as_updated = fr"C:\Users\adeba\OneDrive\Documents\Datascience\Hydrology\Jupyter_lab\US_weather\Figures\{save_as}"
        plt.savefig(save_as_updated, dpi=500, bbox_inches="tight")
        print(f"The picture has been saved as {save_as}")
        
    # Show the plot
    plt.show()

    return percentages
def plot_trend_percentage(trend_area_data, area_data_list, trend_columns, labels = ["100 years", "50 years", "30 years", "10 years"], colors_increasing= ['blue', 'blue', 'blue', 'blue'], colors_decreasing =['red', 'red', 'red', 'red'], ylabel="AI trend", save_as=None):
    """
    Calculates the percentage of area with increasing and decreasing trends, plots a grouped bar chart,
    and optionally saves the figure.

    Parameters:
        trend_area_data (list of DataFrame): List of dataframes with trend data for each time period.
        area_data_list (list of DataFrame): List of dataframes with area data for each time period.
        trend_columns (list of str): List of column names representing trends in the dataframes.
        labels (list of str): Labels for each time period.
        colors_increasing (list of str): Colors for the bars representing increasing trends.
        colors_decreasing (list of str): Colors for the bars representing decreasing trends.
        ylabel (str): Label for the y-axis.
        save_as (str, optional): Path to save the figure as a `.png` file. If None, the figure will not be saved.

    Returns:
        tuple: Percentages of areas with increasing and decreasing trends for each time period.
    """
    percentages_increasing = []
    percentages_decreasing = []

    for df_trend, area_df, trend_column, label in zip(trend_area_data, area_data_list, trend_columns, labels):
        # Merge trend data with area data
        df_trend_area = df_trend.merge(area_df, on="ID", how="left")

        # Filter for increasing and decreasing trends
        df_trend_area_increase = df_trend_area[df_trend_area[trend_column] == "increasing"]
        df_trend_area_decrease = df_trend_area[df_trend_area[trend_column] == "decreasing"]

        # Calculate the percentages
        percent_increase = 100 * (df_trend_area_increase["Area"].sum()) / (area_df["Area"].sum())
        percent_decrease = 100 * (df_trend_area_decrease["Area"].sum()) / (area_df["Area"].sum())

        percentages_increasing.append(percent_increase)
        percentages_decreasing.append(percent_decrease)

    # Plotting the grouped bar chart
    x = np.arange(len(labels))  # Label locations
    width = 0.35  # Width of the bars

    plt.figure(figsize=(12, 7))
    bars_increasing = plt.bar(x - width / 2, percentages_increasing, width, color=colors_increasing, label="Increasing")
    bars_decreasing = plt.bar(x + width / 2, percentages_decreasing, width, color=colors_decreasing, label="Decreasing")

    # Annotate bars with integer values positioned above each bar
    for bar, value in zip(bars_increasing, percentages_increasing):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{int(round(value, 0))}%", 
                 ha='center', va='bottom', color='black')

    for bar, value in zip(bars_decreasing, percentages_decreasing):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{int(round(value, 0))}%", 
                 ha='center', va='bottom', color='black')

    # Customize plot appearance
    plt.xlabel("Length of record", fontsize=20)
    plt.ylabel(f"Percentage of CONUS\n with {ylabel}", fontsize=20)
    plt.xticks(x, labels, fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0, max(max(percentages_increasing), max(percentages_decreasing)) + 10)

    # Remove the upper and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Add a legend
    plt.legend(loc="upper right", fontsize=20)

    # Adjust layout
    plt.tight_layout()

    # Save the figure if a file path is provided
    if save_as:
        save_as_updated = fr"C:\Users\adeba\OneDrive\Documents\Datascience\Hydrology\Jupyter_lab\US_weather\Figures\{save_as}"
        plt.savefig(save_as_updated, dpi=500, bbox_inches="tight")
        print(f"The picture has been saved as {save_as}")

    # Show the plot
    plt.show()

    return percentages_increasing, percentages_decreasing

from matplotlib import cm
from matplotlib.colors import Normalize

def map_plot_categorical_class(ax, gdf_shape, df3, font=20, column="Value_Column", title="", categories=None, reverse=False):
    """
    Plot categorical data on a map using defined categories with consistent colors in the legend.

    Parameters:
        ax (AxesSubplot): The matplotlib axes to plot on.
        gdf_shape (GeoDataFrame): Shapefile for the background.
        df3 (DataFrame): DataFrame containing the data to plot.
        font (int): Font size for the title.
        column (str): Column name for the values to be plotted.
        title (str): Title of the plot.
        categories (list of tuples): List defining ranges and categories [(upper_limit, category_name), ...].
        reverse (bool): If True, reverse the color order in the colormap.

    Returns:
        tuple: (scatter plot object, category_mapping)
    """
    df_map = df3.copy()

    # Categorize values based on provided ranges
    if categories:
        def categorize(value):
            for upper_limit, category in categories:
                if value <= upper_limit:
                    return category
            return categories[-1][1]  # Default to the last category if no match

        df_map[column] = df_map[column].apply(categorize)

    gdf_shape.plot(ax=ax, color="white", edgecolor="black", linewidth= 1.0, zorder=1)

    # Create colormap and reverse if needed
    colormap = cm.get_cmap('rainbow', len(categories))
    if reverse:
        colormap = colormap.reversed()
    norm = Normalize(vmin=0, vmax=len(categories) - 1)

    # Map categories to numeric codes
    category_mapping = {category: idx for idx, category in enumerate([cat[1] for cat in categories])}
    df_map["Category_Code"] = df_map[column].map(category_mapping)

    scatter = ax.scatter(
        df_map.Lon,
        df_map.Lat,
        c=df_map["Category_Code"],
        cmap=colormap,
        s=10,
        alpha=0.7,
        zorder=3
    )
    gdf_shape.plot(ax=ax, color="none", edgecolor="black", linewidth= 1.2, zorder=4)
    ax.set_title(title, fontsize=font, loc="left")
    ax.set_axis_off()
    return scatter, category_mapping

def plot_all_maps_categorical_grid_class(
    gdf_shape, df_list, column_names, titles, font=20, categories=None, reverse=False, cbar= "AI", save_as=None
):
    """
    Plot all maps for categorical data in a grid layout with a single horizontal legend below the figures.

    Parameters:
        gdf_shape (GeoDataFrame): Shapefile for the background.
        df_list (list of DataFrames): List of DataFrames to plot.
        column_names (list of str): Column names for the values to plot.
        titles (list of str): Titles for the plots.
        font (int): Font size for the title.
        categories (list of tuples): Categories for the categorical map.
        reverse (bool): If True, reverse the color order in the colormap.
        save_as (str): Filename to save the plot.

    Returns:
        None
    """
    plt.rcParams.update({"font.size": font, "font.family": "Arial", "font.weight": "regular"})
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    colormap = cm.get_cmap('rainbow', len(categories))
    if reverse:
        colormap = colormap.reversed()
    norm = Normalize(vmin=0, vmax=len(categories) - 1)

    # Plot each DataFrame
    for i, df in enumerate(df_list):
        scatter, category_mapping = map_plot_categorical_class(
            axes[i], gdf_shape, df, column=column_names[i], title=titles[i], categories=categories, reverse=reverse
        )

    # Create legend handles using the same colors from the colormap
    handles = [
        plt.Line2D(
            [0], [0],
            marker='o',
            color=colormap(norm(idx)),  # Use the same color mapping
            markersize=10,
            label=category
        )
        for idx, (_, category) in enumerate(categories)
    ]

    # Add the horizontal legend below the plots
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(categories),
        title=f"{cbar} Categories",
        fontsize=font - 2,
        title_fontsize=font - 2,
        frameon=False
    )

    # Adjust spacing to accommodate the legend
    fig.subplots_adjust(bottom=0.15)

    # Save the picture if a name is passed
    if save_as:
        save_as_updated = fr"C:\Users\adeba\OneDrive\Documents\Datascience\Hydrology\Jupyter_lab\US_weather\Figures\{save_as}"
        plt.savefig(save_as_updated, dpi=500, bbox_inches="tight")
        print(f"The picture has been saved as {save_as}")

    plt.show()
def clip_list(df_list, column_names, val_min, val_max):
    """
    Clips values in specified columns of DataFrames in a list to a given range.

    Parameters:
        df_list (list of DataFrame): List of DataFrames to process.
        column_names (list of str): List of column names corresponding to each DataFrame.
        val_min (float): Minimum value to clip to.
        val_max (float): Maximum value to clip to.

    Returns:
        list: List of DataFrames with clipped values.
    """
    clipped_list = []  # Initialize a new list to store clipped DataFrames

    for df, column in zip(df_list, column_names):
        df = df.copy()  # Make a copy to avoid modifying the original DataFrame
        df[column] = df[column].clip(lower=val_min, upper=val_max)  # Clip the values
        clipped_list.append(df)  # Append the modified DataFrame to the new list

    return clipped_list

def plot_categorical_percentage_barchart(
    df_list, column_names, categories, list_area, labels=["100 years", "50 years", "30 years", "10 years"],
    font=20, colors=None, reverse_cmap=False, ylabel="Percentage of Categories", save_as=None
):
    """
    Plot bar charts showing percentages of areas for each category across multiple datasets.

    Parameters:
        df_list (list of DataFrames): List of DataFrames to process.
        column_names (list of str): Column names for the categorical values in each DataFrame.
        categories (list of tuples): List defining ranges and categories [(upper_limit, category_name), ...].
        list_area (list of DataFrames): List of area DataFrames corresponding to df_list.
        labels (list of str): Labels for each dataset.
        font (int): Font size for the plot.
        colors (list of str, optional): Colors for the bars in the chart. If None, default colors are used.
        reverse_cmap (bool): If True, reverse the color order in the colormap.
        ylabel (str): Y-axis label for the bar chart.
        save_as (str, optional): Path to save the figure as a `.png` file. If None, the figure will not be saved.

    Returns:
        dict: A dictionary of percentages for each category across datasets.
    """
    # Default colors if not provided
    if colors is None:
        cmap = plt.cm.rainbow
        if reverse_cmap:
            cmap = cmap.reversed()
        colors = cmap(np.linspace(0, 1, len(categories)))

    percentages_dict = {cat[1]: [] for cat in categories}

    for df, area_dataframe, column, label in zip(df_list, list_area, column_names, labels):
        df_area = df.merge(area_dataframe, on="ID", how="left")
        total_area_temp = df_area["Area"].sum()

        # Initialize a mask for unassigned rows
        unassigned_mask = np.ones(len(df_area), dtype=bool)

        for upper_limit, category in categories:
            # Filter rows that belong to the current category
            category_mask = unassigned_mask & (df_area[column] <= upper_limit)
            class_area = df_area.loc[category_mask, "Area"].sum()
            percentage = 100 * class_area / total_area_temp

            # Append percentage and update unassigned mask
            percentages_dict[category].append(percentage)
            unassigned_mask &= ~category_mask

    # Plotting the bar chart
    x = range(len(labels))  # X positions for the groups
    bar_width = 0.8 / len(categories)  # Width of each bar
    offsets = [-0.4 + i * bar_width for i in range(len(categories))]  # Offset positions for each bar

    plt.figure(figsize=(12, 7))
    for i, (category, color) in enumerate(zip(percentages_dict.keys(), colors)):
        plt.bar(
            [pos + offsets[i] for pos in x], 
            percentages_dict[category], 
            bar_width, 
            label=category, 
            color=color
        )

    # Customize plot appearance
    plt.xticks(x, labels, fontsize=14)
    plt.xlabel("Length of record", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.ylim(0, 100)  # Percentages range from 0 to 100

    # Add legend
    plt.legend(title="Categories", fontsize=14, title_fontsize=14, loc="upper right",ncol=3)

    # Annotate bars with percentages
    for i, category in enumerate(percentages_dict.keys()):
        for j, value in enumerate(percentages_dict[category]):
            plt.text(
                x[j] + offsets[i], 
                value + 1, 
                f"{(round(value, 1))}%", 
                ha='center', 
                va='bottom', 
                fontsize=12
            )

    # Remove the upper and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    # Save the figure if a file path is provided
    if save_as:
        save_as_updated = fr"C:\Users\adeba\OneDrive\Documents\Datascience\Hydrology\Jupyter_lab\US_weather\Figures\{save_as}"
        plt.savefig(save_as_updated, dpi=500, bbox_inches="tight")
        print(f"The picture has been saved as {save_as}")

    # Show the plot
    plt.show()

    return percentages_dict
import pandas as pd
import os
import xlsxwriter
def save_dataframes(df_list, filename):
    """
    Saves a list of DataFrames into an Excel file with each DataFrame stored in a separate sheet.
    
    Parameters:
        df_list (list of pd.DataFrame): List of DataFrames to save.
        filename (str): The name of the file to save, must end in `.xlsx` for Excel.
        
    Returns:
        None
    """
    sheet_names = ["100years", "50years", "30years", "10years"]

    # Define the output folder
    output_folder = r"C:\Users\adeba\OneDrive\Documents\Datascience\Hydrology\Jupyter_lab\US_weather\Trends_files"

    # Ensure the folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Combine folder path and filename correctly
    full_path = os.path.join(output_folder, filename)

    if filename.endswith(".xlsx"):
        with pd.ExcelWriter(full_path, engine="xlsxwriter") as writer:
            for df, sheet in zip(df_list, sheet_names):
                df.to_excel(writer, sheet_name=sheet, index=False)
        print(f"DataFrames successfully saved to: {full_path}")
    
    else:
        raise ValueError("Filename must end with .xlsx for Excel.")

# def plot_boxplot(df,df1, column="AI", y_label="Aridity Index (AI)"):
#     # Create figure
#     plt.figure(figsize=(12, 6))  # Adjust figure size

#     # Define the domain list
#     domain_list = [
#         "Desert Southwest",
#         "Great Basin",
#         "Colorado Plateau",
#         "Central Plains",
#         "Northern Plains",
#         "Northern Rockies",
#         "Pacific Southwest",
#         "Southern Plains",
#         "Atlantic Neotropical",
#         "Prairie Peninsula",
#         "Southeast",
#         "Great Lakes",
#         "Mid Atlantic",
#         "Ozarks Complex",
#         "Cumberland Plateau",
#         "Northeast",
#         "Pacific Northwest"
#     ]

#     # Define replacement dictionary
#     replace_dict = {
#         "Southern Rockies / Colorado Plateau": "Colorado Plateau",
#         "Appalachians / Cumberland Plateau": "Cumberland Plateau"
#     }

#     df_updated = df.copy()
#     df_updated = df_updated.merge(df1, on=["ID"], how="left")
#     # Check if 'domainName' column exists before replacing
#     if "domainName" in df_updated.columns:
#         df_updated["domainName"] = df_updated["domainName"].replace(replace_dict)

#     # Create violin plot (slightly transparent)
#     sns.violinplot(x='domainName', y=column, data=df_updated, order=domain_list, color="lightgreen", inner=None, alpha=0.6)

#     # Overlay boxplot (keeps the box visible)
#     sns.boxplot(x='domainName', y=column, data=df_updated, order=domain_list, color="green", width=0.3, boxprops={'zorder': 2})

#     # Set font properties for axis labels and ticks
#     plt.xlabel("Eco region", fontsize=18, fontname="Arial")  # X-axis label
#     plt.ylabel(y_label, fontsize=18, fontname="Arial")  # Y-axis label
#     plt.xticks(rotation=90, fontsize=18, fontname="Arial")  # X-axis ticks
#     plt.yticks(fontsize=18, fontname="Arial")  # Y-axis ticks
#     plt.tight_layout()

#     # Save the figure
#     plt.savefig(fr"C:\Users\adeba\OneDrive/Documents/Datascience/Hydrology/Jupyter_lab/US_weather_Neon/Figures/{column}_boxplot.png", dpi=500)

#     # Show the plot
#     plt.show()


def plot_boxplot(df, df1, column="AI", y_label="Aridity Index (AI)"):
    # Create figure
    plt.figure(figsize=(12, 6))

    # Define replacement dictionary
    replace_dict = {
        "Southern Rockies / Colorado Plateau": "Colorado Plateau",
        "Appalachians / Cumberland Plateau": "Cumberland Plateau"
    }

    df_updated = df.copy()
    df_updated = df_updated.merge(df1, on=["ID"], how="left")

    if "domainName" in df_updated.columns:
        df_updated["domainName"] = df_updated["domainName"].replace(replace_dict)

    # Compute means for ordering
    df_updated_grouped = df_updated.groupby(["domainName"])[column].mean().reset_index()
    df_updated_grouped.sort_values(by=column, ascending=True, inplace=True)

    # Dynamic domain order from grouped data
    domain_order = df_updated_grouped["domainName"].tolist()

    # Plot violin + boxplot
    sns.violinplot(x='domainName', y=column, data=df_updated, order=domain_order,
                   color="lightgreen", inner=None, alpha=0.6)
    sns.boxplot(x='domainName', y=column, data=df_updated, order=domain_order,
                color="green", width=0.3, boxprops={'zorder': 2})

    # Axis formatting
    plt.xlabel("Eco region", fontsize=20, fontname="Arial", fontweight='bold')
    plt.ylabel(y_label, fontsize=20, fontname="Arial", fontweight='bold')
    plt.xticks(rotation=90, fontsize=20, fontname="Arial", fontweight='bold')
    plt.yticks(fontsize=20, fontname="Arial", fontweight='bold')

    # Layout and save
    plt.tight_layout()
    plt.savefig(fr"C:\Users\adeba\OneDrive\Documents\Datascience\Hydrology\Jupyter_lab\US_weather_Neon\Figures\{column}_boxplot.png", dpi=500)
    plt.show()
