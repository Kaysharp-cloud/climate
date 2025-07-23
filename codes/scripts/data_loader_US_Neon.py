# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import geopandas as gpd
from functools import reduce
from mpl_toolkits.axes_grid1 import make_axes_locatable
import general_functions_US_Neon as kay
# Data Loading
file_paths = {"ENSO":r"C:\Users\adeba\OneDrive\Documents\Datascience\Hydrology\Jupyter_lab\df_all_nino_and_anom_oni.csv",
    "daily": r"C:\Users\adeba\OneDrive\Documents\Datascience\Hydrology\Jupyter_lab\US_weather\df_manual_clean_daily.csv",
    "spei": r"C:\Users\adeba\OneDrive\Documents\Datascience\Hydrology\Jupyter_lab\US_weather\SPEI_US.csv",
    # "shape": r"C:\Users\adeba\OneDrive\Documents\Datascience\Hydrology\Jupyter_lab\US_weather\US_CONUS_Shapefile\US_CONUS.shp"
    "shape": r"C:\Users\adeba\OneDrive\Documents\Datascience\Hydrology\Jupyter_lab\US_weather\US_HUC2_clip\US_HUC2_clip.shp"
}
df_id_eco = pd.read_csv(r"C:\Users\adeba\OneDrive\Documents\Datascience\Hydrology\Jupyter_lab\US_weather\ID_ecoregion.csv")

df_id_eco.drop(columns=["Lat","Lon"], inplace=True)

neon_shape=r"C:\Users\adeba\OneDrive\Documents\Datascience\Hydrology\Jupyter_lab\US_weather\Neon_clip\Neon_clip.shp"
# Load Data
df_enso= pd.read_csv(file_paths["ENSO"])
#US = gpd.read_file(file_paths["shape"])
US =gpd.read_file(neon_shape)
df_daily= pd.read_csv(file_paths["daily"])
df_daily = df_daily[~(df_daily["ID"]=="USW00014755")]

df_monthly= pd.read_csv(file_paths["spei"])
df_monthly = df_monthly[~(df_monthly["ID"]=="USW00014755")]


df_monthly = df_monthly[df_monthly["S_year"]<=1924]
df_daily = df_daily[df_daily["S_year"]<=1924]

# Clipping function for all spei and spi columns
def clip_spei_spi(df, lower_bound=-3, upper_bound=3):
    """
    Clips all columns with 'spei' and 'spi' in their names to a specified range.

    Parameters:
        df (DataFrame): The DataFrame containing spei and spi columns.
        lower_bound (float): The lower bound for clipping.
        upper_bound (float): The upper bound for clipping.

    Returns:
        DataFrame: The DataFrame with clipped values for spei and spi columns.
    """
    # Select columns containing 'spei' or 'spi' in their names
    spei_spi_columns = [col for col in df.columns if 'spei' in col.lower() or 'spi' in col.lower()]
    
    # Clip values to the specified range
    for col in spei_spi_columns:
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df
# file_path_basin =r"C:\Users\adeba\OneDrive\Documents\Datascience\Hydrology\Jupyter_lab\US_weather\thiessen_updated\df_10years_info_huc2.csv"

# df_basin = pd.read_csv(file_path_basin)
# df_basin_name = df_basin[["ID", "name"]]
# df_basin_name.rename(columns={"name":"basin"}, inplace= True)
df_monthly = df_monthly.merge(df_id_eco, on="ID", how="left")
df_daily = df_daily.merge(df_id_eco, on="ID", how="left")

# Apply the function to clip spei and spi columns
df_monthly = clip_spei_spi(df_monthly, lower_bound=-3, upper_bound=3)

df_monthly["DATE"]= pd.to_datetime(df_monthly[["YEAR","MONTH"]].assign(DAY=1))

df_yearly = df_monthly.groupby(["ID","YEAR","Lat","Lon","STATE","S_year","domainName"]).agg({"PRCP":"sum","TMAX":"mean","TMIN":"mean","PET_thornwaite":"sum","spei1":"mean","spei3":"mean","spei6":"mean","spei12":"mean"}).reset_index()
df_yearly["AI"]= df_yearly["PRCP"]/df_yearly["PET_thornwaite"]

# List of CSV file paths
area_thie = [
    r"C:\Users\adeba\OneDrive\Documents\Datascience\Hydrology\Jupyter_lab\US_weather\thiessen_updated\Thiessen_Polygons_100yrs.csv",
    r"C:\Users\adeba\OneDrive\Documents\Datascience\Hydrology\Jupyter_lab\US_weather\thiessen_updated\Thiessen_Polygons_50yrs.csv",
    r"C:\Users\adeba\OneDrive\Documents\Datascience\Hydrology\Jupyter_lab\US_weather\thiessen_updated\Thiessen_Polygons_30yrs.csv",
    r"C:\Users\adeba\OneDrive\Documents\Datascience\Hydrology\Jupyter_lab\US_weather\thiessen_updated\Thiessen_Polygons_10yrs.csv"
]
df_clean_area = []
# Loop through each CSV file
for file in area_thie:
    # Read the CSV file into a Pandas DataFrame
    df_area_needed = pd.read_csv(file)
    
    # Rename columns as needed
    df_area_needed.rename(columns={"ID_1": "ID", "Area_sqkm": "Area"}, inplace=True)
    df_area_needed = df_area_needed[["ID","Area"]]
    # Display the first few rows to verify
    print(f"Processed file: {file}")
    print(df_area_needed.head())
    df_clean_area.append(df_area_needed)
    # Save the cleaned file (optional)
    # output_path = file.replace(".csv", "_cleaned.csv")
    # df_area_needed.to_csv(output_path, index=False)
    print("Cleaned data ")
thies_dfs = df_clean_area 
titles_journal= ["(a)", "(b)", "(c)","(d)"]
titles_journal_2= ["(e)", "(f)", "(g)","(h)"]