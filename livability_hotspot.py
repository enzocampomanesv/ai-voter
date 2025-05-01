import pandas as pd
import geopandas as gpd
import numpy as np
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
import esda
import libpysal
from splot.esda import plot_local_autocorrelation

def assign_cluster(row, alpha = 0.05):
    moran_i = row['Local_Moran_I']
    p_value = row['p_value']
    z_score = row['z_value']

    if moran_i > 0 and p_value < alpha and z_score > 0:
        return 'HH'
    elif moran_i > 0 and p_value < alpha and z_score < 0:
        return 'LL'
    elif moran_i < 0 and p_value < alpha and z_score > 0:
        return 'HL'
    elif moran_i < 0 and p_value < alpha and z_score < 0:
        return 'LH'
    else:
        return 'Not Significant'

city = "ACC"
base_tm_set = ['communities_wtsm','planners_wtsm','combined_wtsm']
num_votes_set = ['860k']
d_src = "S1_200m"
resolution = 200

final_df = gpd.GeoDataFrame(columns=['model', 'num_votes', 'PARTIMAP_I', 'n_ts', 'conf', 'geometry'])
for base_tm in base_tm_set:
    for num_votes in num_votes_set:
        in_gpkg_path = f"{city}/gpkg/{city}_200m_ranked_{base_tm}_{num_votes}.gpkg"
        gdf = gpd.read_file(in_gpkg_path)
        final_df.crs = gdf.crs
        # print(f"{len(gdf)} rows in {base_tm}->{num_votes}")
        
        # Ensure the required columns exist
        if all(col in gdf.columns for col in ['PARTIMAP_I', 'n_ts', 'conf']):
            # Add the model_set and num_votes_set columns
            gdf['model'] = base_tm
            gdf['num_votes'] = num_votes
            
            # Append the relevant columns to the final DataFrame
            final_df = pd.concat([final_df, gdf[['model', 'num_votes', 'PARTIMAP_I', 'n_ts', 'conf', 'geometry']]], ignore_index=True)
        else:
            print(f"Required columns not found in {in_gpkg_path}")
            
var_set = ['n_ts', 'conf']
for var in var_set:
    for curr_model in base_tm_set:
        for n_votes in num_votes_set:
            # curr_model = 'planners_base'
            # n_votes = '2M'
            print(f"Hotspot analysis for {var} {curr_model} {n_votes}") 
            curr_gdf = final_df[(final_df['model']==curr_model)&(final_df['num_votes']==n_votes)]
            y = curr_gdf[var].values

            # Create a spatial weights matrix (e.g., Queen contiguity)
            w = libpysal.weights.Queen.from_dataframe(curr_gdf)
            islands = [i for i, neighbors in w.neighbors.items() if not neighbors]
            curr_gdf = curr_gdf.drop(curr_gdf.index[islands]).reset_index(drop=True)
            w = libpysal.weights.Queen.from_dataframe(curr_gdf)
            y = curr_gdf[var].values

            # Calculate Local Moran's I
            lisa = esda.Moran_Local(y, w)

            # Add results to the GeoDataFrame
            curr_gdf['Local_Moran_I'] = lisa.Is
            curr_gdf['p_value'] = lisa.p_sim
            curr_gdf['cluster'] = lisa.q  # 1=HH, 2=LH, 3=LL, 4=HL
            curr_gdf = gpd.GeoDataFrame(curr_gdf, geometry='geometry')

            # 1. Calculate the z-score:
            mean_attr = curr_gdf[var].mean()
            std_attr = curr_gdf[var].std()
            curr_gdf['z_value'] = (curr_gdf[var] - mean_attr) / std_attr


            curr_gdf['cluster_type'] = curr_gdf.apply(assign_cluster, axis=1)
            out_hotspot_path = f"{city}/hotspot/{var}/{city}_{var}_hotspot_{curr_model}_{n_votes}.gpkg"
            curr_gdf.to_file(out_hotspot_path, layer=f'{var}_hotspot', driver="GPKG")