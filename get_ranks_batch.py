import pandas as pd
import geopandas as gpd
import sqlite3
import numpy as np
import rasterio
from rasterio.features import rasterize
import votes.packages.trueskill
from votes.packages.trueskill import Rating, quality_1vs1, rate_1vs1
from sklearn.preprocessing import MinMaxScaler
import random

# Set vars
# city = sys.argv[1]
# base_tm = sys.argv[2]
# num_votes = sys.argv[3]

city = "ACC"
base_tm_set = ['planners_wtsm', 'communities_wtsm', 'combined_wtsm']
# num_votes_set = ['150k', '430k', '860k']
num_votes_set = ['860k']
resolution = 200
d_src = "S1_200m"

def rasterize_gdf(gdf, col_name, out_name, resolution):
    xmin, ymin, xmax, ymax = gdf.total_bounds
    width = int(np.ceil((xmax - xmin) / resolution))
    height = int(np.ceil((ymax - ymin) / resolution))
    transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)
    
    rasterized = rasterize(
                shapes=((geom, value) for geom, value in zip(gdf.geometry, gdf[col_name])),
                out_shape=(height, width),
                transform=transform,
                fill=0,  # Background value
                dtype=np.float32 )
    with rasterio.open(
        out_name,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=rasterized.dtype,
        crs=gdf.crs,
        transform=transform,
    ) as dst:
        dst.write(rasterized, 1)
        
def get_win_lose(row):
    if row.prediction == 0:
        return pd.Series([row.left, row.right])
    else:
        return pd.Series([row.right, row.left])

def process_ranks(city, voter, n_votes, reso):
    tbl_name = f'{city}_voter_{voter}_votes_{n_votes}'
    out_tbl = f'{city}_voter_{voter}_ranks_{n_votes}'
    path = f'votes/db/{city}/{city}_voterAI.db'
    con = sqlite3.connect(path)

    # Read votes
    sql= f"select * from {tbl_name}"
    vf = pd.read_sql(sql, con = con)

    # Preprocess votes
    if 'wins' not in vf.columns:
        vf[['wins','lose']] = vf.apply(get_win_lose, axis=1)
    vf['wins'] = vf['wins'].astype(np.int64)
    vf['lose'] = vf['lose'].astype(np.int64)
    # TrueSkill ranking
    print("  playing matches")
    rank={}
    for n, v in vf.iterrows():
        print(n, end = '\r')
        wins  = v['wins']
        lose = v['lose']

        # Create new "rating" for new "player"
        if wins not in rank.keys():
            rank[wins]=Rating() # Default rating is mu=25, sigma=8.33
        if lose not in rank.keys():
            rank[lose]=Rating()

        # Play match
        rank[wins], rank[lose] = rate_1vs1(rank[wins], rank[lose])

    k=list(rank.keys())
    k.sort()

    results=[]
    for r in k:
        results.append({'PARTIMAP_I':r, 'value':rank[r].mu, 'conf':rank[r].sigma})

    df = pd.DataFrame(results)
    print(df.shape)

    # Norm ratings to 0-1
    scaler = MinMaxScaler()
    df['n_ts'] = scaler.fit_transform(df[['value']])
    df.head()

    # Export table
    print("  exporting")
    df.to_sql(out_tbl, con = con, if_exists= 'replace')

    # Export to gpkg
    gpkg_file = f"clustering/{city}/{city}_LCZ_morph_complete_200m.gpkg"
    city_tiles = gpd.read_file(gpkg_file)
    city_tiles_ranked = city_tiles[['PARTIMAP_I','geometry']].merge(df, on='PARTIMAP_I')

    output_gpkg = f"ranking/{city}/gpkg/{city}_200m_ranked_{voter}_{n_votes}.gpkg"
    city_tiles_ranked.to_file(output_gpkg, layer='citytiles_ranked', driver="GPKG")

    rasterize_gdf(city_tiles_ranked, 'n_ts', f"ranking/{city}/tif/score/{city}_200m_score_{voter}_{n_votes}.tif", reso)
    rasterize_gdf(city_tiles_ranked, 'conf', f"ranking/{city}/tif/conf/{city}_200m_conf_{voter}_{n_votes}.tif", reso)

for base_tm in base_tm_set:
    for num_votes in num_votes_set:
        print(f"Ranking {base_tm} based on {num_votes} votes for {city}")
        process_ranks(city, base_tm, num_votes, resolution)
