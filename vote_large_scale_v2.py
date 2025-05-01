import sys
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import csv
import sqlite3
from osgeo import gdal
from train_voter import load_and_preprocess_image
import matplotlib.pyplot as plt
import random

def retrieve_db(db_path, sql):
    con = sqlite3.connect(db_path)
    return con, pd.read_sql(sql, con = con)

def load_image_viz(city, data, yr, tile_id, sub_bands):
    if data in ['RGB','RGBN','RGN']:
        data = "S2"
    path = f"votes/{yr}/{city}/{data}/"
    img = load_and_preprocess_image(path+f"{city}_{data}_{tile_id}.tif", sub_bands)
    img = np.transpose(img, (1,2,0))
    img = img[:-1, :-1, :]  # Shape: (num_samples, 20, 20, num_bands)
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, 0)
    return img

def vote(row, d_src, sub_bands, model):
    print(row.name, end='\r')
    left = row['left'].astype(int)
    right = row['right'].astype(int)
    img_l = load_image_viz(d_src, left, sub_bands)
    img_r = load_image_viz(d_src, right, sub_bands)
    prediction = model([img_l, img_r]).numpy()[0][0]
    if prediction < 0.5:
        p = pd.Series([prediction, left, right])
    else:
        p = pd.Series([prediction, right, left])
    
    return p

def get_sub_bands(d_src):
    if d_src == "RGBN":
        sub_bands = [0,1,2,6]
    elif d_src == "RGB":
        sub_bands = [0,1,2]
    elif d_src == "RGN":
        sub_bands = [0,1,6]
    elif d_src == "all":
        sub_bands = [0,1,2,3,4,5,6,7,8,9]
    elif d_src in ["S1", "S1_200m"]:
        sub_bands = [0,1]
    return sub_bands

def main():
    # Parameters
    se=1
    euclid= 1
    d_src = "S1_200m"
    sub_bands = get_sub_bands(d_src)
    n_channels = len(sub_bands)
    if se == 1:
        se_fn = "se"
    else:
        se_fn = ""
    if euclid == 1:
        euc_fn = "diff"
    else:
        euc_fn = ""
    city = sys.argv[1]
    voter = sys.argv[2]
    year = sys.argv[3]
    num_votes = sys.argv[4]

    db_path = f'votes/db/{city}/{city}_voterAI.db'
    tbl_name = f'{city}_image_pairs_{num_votes}'
    base_model = f"{city}_voter_{voter}"
    model_name = f"{base_model}_ensembleVoter"
    out_tbl = f'{base_model}_votes_{num_votes}'
    sql = f"select * from " + tbl_name
    con, vf = retrieve_db(db_path, sql)
    model_dir = "/home/jovyan/private/slums/data/PARTIMAP/trained_models/voter/"+model_name+"_"+d_src+"_1_dcai.ckpt"
    model = tf.saved_model.load(model_dir)

    preds = []
    confs = []
    print(f"Total image pairs to be voted: {len(vf)}")
    # vf[['prediction', 'wins', 'lose']] = vf.apply(vote, axis=1,
    #                                               d_src=d_src,
    #                                               sub_bands=sub_bands,
    #                                               model=model)
    for row in vf.itertuples():
        img_l = load_image_viz(city, d_src, year, row.left, sub_bands)
        img_r = load_image_viz(city, d_src, year, row.right, sub_bands)
        predictions = model([img_l, img_r]).numpy()
        pred = np.argmax(predictions)
        prob = predictions[0]
        preds.append(pred)
        confs.append(prob[pred])
        print(f"left: {row.left} vs right: {row.right}, winner: {pred}, matches left: {len(vf)-len(preds)}", end='\r')
        if ((len(preds) % 1000000) == 0):
            pred_col = pd.Series(preds)
            conf_col = pd.Series(confs)
            vf['prediction'] = pred_col
            vf['conf'] = conf_col
            vf.to_sql(out_tbl, con = con, if_exists= 'replace')
            print(f"saved {len(preds)} votes")

    pred_col = pd.Series(preds)
    conf_col = pd.Series(confs)
    vf['prediction'] = pred_col
    vf['conf'] = conf_col
    vf.to_sql(out_tbl, con = con, if_exists= 'replace')
    print(f"saved {len(preds)} votes")

    def get_winner_loser_stochastic(row, threshold=0.02):  # Added threshold parameter
        if 0.5 - threshold <= row.prediction <= 0.5 + threshold:
            if random.random() < 0.5:  # 50/50 chance
                return pd.Series([row.left, row.right])
            else:
                return pd.Series([row.right, row.left])
        elif row.prediction < 0.5:
            return pd.Series([row.left, row.right])
        else:
            return pd.Series([row.right, row.left])
        
    vf.to_sql(out_tbl, con = con, if_exists= 'replace')

if __name__ == "__main__":
    main()
