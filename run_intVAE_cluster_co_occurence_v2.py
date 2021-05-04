#!/usr/bin/env python

from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn import cluster
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import power_transform
import umap
from scipy.stats.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances, pairwise_distances
from scipy import stats
from scipy.spatial import distance
from statsmodels.stats.multitest import multipletests
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import random
import copy
import scipy
from scipy import stats
plt.style.use('seaborn-whitegrid')

import os, sys
import torch
import numpy as np
from torch.utils import data
import re

from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from itertools import combinations
import math

path = ""
sys.path.append(path + "")
import intVAE_v1_5
from plots import embedding_plot_discrete, embedding_plot_float, plot_error
from read_files import encode_binary, encode_cat, encode_con, remove_not_obs_cat, remove_not_obs_ordinal, read_cat, read_con, read_header

### Functions
def mahalanobisR(X,meanCol,IC):
    m = []
    for i in range(X.shape[0]):
        m.append(distance.mahalanobis(X[i,:],meanCol,IC) ** 2)
    return(m)

def euclideanR(X,meanCol):
    m = []
    for i in range(X.shape[0]):
        m.append(distance.euclidean(X[i,:],meanCol))
    return(m)

class Point:
   def __init__(self,initx,inity):
      self.x = initx
      self.y = inity
   def getX(self):
      return self.x
   def getY(self):
      return self.y
   def __str__(self):
      return "x=" + str(self.x) + ", y=" + str(self.y)
   def distance_from_point(self,the_other_point):
      dx = the_other_point.getX() - self.x
      dy = the_other_point.getY() - self.y
   def slope(self,other_point):
      if self.x - other_point.getX() == 0 :
         return 0
      else:
         panta = (self.y - other_point.getY())/ (self.x - other_point.getX())
         return panta
   def distance_to_line(self, p1, p2):
      x_diff = p2.x - p1.x
      y_diff = p2.y - p1.y
      num = abs(y_diff*self.x - x_diff*self.y + p2.x*p1.y - p2.y*p1.x)
      den = math.sqrt(y_diff**2 + x_diff**2)
      return num / den

def k_mean_eval(data, range_n_clusters=range(2,16)):
   averaged = []
   wcss = []
   for n_clusters in range_n_clusters:
      clusterer = KMeans(n_clusters=n_clusters, random_state=42)
      #clusterer = GaussianMixture(n_clusters, random_state=10, covariance_type="full")
      #clusterer = SpectralClustering(n_clusters=n_clusters, random_state=10)
      clusterer = clusterer.fit(data)
      cluster_labels = clusterer.predict(data)
      wcss.append(clusterer.inertia_)
      silhouette_avg = silhouette_score(data, cluster_labels)
      averaged.append(silhouette_avg)
      
   distances = []
   p1 = Point(initx=np.min(range_n_clusters),inity=wcss[0])
   p2 = Point(initx=np.max(range_n_clusters),inity=wcss[len(range_n_clusters) - 1])
   for i in range(0,len(range_n_clusters) - 1):
      p = Point(initx=i+1,inity=wcss[i])
      distances.append(p.distance_to_line(p1,p2))
   
   return distances,wcss,averaged

def gmm_eval(data, range_n_clusters=range(2,16)):
   aics = []
   bics = []
   for n_clusters in range_n_clusters:
      clusterer = GaussianMixture(n_clusters, random_state=42)
      clusterer = clusterer.fit(data)
      cluster_labels = clusterer.predict(data)
      aics.append(clusterer.aic(data))
      bics.append(clusterer.bic(data))
      
   distances_1 = []
   p1 = Point(initx=np.min(range_n_clusters),inity=aics[0])
   p2 = Point(initx=np.max(range_n_clusters),inity=aics[len(range_n_clusters) - 1])
   for i in range(0,len(range_n_clusters) - 1):
      p = Point(initx=i+1,inity=aics[i])
      distances_1.append(p.distance_to_line(p1,p2))
   
   distances_2 = []
   p1 = Point(initx=np.min(range_n_clusters),inity=bics[0])
   p2 = Point(initx=np.max(range_n_clusters),inity=bics[len(range_n_clusters) - 1])
   for i in range(0,len(range_n_clusters) - 1):
      p = Point(initx=i+1,inity=bics[i])
      distances_2.append(p.distance_to_line(p1,p2))
   
   return aics,bics,distances_1,distances_2

path = ''
## load pheno data categorical
F_pheno_cat, F_pheno_input = read_cat(path + "/data_encoded/input/pheno_F_int.npy")
F_pheno_h_cat = read_header(path + "/data_encoded/phenotypes_age/pheno_F_headers_age.txt")

# select for depression
skitzo = F_pheno_cat[:,list(F_pheno_h_cat).index("age_F2000")]
skitzo_class = np.argmax(skitzo, 1)
dep = F_pheno_cat[:,list(F_pheno_h_cat).index("age_F3000")]
dep_class = np.argmax(dep, 1)

F_pheno = read_con(path + "/data_encoded/input/pheno_F_con.npy")
F_pheno = F_pheno[skitzo_class != 0]
F_pheno, mask_F = encode_con(F_pheno, 0.01)
F_pheno_h = read_header(path + "/data_encoded/phenotypes_age/pheno_F_headers_con.txt", mask_F)

tmp_raw = read_con(path + "/data_encoded/input/pheno_F_con.npy")
con_all_raw = tmp_raw[:,mask_F]

tmp_h = [i for i in F_pheno_h if not i.startswith('age_F')]
other_LPR = F_pheno[:,np.where(np.isin(F_pheno_h,tmp_h))].reshape(F_pheno.shape[0],len(tmp_h))
other_LPR_h = F_pheno_h[np.where(np.isin(F_pheno_h,tmp_h))]
other_raw = con_all_raw[:,np.where(np.isin(F_pheno_h,tmp_h))].reshape(con_all_raw.shape[0],len(tmp_h))

tmp_h = [i for i in F_pheno_h if i.startswith('age_F')]
F_pheno =  F_pheno[:,np.where(np.isin(F_pheno_h,tmp_h))].reshape(F_pheno.shape[0],len(tmp_h))
con_all_raw = con_all_raw[:,np.where(np.isin(F_pheno_h,tmp_h))].reshape(con_all_raw.shape[0],len(tmp_h))
F_pheno_h = F_pheno_h[np.where(np.isin(F_pheno_h,tmp_h))]

## load in continuous pheno data
severity_pheno = read_con(path + "/data_encoded/input/sev_con.npy")
severity_pheno = severity_pheno[skitzo_class != 0]
severity_pheno, mask = encode_con(severity_pheno, 0.01)
severity_pheno_h = read_header(path + "/data_encoded/phenotypes_age/sev_con_headers.txt", mask)

tmp_raw = read_con(path + "/data_encoded/input/sev_con.npy")
con_all_raw = np.concatenate((con_all_raw, tmp_raw[:,mask]), axis=1)

mbr = read_con(path + "/data_encoded/input/mbr_con_age.npy")
mbr = mbr[skitzo_class != 0]
mbr, mask = encode_con(mbr, 0.01)
mbr_h = read_header(path + "/data_encoded/phenotypes_age/mbr_con_headers_age.txt", mask)

tmp_raw = read_con(path + "/data_encoded/input/mbr_con_age.npy")
tmp_raw = tmp_raw[:,mask]
con_all_raw = np.concatenate((con_all_raw, tmp_raw), axis=1)

LPR = read_con(path + "/data_encoded/input/other_LPR_con.npy")
LPR = LPR[skitzo_class != 0]
LPR, mask = encode_con(LPR, 0.01)
LPR_h = read_header(path + "/data_encoded/phenotypes_age/other_LPR_headers_con.txt", mask)
LPR = np.concatenate((LPR, other_LPR), axis=1)
LPR_h = np.concatenate((LPR_h, other_LPR_h))

tmp_raw = read_con(path + "/data_encoded/input/other_LPR_con.npy")
con_all_raw = np.concatenate((con_all_raw, tmp_raw[:,mask]), axis=1)
con_all_raw = np.concatenate((con_all_raw, other_raw), axis=1)
con_all_raw = con_all_raw[skitzo_class != 0]
## load pheno data categorical

MBR_pheno, MBR_pheno_input = read_cat(path + "/data_encoded/input/mbr_cat_age.npy")
MBR_pheno = MBR_pheno[skitzo_class != 0]
MBR_pheno_input = MBR_pheno_input[skitzo_class != 0]
MBR_pheno_h = read_header(path + "/data_encoded/phenotypes_age/mbr_cat_headers_age.txt")
MBR_pheno, MBR_pheno_input, MBR_pheno_h = remove_not_obs_cat(MBR_pheno, MBR_pheno_input, MBR_pheno_h, 0.01)

sibling_pheno, sibling_pheno_input = read_cat(path + "/data_encoded/input/sibling_cat.npy")
sibling_pheno = sibling_pheno[skitzo_class != 0]
sibling_pheno_input = sibling_pheno_input[skitzo_class != 0]
sibling_pheno_h = read_header(path + "/data_encoded/phenotypes_age/sibling_cat_headers.txt")
sibling_pheno, sibling_pheno_input, sibling_pheno_h = remove_not_obs_cat(sibling_pheno, sibling_pheno_input, sibling_pheno_h, 0.01)
sibling_pheno = np.compress((sibling_pheno!=0).sum(axis=(0,1)), sibling_pheno, axis=2)

## load in genotype
geno, geno_input = read_cat(path + "/data_encoded/input/genotypes_all.npy")
geno = geno[skitzo_class != 0]
geno_input = geno_input[skitzo_class != 0]
geno_h = read_header(path + "/data_encoded/genomics/genotypes_headers_all.txt")
geno, geno_input, geno_h = remove_not_obs_ordinal(geno, geno_input, geno_h, 0.01)

hla_pheno, hla_pheno_input = read_cat(path + "/data_encoded/input/geno_hla.npy")
hla_pheno = hla_pheno[skitzo_class != 0]
hla_pheno_input = hla_pheno_input[skitzo_class != 0]
hla_pheno_h = read_header(path + "/data_encoded/genomics/geno_hla_headers.txt")
hla_pheno, hla_pheno_input, hla_pheno_h = remove_not_obs_ordinal(hla_pheno, hla_pheno_input, hla_pheno_h, 0.01)

# Load binary LPR diagnosis
f_LPR = read_con(path + "/data_encoded/input/father_LPR_con.npy")
f_LPR = f_LPR[skitzo_class != 0]
f_LPR, f_LPR_input, mask = encode_binary(f_LPR, 0.01)
f_LPR_h = read_header(path + "/data_encoded/phenotypes_age/father_LPR_headers_con.txt", mask)

m_LPR = read_con(path + "/data_encoded/input/mother_LPR_con.npy")
m_LPR = m_LPR[skitzo_class != 0]
m_LPR, m_LPR_input, mask = encode_binary(m_LPR, 0.01)
m_LPR_h = read_header(path + "/data_encoded/phenotypes_age/mother_LPR_headers_con.txt", mask)

# combine parents and sibling
family_LPR = np.concatenate((f_LPR, m_LPR, sibling_pheno), axis=1)
family_LPR_h = np.concatenate((f_LPR_h, m_LPR_h, sibling_pheno_h))
family_LPR_input = np.concatenate((f_LPR_input, m_LPR_input, sibling_pheno_input), axis=1)

## Initiate variables 
cuda = False
device = torch.device("cuda" if cuda == True else "cpu")

con_list = [F_pheno, severity_pheno, mbr, LPR]
mbr_geno = np.concatenate((MBR_pheno, hla_pheno, geno), axis=1)
mbr_geno_h = np.concatenate((MBR_pheno_h, hla_pheno_h, geno_h))
mbr_geno_input = np.concatenate((MBR_pheno_input, hla_pheno_input, geno_input), axis=1)
cat_list_input = [family_LPR, mbr_geno]
cat_list = [family_LPR, MBR_pheno, hla_pheno, geno]
cat_db_names = ['Family LPR', 'MBR categorical', 'HLA phenotype', 'Genotype']
cat_names = np.concatenate((family_LPR_h, MBR_pheno_h, hla_pheno_h, geno_h))
con_names = np.concatenate((F_pheno_h, severity_pheno_h, mbr_h, LPR_h))
con_db_names = ['PSYK registry', 'Severity',  'MBR continuous', 'LPR']

all_data_names = pd.DataFrame(np.concatenate((cat_names, con_names)))
all_data_names.to_csv(path + "/included_names_SCZ.txt", sep = "\t")

nepochs=250
lrate=1e-3
kldsteps=[4, 6, 8, 10]
batchsteps=[100, 150, 200]
log_steps = [1, 5, 10] + list(range(25, 251, 25))

losses = list()
cat_loss = list()
con_loss = list()
ce = list()
sse = list()
KLD = list()
kld_w = 0
l = len(kldsteps)
r = 20/l
update = 1 + r
epochs = range(1, nepochs + 1)
count = 0
l_min = 100000
min_temp = 1
temp_rate = 0.003
temp = 0

## make data loader
mask, train_loader = intVAE_v1_5.make_dataloader(cat_list=cat_list_input, con_list=con_list, batchsize=5)

ncontinuous = train_loader.dataset.con_all.shape[1]
con_shapes = train_loader.dataset.con_shapes

ncategorical = train_loader.dataset.cat_all.shape[1]
cat_shapes = train_loader.dataset.cat_shapes

## Make model
model = intVAE_v1_5.VAE(ncategorical=ncategorical, ncontinuous=ncontinuous, con_shapes=con_shapes,
                        cat_shapes=cat_shapes, nhiddens=[800, 800], nlatent=40, beta=0.0001, dropout=0.1, cuda=cuda).to(device)

optimizer = optim.Adam(model.parameters(), lr=lrate)

## Run analysis
for epoch in range(1, nepochs + 1):
    
    if epoch in kldsteps:
      kld_w = 1/20 * update
      update += r
   
    if epoch in batchsteps:
      # Update loader
      train_loader = DataLoader(dataset=train_loader.dataset,
                                batch_size=int(train_loader.batch_size * 2),
                                shuffle=True,
                                drop_last=True,
                                num_workers=train_loader.num_workers,
                                pin_memory=train_loader.pin_memory)
    
    l, c, s, k, con_err, cat_err = model.enodeing(train_loader, epoch, lrate, kld_w, optimizer, temp)
    if epoch in log_steps:
      cat_loss.append(cat_err)
      con_loss.append(con_err)
      losses.append(l)
      ce.append(c)
      sse.append(s)
      KLD.append(k)
    

train_test_loader = DataLoader(dataset=train_loader.dataset, batch_size=train_loader.batch_size, drop_last=False,
                          shuffle=False, num_workers=1, pin_memory=train_loader.pin_memory)

latent, latent_var, cat_recon, cat_class, con_recon, loss, likelihood = model.latent(train_test_loader, kld_w, temp)

con_recon = np.array(con_recon)
con_recon = torch.from_numpy(con_recon)

analysis_type = "skizo"
version = "v1"

cat_loss = np.array(cat_loss)
con_loss = np.array(con_loss)
cols = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080']
plot_error(log_steps, cat_loss, con_loss, cat_db_names, con_db_names, cols, path, version = 'v1')

# Evaluation of results
cat_true_recon = []
cat_total_recon = []
pos = 0
missings = []
cat_shapes = [family_LPR.shape, MBR_pheno.shape, hla_pheno.shape, geno.shape]
for s in cat_shapes:
   n = s[1]
   cat_class_tmp = cat_class[:,pos:(n + pos)]
   cat_recon_tmp = cat_recon[:,pos:(n + pos)]
   
   missing_cat = cat_recon_tmp[cat_class_tmp == -1]
   diff_cat = cat_class_tmp - cat_recon_tmp
   
   diff_cat[diff_cat != 0] = -1
   true_cat = diff_cat[diff_cat == 0]
   false_cat = diff_cat[diff_cat != 0]
   cat_true = len(true_cat)/(float(diff_cat.size) - missing_cat.size)
   cat_true_recon.append(cat_true)
   
   # Calculate for each data point
   not_missing_counts = np.count_nonzero(cat_class_tmp != -1,axis=0)
   missings.append(not_missing_counts)
   ntrue_vals = np.count_nonzero(diff_cat == 0, 0)
   ntrue_vals = ntrue_vals.astype('float')
   ntrue_vals[not_missing_counts == 0] = 'nan'
   not_missing_counts = not_missing_counts.astype('float')
   not_missing_counts[not_missing_counts == 0] = 'nan'
   accuracy = ntrue_vals[not_missing_counts != 0] / not_missing_counts[not_missing_counts != 0]
   cat_total_recon.append(accuracy)
   pos += n

# Seperate in variables pr. dataset
total_shape = 0
true_recon = []
cos_values = []
all_values = []
for s in con_shapes:
   cor_con = list()
   cos_con = list()
   all_val = list()
   for n in range(0, (s-1)):
      data_subset = train_loader.dataset.con_all[:,total_shape:(s + total_shape - 1)]
      data_subset = data_subset[:,n]
      con_no_missing = data_subset[data_subset != 0]
      
      if len(con_no_missing) <= 1:
         all_val.append(np.nan)
         continue
      
      recon_subset = con_recon[:,total_shape:(s + total_shape - 1)]
      recon_subset = recon_subset[:,n]
      con_out_no_missing = recon_subset[data_subset != 0]
      cor = pearsonr(con_no_missing, con_out_no_missing)[0]
      cor_con.append(cor)
      com = np.vstack([con_no_missing, con_out_no_missing])
      cos = cosine_similarity(com)[0,1]
      cos_con.append(cos)
      all_val.append(cos)
   
   cor_con = np.array(cor_con)
   cos_con = np.array(cos_con)
   cos_values.append(cos_con)
   all_values.append(np.array(all_val))
   true_recon.append(len(cos_con[cos_con >= 0.9]) / len(cos_con))
   total_shape += s

# Distribution
df = pd.DataFrame(cat_total_recon + all_values, index = cat_db_names+con_db_names)
df_t = df.T

fig = plt.figure(figsize=(25,15))
ax = sns.boxplot(data=df_t, palette="colorblind", width=0.7)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
plt.ylabel('Reconstruction accuracy', fontsize=24)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)
fig.subplots_adjust(bottom=0.2)
plt.savefig(path + "/evaluation/recon_" + version + "_skitzo.png")
plt.close("all")

# UMAP
reducer = umap.UMAP()
embedding = reducer.fit_transform(latent)
np.save(path + "/clustering/" + analysis_type + "/embedding_UMAP_" + version + "_skitzo.npy", embedding)
plt.style.use('seaborn-whitegrid')

# clustering
n_clusts = range(2,16)
dists,wcss,silluet = k_mean_eval(latent, range_n_clusters=n_clusts)
best = n_clusts[np.argmax(dists)] - 1

# calcualte concensus clusters
repeat = 100
labels_list = []
for r in range(repeat):
   clr = KMeans(n_clusters=best)
   l = clr.fit_predict(latent)
   labels_list.append(l)

labels_list= np.array(labels_list)
cooccurrence_matrix = np.dot(labels_list.transpose(),labels_list)

# Compute cooccurrence matrix in percentage
cooccurrence_matrix_diagonal = np.diagonal(cooccurrence_matrix)
with np.errstate(divide='ignore', invalid='ignore'):
    cooccurrence_matrix_percentage = np.nan_to_num(np.true_divide(cooccurrence_matrix, cooccurrence_matrix_diagonal[:, None]))

clr = KMeans(n_clusters=best)

km = clr.fit(cooccurrence_matrix_percentage)
labels = km.predict(cooccurrence_matrix_percentage)
counts = np.bincount(labels)
labels = labels.astype(str)
np.save(path + "/clustering/" + analysis_type + "/labels_kmeans_" + version + ".npy", labels)

labels_name = ["C-SCZ3", "C-SCZ5", "C-SCZ2", "C-SCZ4", "C-SCZ1", "C-SCZ6", "C-SCZ7"]
old_labels = list(np.unique(labels))
labels_names = []
for l in labels:
   labels_names.append(labels_name[old_labels.index(l)])

palette={"C-SCZ1":'lightskyblue',"C-SCZ2":'royalblue',"C-SCZ3":'darkblue',"C-SCZ4":'salmon',"C-SCZ5":'red',"C-SCZ6":'crimson',"C-SCZ7":'maroon'}

embedding_plot_discrete(embedding, labels_names, "K-means clusters", path + "clustering/" + analysis_type + "/umap_kmeans_"  + analysis_type + "_" + version + ".png", palette=palette)

# PCA analysis
pca = PCA(n_components=40)
principalComponents = pca.fit_transform(all)

kmeans_raw = KMeans(n_clusters=best)
# Fitting the input data
kmeans_raw_pca = kmeans_raw.fit(principalComponents)
labels_raw = kmeans_raw.predict(principalComponents)
labels_raw_pca = labels_raw.astype(str)

reducer = umap.UMAP()
embedding_raw_pca = reducer.fit_transform(principalComponents)


embedding_plot_discrete(embedding_raw_pca, labels_raw_pca, "K-means clusters (PCA)", path + "clustering/" + analysis_type + "_raw/umap_kmeans_"  + analysis_type + "_" + version + "_pca.png")

## Plotting of measurments
type =  F_pheno[:,list(F_pheno_h).index("age_F3000")]
embedding_plot_float(embedding_raw_pca, type, "F30 (PCA)", path + "clustering/" + analysis_type + "_raw/umap_F30_"  + analysis_type + "_" + version + "_pca.png")

type = F_pheno[:,list(F_pheno_h).index("age_F2000")]
embedding_plot_float(embedding_raw_pca, type, "F20 (PCA)", path + "clustering/" + analysis_type + "_raw/umap_F20_"  + analysis_type + "_" + version + "_pca.png")
