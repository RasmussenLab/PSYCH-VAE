
from sklearn.preprocessing import power_transform
import umap
from scipy.stats.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pandas as pd
from sklearn.decomposition import PCA
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

import os, sys
import numpy as np 

def read_cat(file):
   data = np.load(file)
   data = data.astype(np.float32)
   data_input = data.reshape(data.shape[0], -1)
   data_label = np.argmax(data, 2)
   data_label = data_label.astype('float')
   data_label[data.sum(2) == 0] = -1
   data_label = data_label + 1
   
   return data, data_label

def read_con(file):
   data = np.load(file)
   data = data.astype(np.float32)
   
   return data

def read_header(file, mask=None):
   with open(file, "r") as f:
      h = list()
      for line in f:
         h.append(line.rstrip())
   
   if not mask is None:
      h = np.array(h)
      h = h[mask]
   
   return h

def remove_not_obs_cat(pheno, ind, h, p=0.01):
   pheno = pheno[:,~np.all(ind == ind[0,:], axis = 0)]
   h = np.array(h)[~np.all(ind == ind[0,:], axis = 0)]
   ind = ind[:,~np.all(ind == ind[0,:], axis = 0)]
   
   ind_tmp = np.copy(ind)
   ind_tmp[ind_tmp == 1] = 0
   pheno = pheno[:,np.count_nonzero(ind_tmp, 0) > (pheno.shape[0] * p)]
   h = h[np.count_nonzero(ind_tmp, 0) > (pheno.shape[0] * p)]
   ind = ind[:,np.count_nonzero(ind_tmp, 0) > (pheno.shape[0] * p)]
   
   return pheno, ind, h

def remove_not_obs_ordinal(pheno, ind, h, p=0.01):
   pheno = pheno[:,~np.all(ind == ind[0,:], axis = 0)]
   h = np.array(h)[~np.all(ind == ind[0,:], axis = 0)]
   ind = ind[:,~np.all(ind == ind[0,:], axis = 0)]
   
   ind_tmp = np.copy(ind)
   ind_tmp[ind_tmp == 1] = 0
   pheno = pheno[:,np.count_nonzero(ind_tmp, 0) > (pheno.shape[0] * p)]
   h = h[np.count_nonzero(ind_tmp, 0) > (pheno.shape[0] * p)]
   ind = ind[:,np.count_nonzero(ind_tmp, 0) > (pheno.shape[0] * p)]
   
   return pheno, ind, h

def encode_con(raw_input, p = 0.01):
   
   matrix = np.array(raw_input)
   tmp_matrix = np.copy(matrix)
   tmp_matrix[np.isnan(tmp_matrix)] = 0
   
   # remove less than 1% observations
   mask_col = np.count_nonzero(tmp_matrix, 0) > (tmp_matrix.shape[0] * p)
   
   #data_input = np.log2(matrix + 0.1)
   #data_input = matrix
   scaler = MinMaxScaler()
   data_input = scaler.fit_transform(matrix)
   #data_input = power_transform(matrix, method='yeo-johnson', standardize=True)
   
   # remove 0 variance
   std = np.nanstd(data_input, axis=0)
   mask_col &= std != 0
   
   # scale around -3 to 3
   #scaler = MinMaxScaler((-3,3)) 
   #data_scaled = scaler.fit_transform(data_input)
   
   # z-score normalize
   #mean = np.nanmean(data_input, axis=0)
   #std = np.nanstd(data_input, axis=0)
   
   #data_input -= mean
   #data_input /= std
   
   #data_input[np.isnan(data_input)] = 0
   mean = np.nanmean(data_input, axis = 0)
   data_input = np.where(np.isnan(matrix), mean, data_input)
   data_input = data_input[:,mask_col]
   
   return data_input, mask_col

def encode_binary(raw_input, p = 0.01):
   
   matrix = np.array(raw_input)
   tmp_matrix = np.copy(matrix)
   tmp_matrix[np.isnan(tmp_matrix)] = 0
   
   # remove less than 1% observations
   mask_col = np.count_nonzero(tmp_matrix, 0) > (tmp_matrix.shape[0] * p)
   data_input = tmp_matrix
   
   # remove 0 variance
   std = np.nanstd(data_input, axis=0)
   mask_col &= std != 0
   data_input = data_input[:,mask_col]
   
   data_input[data_input != 0] = 1
   
   data_input[np.isnan(matrix[:,mask_col])] = np.nan
   
   data_input = encode_cat(data_input, num_classes = 2, uniques = [0,1], na = np.nan)
   data_label = np.argmax(data_input, 2)
   data_label = data_label.astype('float')
   data_label[data_input.sum(2) == 0] = -1
   data_label = data_label + 1
   
   return data_input, data_label, mask_col

def encode_cat(raw_input, num_classes=None, uniques=None, na='NA'):
   matrix = np.array(raw_input)
   n_labels = matrix.shape[1]
   n_samples = matrix.shape[0]
   
   # make endocding dict
   encodings = defaultdict(dict)
   count = 0
   no_unique = 0
   
   if uniques is None:
      no_unique = 1
      encodings = defaultdict(dict)
      for lab in range(0,n_labels):
         uniques = np.unique(matrix[:,lab])
         uniques = sorted(uniques)
         num_classes = len(uniques[uniques != na])
         count = 0
         for u in uniques:
            if u == na:
               encodings[lab][u] = np.zeros(num_classes)
               continue
            encodings[lab][u] = np.zeros(num_classes)
            encodings[lab][u][count] = 1
            count += 1
   else:
      for u in uniques:
         if u == na:
            encodings[u] = np.zeros(num_classes)
            continue
         encodings[u] = np.zeros(num_classes)
         encodings[u][count] = 1
         count += 1
   
   # encode the data
   data_input = np.zeros((n_samples,n_labels,num_classes))
   i = 0
   for patient in matrix:
      
      data_sparse = np.zeros((n_labels, num_classes))
      
      count = 0
      for lab in patient:
         if no_unique == 1:
            data_sparse[count] = encodings[count][lab]
         else:
            if lab != na:
               lab = int(float(lab))
            data_sparse[count] = encodings[lab]
         count += 1
      
      data_input[i] = data_sparse
      i += 1
      
   return data_input.astype(np.float32)

def concat_cat_list(cat_list):
  n_cat = 0
  cat_shapes = list()
  first = 0
 
  for cat_d in cat_list:
    cat_shapes.append(cat_d.shape)
    cat_input = cat_d.reshape(cat_d.shape[0], -1)
   
    if first == 0:
      cat_all = cat_input
      del cat_input
      first = 1
    else:
      cat_all = np.concatenate((cat_all, cat_input), axis=1)
 
  # Make mask for patients with no measurments
  catsum = cat_all.sum(axis=1)
  mask = catsum > 1
  del catsum
  return cat_shapes, mask, cat_all

def concat_cat_list_ordinal(cat_list):
  n_cat = 0
  cat_shapes = list()
  first = 0
 
  for cat_d in cat_list:
    #cat_shapes.append(cat_d.shape)
    #cat_input = cat_d.reshape(cat_d.shape[0], -1)
    cat_input = np.argmax(cat_d, 2)
   
    if first == 0:
      cat_all = cat_input
      del cat_input
      first = 1
    else:
      cat_all = np.concatenate((cat_all, cat_input), axis=1)
 
  # Make mask for patients with no measurments
  catsum = cat_all.sum(axis=1)
  mask = catsum > 1
  del catsum
  return cat_shapes, mask, cat_all


def concat_con_list(con_list, mask=[]):
  n_con_shapes = []
 
  first = 0
  for con_d in con_list:
    con_d = con_d.astype(np.float32)
    n_con_shapes.append(con_d.shape[1])
     
    if first == 0:
      con_all = con_d
      first = 1
    else:
      con_all = np.concatenate((con_all, con_d), axis=1)
 
  consum = con_all.sum(axis=1)
  if len(mask) == 0:
      mask = consum != 0
  else:
      mask &= consum != 0
  del consum
  return n_con_shapes, mask, con_all

def get_categories(sorted_data):
   sorted_data_cat = np.copy(sorted_data.astype(str))
   mask = list()
   for col in range(sorted_data.shape[1]):
      tmp = sorted_data[:,col]
      
      if len(tmp[tmp != 0]) == 0:
         mask.append(False)
         sorted_data_cat[:,col] = tmp 
         continue
      
      mask.append(True)
      new_cat = np.array(pd.qcut(tmp[tmp != 0], 4, labels=False, duplicates = 'drop'))
      new_cat = new_cat + 1
      tmp[tmp != 0] = new_cat
      tmp_new = np.copy(tmp).astype(str)
      sorted_data_cat[:,col] = tmp
      
   return sorted_data_cat, mask

final_ids = list()
file = path + "/data_encoded/included_all.txt"
with open(file, "r") as r:
  for line in r:
     final_ids.append(line.rstrip())

from datetime import datetime
def years_between(d1, d2):
    d1 = datetime.strptime(d1, "%d/%m/%Y")
    d2 = datetime.strptime(d2, "%d/%m/%Y")
    return abs((d2 - d1).days)/365

# Get age
age = {}
file = '/faststorage/jail/project/Register/FromCrome/phenotype2016d.csv'
with open(file, "r") as f:
    header = f.readline().rstrip()
    header = header.split(",")
    for line in f: 
      if len(line.rstrip()) == 0:
         continue
      tmp = line.rstrip().split(",")
      
      if tmp[1] == '':
         print('isuse')
         continue
      
      age[tmp[0]] = years_between(tmp[1],'01/01/2016')
      

ages_final = []

for ids in final_ids:
   if ids in age:
      ages_final.append(age[ids])
   else:
      print('issue')
      ages_final.append(np.nan)


np.save('/faststorage/jail/project/gentofte_projects/prediction/included_patient_age.npy', np.array(ages_final))