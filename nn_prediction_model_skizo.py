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

matplotlib.use('agg')
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
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torch.utils.data.dataset import TensorDataset

from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from itertools import combinations
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from scipy import interp
import pickle
from itertools import cycle

## Functions

class ClassifierDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class, num_hidden1, num_hidden2):
        super(MulticlassClassification, self).__init__()
        
        self.layer_1 = nn.Linear(num_feature, num_hidden1)
        self.layer_2 = nn.Linear(num_hidden1, num_hidden2)
        #self.layer_3 = nn.Linear(64, 32)
        self.layer_out = nn.Linear(num_hidden2, num_class) 
        
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(num_hidden1)
        self.batchnorm2 = nn.BatchNorm1d(num_hidden2)
        #self.batchnorm3 = nn.BatchNorm1d(32)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        #x = self.layer_3(x)
        #x = self.batchnorm3(x)
        #x = self.relu(x)
        #x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc) * 100
    
    return acc, y_pred_softmax, y_pred_tags


def evaluate_model_nn(predictions, probs, test_labels, labels_name, f_name=None, colors=None):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_labels, predictions, average='micro')
    results['precision'] = precision_score(test_labels, predictions, average='micro')
    #results['roc'] = roc_auc_score(test_labels, probs)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    probs_list = []
    mcc = dict()
    n_classes = probs.shape[1]
    y = label_binarize(test_labels, classes=np.unique(test_labels))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], probs[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        #mcc[i] = matthews_corrcoef(test_labels[test_labels == i], predictions[test_labels == i])
        results[i] = roc_auc[i]
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), np.array(probs).ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    results['roc'] =  roc_auc["micro"]
    
    if f_name != None:
        plt.figure(figsize = (12, 10))
        plt.style.use('seaborn-whitegrid')
        plt.rcParams['font.size'] = 14
        lw = 2
        plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc["micro"])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.style.use('seaborn-whitegrid')
        plt.savefig(f_name + '.pdf', format = 'pdf', dpi = 800)
    
    # Plot test across classes
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    if f_name != None:
        # Plot all ROC curves
        plt.figure(figsize = (12, 10))
        plt.style.use('seaborn-whitegrid')
        plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc["micro"])
        
        s_names = np.sort(labels_name)
        for n, color in zip(s_names, colors):
            i = list(labels_name).index(n)
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(n, roc_auc[i]))
        
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.style.use('seaborn-whitegrid')
        plt.savefig(f_name + '_all.pdf', format = 'pdf', dpi = 800)
        
    return results

#path = "/data/projects/IBP/rosa/VAE_V1/"
path = "/data/projects/IBP/to_transfer_to_computerome2/VAE_rosa/"
sys.path.append(path + "people/rosal/scripts/")
del sys.modules["intVAE_v1_5"]
import intVAE_v1_5
from plots import embedding_plot_discrete, embedding_plot_float, plot_error
from read_files import encode_binary, encode_cat, encode_con, remove_not_obs_cat, remove_not_obs_ordinal, read_cat, read_con, read_header

path = "/data/projects/IBP/to_transfer_to_computerome2/VAE_rosa/data/"
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

# remove collacted measurements
h_to_remove = ['age_F1000', 'age_F2000', 'age_F2101', 'age_F3000', 'age_F3001', 'age_F3101', 'age_F4100', 'age_F4300', 'age_F5100', 'age_F5200', 'age_F6100', 'age_F7000', 'age_F8100', 'age_F8101', 'age_F8300', 'age_F9100', 'age_F9200','age_F9297']
tmp_h = [i for i in F_pheno_h if i not in h_to_remove]
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
mbr[:,4]  = mbr[:,4] / 1000
mbr = mbr[skitzo_class != 0]
mbr, mask = encode_con(mbr, 0.01)
mbr_h = read_header(path + "/data_encoded/phenotypes_age/mbr_con_headers_age.txt", mask)
mbr_h = np.delete(mbr_h, 3)
mbr = np.delete(mbr, 3, axis=1)

tmp_raw = read_con(path + "/data_encoded/input/mbr_con_age.npy")
tmp_raw[:,4]  = tmp_raw[:,4] / 1000
tmp_raw = tmp_raw[:,mask]
tmp_raw = np.delete(tmp_raw, 3, axis=1)
con_all_raw = np.concatenate((con_all_raw, tmp_raw), axis=1)

LPR = read_con(path + "/data_encoded/input/other_LPR_con.npy")
LPR = LPR[skitzo_class != 0]
#LPR, LPR_input, mask = encode_binary(LPR, 0.01)
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

# combine MBR and sibling
#MBR_sibling = np.concatenate((MBR_pheno, sibling_pheno), axis=1)
#MBR_sibling_h = np.concatenate((MBR_pheno_h, sibling_pheno_h))

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
#f_LPR,mask = encode_con(f_LPR, 0.01)
f_LPR_h = read_header(path + "/data_encoded/phenotypes_age/father_LPR_headers_con.txt", mask)

m_LPR = read_con(path + "/data_encoded/input/mother_LPR_con.npy")
m_LPR = m_LPR[skitzo_class != 0]
m_LPR, m_LPR_input, mask = encode_binary(m_LPR, 0.01)
#m_LPR, mask = encode_con(m_LPR, 0.01)
m_LPR_h = read_header(path + "/data_encoded/phenotypes_age/mother_LPR_headers_con.txt", mask)

# combine parents and sibling
family_LPR = np.concatenate((f_LPR, m_LPR, sibling_pheno), axis=1)
family_LPR_h = np.concatenate((f_LPR_h, m_LPR_h, sibling_pheno_h))
family_LPR_input = np.concatenate((f_LPR_input, m_LPR_input, sibling_pheno_input), axis=1)

analysis_type = "skitzo"
version = "v2"

sns.set(font_scale=1.5)
plt.style.use('seaborn-whitegrid')

# Prepare data
mbr_geno = np.concatenate((MBR_pheno, hla_pheno, geno), axis=1)
mbr_geno_h = np.concatenate((MBR_pheno_h, hla_pheno_h, geno_h))
mbr_geno_input = np.concatenate((MBR_pheno_input, hla_pheno_input, geno_input), axis=1)

cat_names = np.concatenate((family_LPR_h, MBR_pheno_h, hla_pheno_h, geno_h))
con_names = np.concatenate((F_pheno_h, severity_pheno_h, mbr_h, LPR_h))
all = np.concatenate((F_pheno, severity_pheno, mbr, LPR, family_LPR_input,mbr_geno_input), axis=1)
data_df = pd.DataFrame(all, columns = np.concatenate((con_names,cat_names)))

labels = np.load(path + "/clustering/" + analysis_type + "/labels_kmeans_" + version + ".npy")
y = label_binarize(labels, classes=np.unique(labels))
n_classes = y.shape[1]

#labels_name = ["C-SCZ4", "C-SCZ1", "C-SCZ2", "C-SCZ6", "C-SCZ5", "C-SCZ7", "C-SCZ3"]
labels_name = ["C-SCZ3", "C-SCZ5", "C-SCZ2", "C-SCZ4", "C-SCZ1", "C-SCZ6", "C-SCZ7"]
old_labels = list(np.unique(labels))
labels_names = []
for l in labels:
   labels_names.append(labels_name[old_labels.index(l)])

# 30% examples in test data
RSEED = 42
train, test, train_labels, test_labels = train_test_split(data_df, labels.astype(int), 
                                                          stratify = labels.astype(int),
                                                          test_size = 0.2, 
                                                          random_state = RSEED)

np.save(path + "/prediction/" + analysis_type + "/nn_train_" + version + "_" + analysis_type + ".npy", train)
np.save(path + "/prediction/" + analysis_type + "/nn_train_label_" + version + "_" + analysis_type + ".npy", train_labels)
np.save(path + "/prediction/" + analysis_type + "/nn_test_" + version + "_" + analysis_type + ".npy", test)
np.save(path + "/prediction/" + analysis_type + "/nn_test_label_" + version + "_" + analysis_type + ".npy", test_labels)

# train = np.load(path + "/prediction/" + analysis_type + "/nn_train_" + version + "_" + analysis_type + ".npy")
# train_labels = np.load(path + "/prediction/" + analysis_type + "/nn_train_label_" + version + "_" + analysis_type + ".npy")
# test = np.load(path + "/prediction/" + analysis_type + "/nn_test_" + version + "_" + analysis_type + ".npy")
# test_labels = np.load(path + "/prediction/" + analysis_type + "/nn_test_label_" + version + "_" + analysis_type + ".npy")
# train = pd.DataFrame(train, columns = np.concatenate((con_names,cat_names)))
# test = pd.DataFrame(test, columns = np.concatenate((con_names,cat_names)))

EPOCHS = 300
BATCH_SIZE = 10
LEARNING_RATE = 0.001
NUM_FEATURES = len(data_df.columns)
NUM_CLASSES = len(np.unique(labels))

train_dataset = ClassifierDataset(torch.from_numpy(np.array(train)).float(), torch.from_numpy(train_labels).long())
val_dataset = ClassifierDataset(torch.from_numpy(np.array(test)).float(), torch.from_numpy(test_labels).long())

# weighted sampler
target_list = []
for _, t in train_dataset:
    target_list.append(t)

target_list = torch.tensor(target_list)
target_list = target_list[torch.randperm(len(target_list))]
class_count = np.bincount(labels.astype(int))
class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
class_weights_all = class_weights[target_list]
weighted_sampler = WeightedRandomSampler(
    weights=class_weights_all,
    num_samples=len(class_weights_all),
    replacement=True
)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          sampler=weighted_sampler
)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
#test_loader = DataLoader(dataset=test_dataset, batch_size=1)

cuda = False
device = torch.device("cuda" if cuda == True else "cpu")

model = MulticlassClassification(num_feature = NUM_FEATURES, num_class=NUM_CLASSES, num_hidden1= 256, num_hidden2 = 128)
model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}

for e in range(1, EPOCHS+1):
    
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        
        y_train_pred = model(X_train_batch)
        
        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc, softmax_pred, correct_pred = multi_acc(y_train_pred, y_train_batch)
        
        train_loss.backward()
        optimizer.step()
        
        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()
    
    
    # VALIDATION    
    with torch.no_grad():
        
        val_epoch_loss = 0
        val_epoch_acc = 0
        
        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            
            y_val_pred = model(X_val_batch)
                        
            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc, softmax_pred, correct_pred = multi_acc(y_val_pred, y_val_batch)
            
            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
    
    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))
    accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
    
    print('Epoch ' + str(e) + ' | Train Loss: ' + str(train_epoch_loss/len(train_loader)) + ' | Val Loss: ' + str(val_epoch_loss/len(val_loader)) + ' | Train Acc: ' + str(train_epoch_acc/len(train_loader)) + ' | Val Acc: ' + str(val_epoch_acc/len(val_loader)))

# Best model
filename = path + "/prediction/" + analysis_type + "/nn_model_" + version + "_" + analysis_type + "pt"
torch.save(model, filename)
np.save(path + "/prediction/" + analysis_type + "/nn_loss_" + version + "_" + analysis_type + ".npy", loss_stats)
np.save(path + "/prediction/" + analysis_type + "/nn_acc_" + version + "_" + analysis_type + ".npy", accuracy_stats)
# model= torch.load(filename)
# model.eval()

with torch.no_grad():
        val_epoch_loss = 0
        val_epoch_acc = 0
        predictions = []
        probs = []
        
        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            
            y_val_pred = model(X_val_batch)
                        
            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc, softmax_pred, correct_pred = multi_acc(y_val_pred, y_val_batch)
            
            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
            predictions.append(int(correct_pred))
            probs.append(np.array(softmax_pred).ravel())

val_loss = val_epoch_loss / len(val_loader)
val_acc = val_epoch_acc / len(val_loader)

colors = cycle(['lightskyblue','royalblue', 'darkblue', 'salmon', 'red', 'crimson', 'maroon'])
f_name = path + "/prediction/" + analysis_type + "/nn_roc_" + version  + "_" + analysis_type
test_eval = evaluate_model_nn(np.array(predictions), np.array(probs), test_labels, f_name, colors, labels_name)
mcc_all = matthews_corrcoef(test_labels, np.array(predictions))

fi = pd.DataFrame({'feature': list(train.columns),
                   'importance': best_model.feature_importances_}).\
                    sort_values('importance', ascending = False)

feature_imp = fi[fi['importance'] > 0]
f_name = path + "/prediction/" + analysis_type + "/feature_importance_" + version  + "_" + analysis_type

sns.set(font_scale=1.5)
fig = plt.figure(figsize=(14,10))
g = sns.barplot(data=feature_imp.iloc[0:25], y='importance', x='feature')
# Add labels to your graph
plt.ylabel('Feature Importance Score')
plt.xlabel('Features')
g.set_xticklabels(g.xaxis.get_majorticklabels(), rotation=90)
fig.subplots_adjust(bottom=0.4)
plt.savefig(f_name + ".pdf", format = 'pdf', dpi = 800)

##### Remove data after diagnosis ####

first_age_scz = con_all_raw[:,list(F_pheno_h).index("age_F2001")]
first_age_mdd1 = con_all_raw[:,list(F_pheno_h).index("age_F3200")]
first_age_mdd2 = con_all_raw[:,list(F_pheno_h).index("age_F3300")]
#stacked_arrays = np.stack((first_age_mdd1,first_age_mdd2))
#first_age_mdd =  np.min(stacked_arrays, axis=0)
first_age = []
for i in range(len(first_age_scz)):
    if first_age_scz[i] == 0:
        if first_age_mdd1[i] == 0 and first_age_mdd2[i] == 0:
            first_age.append(100)
        else:
            a = np.array([first_age_mdd1[i], first_age_mdd2[i]])
            first_age.append(np.min(a[np.nonzero(a)]))
    else:
        first_age.append(first_age_scz[i])

all_raw = np.concatenate((con_all_raw, family_LPR_input, mbr_geno_input), axis=1)
data_df_raw = pd.DataFrame(all_raw, columns = np.concatenate((con_names,cat_names)))

filtered = []
tmp_data = data_df_raw[np.concatenate((F_pheno_h, LPR_h))]
for j in range(len(first_age_scz)):
    p = tmp_data.loc[j,:]
    p[p >= first_age[j]] = 0
    filtered.append(np.array(p))

filtered_enc, mask_enc = encode_con(np.array(filtered), 0.0001)
filtered_h = np.concatenate((F_pheno_h, LPR_h))[mask_enc]
data_filtered_h = np.concatenate((filtered_h, mbr_h, family_LPR_h, mbr_geno_h))
data_filtered = np.concatenate((filtered_enc, mbr, family_LPR_input, mbr_geno_input), axis=1)
data_df_filtered = pd.DataFrame(data_filtered, columns = data_filtered_h)

RSEED = 42
train_filtered, test_filtered, train_labels_filtered, test_labels_filtered = train_test_split(data_df_filtered, labels.astype(int), 
                                                          stratify = labels.astype(int),
                                                          test_size = 0.2, 
                                                          random_state = RSEED)

train_filtered, val_filtered, train_labels_filtered, val_labels_filtered = train_test_split(train_filtered, train_labels_filtered, 
                                                          stratify = train_labels_filtered,
                                                          test_size = 0.1, 
                                                          random_state = RSEED)

np.save(path + "/prediction/" + analysis_type + "/nn_train_filtered_" + version + "_" + analysis_type + ".npy", train_filtered)
np.save(path + "/prediction/" + analysis_type + "/nn_train_label_filtered_" + version + "_" + analysis_type + ".npy", train_labels_filtered)
np.save(path + "/prediction/" + analysis_type + "/nn_test_filtered_" + version + "_" + analysis_type + ".npy", test_filtered)
np.save(path + "/prediction/" + analysis_type + "/nn_test_label_filtered_" + version + "_" + analysis_type + ".npy", test_labels_filtered)
np.save(path + "/prediction/" + analysis_type + "/nn_val_filtered_" + version + "_" + analysis_type + ".npy", val_filtered)
np.save(path + "/prediction/" + analysis_type + "/nn_val_label_filtered_" + version + "_" + analysis_type + ".npy", val_labels_filtered)


path = '/faststorage/jail/project/gentofte_projects/VAE_rosa/data/'
# train_filtered = np.load(path + "/prediction/" + analysis_type + "/nn_train_filtered_" + version + "_" + analysis_type + ".npy")
# train_labels_filtered = np.load(path + "/prediction/" + analysis_type + "/nn_train_label_filtered_" + version + "_" + analysis_type + ".npy")
# test_filtered = np.load(path + "/prediction/" + analysis_type + "/nn_test_filtered_" + version + "_" + analysis_type + ".npy")
# test_labels_filtered = np.load(path + "/prediction/" + analysis_type + "/nn_test_label_filtered_" + version + "_" + analysis_type + ".npy")
# val_filtered = np.load(path + "/prediction/" + analysis_type + "/nn_val_filtered_" + version + "_" + analysis_type + ".npy")
# val_labels_filtered = np.load(path + "/prediction/" + analysis_type + "/nn_val_label_filtered_" + version + "_" + analysis_type + ".npy")
# train_filtered = pd.DataFrame(train_filtered, columns = data_filtered_h)
# test_filtered = pd.DataFrame(test_filtered, columns = data_filtered_h)
# val_filtered = pd.DataFrame(val_filtered, columns = data_filtered_h)

EPOCHS = 200
NUM_FEATURES = len(data_df_filtered.columns)
NUM_CLASSES = len(np.unique(labels))

train_dataset_filtered = ClassifierDataset(torch.from_numpy(np.array(train_filtered)).float(), torch.from_numpy(train_labels_filtered).long())
val_dataset_filtered = ClassifierDataset(torch.from_numpy(np.array(val_filtered)).float(), torch.from_numpy(val_labels_filtered).long())
test_dataset_filtered = ClassifierDataset(torch.from_numpy(np.array(test_filtered)).float(), torch.from_numpy(test_labels_filtered).long())

# weighted sampler
target_list_filtered = []
for _, t in train_dataset_filtered:
    target_list_filtered.append(t)

target_list_filtered = torch.tensor(target_list_filtered)
target_list_filtered = target_list_filtered[torch.randperm(len(target_list_filtered))]
class_count = np.bincount(labels.astype(int))
class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
class_weights_all_filtered = class_weights[target_list_filtered]
weighted_sampler_filtered = WeightedRandomSampler(
    weights=class_weights_all_filtered,
    num_samples=len(class_weights_all_filtered),
    replacement=True)

val_loader_filtered = DataLoader(dataset=val_dataset_filtered, batch_size=1)
test_loader_filtered = DataLoader(dataset=test_dataset_filtered, batch_size=1)

cuda = False
device = torch.device("cuda" if cuda == True else "cpu")
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

#BATCH_SIZE = 20
#LEARNING_RATE = 0.001
val_loss_min = 1000
best_hyp = []

batch_sizes = [5, 10, 20, 25]
learning_rates = [0.001, 0.0001]
num_hiddens = [[64, 32], [128, 64], [256, 128]]

for BATCH_SIZE in batch_sizes:
    for LEARNING_RATE in learning_rates:
        for num_hidden in num_hiddens:
            num_hidden1 = num_hidden[0]
            num_hidden2 = num_hidden[1]
            
            train_loader_filtered = DataLoader(dataset=train_dataset_filtered,
                                    batch_size=BATCH_SIZE,
                                    sampler=weighted_sampler_filtered)
            
            model_filtered = MulticlassClassification(num_feature = NUM_FEATURES, num_class=NUM_CLASSES, num_hidden1=num_hidden1, num_hidden2=num_hidden2)
            model_filtered.to(device)
            
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            optimizer = optim.Adam(model_filtered.parameters(), lr=LEARNING_RATE)
            
            accuracy_stats_filtered = {'train': [],"val": []}
            loss_stats_filtered = {'train': [],"val": []}
            
            for e in range(1, EPOCHS+1):
                # TRAINING
                train_epoch_loss = 0
                train_epoch_acc = 0
                model_filtered.train()
                for X_train_batch, y_train_batch in train_loader_filtered:
                    X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
                    optimizer.zero_grad()
                    
                    y_train_pred = model_filtered(X_train_batch)
                    
                    train_loss = criterion(y_train_pred, y_train_batch)
                    train_acc, softmax_pred, correct_pred = multi_acc(y_train_pred, y_train_batch)
                    
                    train_loss.backward()
                    optimizer.step()
                    
                    train_epoch_loss += train_loss.item()
                    train_epoch_acc += train_acc.item()
                
                # VALIDATION    
                with torch.no_grad():
                    val_epoch_loss = 0
                    val_epoch_acc = 0
                    
                    model_filtered.eval()
                    for X_val_batch, y_val_batch in val_loader_filtered:
                        X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                        
                        y_val_pred = model_filtered(X_val_batch)
                                    
                        val_loss = criterion(y_val_pred, y_val_batch)
                        val_acc, softmax_pred, correct_pred = multi_acc(y_val_pred, y_val_batch)
                        
                        val_epoch_loss += val_loss.item()
                        val_epoch_acc += val_acc.item()
                
                if val_loss_min > (val_epoch_loss/len(val_loader_filtered)):
                    val_loss_min = val_epoch_loss/len(val_loader_filtered)
                    best_hyp = [BATCH_SIZE, LEARNING_RATE, num_hidden]
                    best_model_filtered = copy.deepcopy(model_filtered)
                
                loss_stats_filtered['train'].append(train_epoch_loss/len(train_loader_filtered))
                loss_stats_filtered['val'].append(val_epoch_loss/len(val_loader_filtered))
                accuracy_stats_filtered['train'].append(train_epoch_acc/len(train_loader_filtered))
                accuracy_stats_filtered['val'].append(val_epoch_acc/len(val_loader_filtered))
                
                print('Epoch ' + str(e) + ' | Train Loss: ' + str(train_epoch_loss/len(train_loader_filtered)) + ' | Val Loss: ' + str(val_epoch_loss/len(val_loader_filtered)) + ' | Train Acc: ' + str(train_epoch_acc/len(train_loader_filtered)) + ' | Val Acc: ' + str(val_epoch_acc/len(val_loader_filtered)))

# Best model
filename = path + "/prediction/" + analysis_type + "/nn_model_filtered_" + version + "_" + analysis_type + "pt"
torch.save(best_model_filtered, filename)
np.save(path + "/prediction/" + analysis_type + "/nn_loss_filtered_" + version + "_" + analysis_type + ".npy", loss_stats_filtered)
np.save(path + "/prediction/" + analysis_type + "/nn_acc_filtered_" + version + "_" + analysis_type + ".npy", accuracy_stats_filtered)
np.save(path + "/prediction/" + analysis_type + "/nn_best_hyp_" + version + "_" + analysis_type + ".npy", best_hyp)
# best_model_filtered = torch.load(filename)
# best_model_filtered.eval()
# best_hyp = np.load(path + "/prediction/" + analysis_type + "/nn_best_hyp_" + version + "_" + analysis_type + ".npy", allow_pickle=True)

with torch.no_grad():
        test_epoch_loss = 0
        test_epoch_acc = 0
        predictions_filtered = []
        probs_filtered = []
        
        best_model_filtered.eval()
        for X_val_batch, y_val_batch in test_loader_filtered:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            
            y_val_pred = best_model_filtered(X_val_batch)
                        
            test_loss = criterion(y_val_pred, y_val_batch)
            test_acc, softmax_pred, correct_pred = multi_acc(y_val_pred, y_val_batch)
            
            test_epoch_loss += test_loss.item()
            test_epoch_acc += test_acc.item()
            predictions_filtered.append(int(correct_pred))
            probs_filtered.append(np.array(y_val_pred).ravel())

test_loss_filtered = test_epoch_loss / len(test_loader_filtered)
test_acc_filtered = test_epoch_acc / len(test_loader_filtered)

colors = cycle(['lightskyblue','royalblue', 'darkblue', 'salmon', 'red', 'crimson', 'maroon'])
f_name = path + "/prediction/" + analysis_type + "/nn_roc_filtered_" + version  + "_" + analysis_type
test_eval_filtered = evaluate_model_nn(np.array(predictions_filtered), np.array(probs_filtered), test_labels_filtered, labels_name, f_name, colors)
mcc_all_filtered = matthews_corrcoef(test_labels_filtered, np.array(predictions_filtered))

# Get SHAP values
acc_diffs = []
mcc_diffs = []
auc_diffs = []
cluster_diffs_auc = defaultdict(list)
cluster_diffs_mcc = defaultdict(list)
for feature_index in range(NUM_FEATURES):
    
    new_data = np.copy(test_loader_filtered.dataset.X_data)
    new_data[:,feature_index] = 0
    new_data = torch.from_numpy(new_data)
    
    dataset = ClassifierDataset(new_data, torch.from_numpy(test_labels_filtered).long())
    
    new_loader = DataLoader(dataset=dataset, batch_size=1)
    
    with torch.no_grad():
        test_epoch_acc = 0
        test_epoch_acc_clust = defaultdict(list)
        probs_filtered_shap = []
        predictions_filtered_shap = []
        best_model_filtered.eval()
        for X_val_batch, y_val_batch in new_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            
            y_val_pred = best_model_filtered(X_val_batch)
            
            test_acc, softmax_pred, correct_pred = multi_acc(y_val_pred, y_val_batch)
            
            test_epoch_acc += test_acc.item()
            predictions_filtered_shap.append(int(correct_pred))
            probs_filtered_shap.append(np.array(softmax_pred).ravel())
    
    test_acc_filtered_shap = test_epoch_acc / len(test_loader_filtered)
    acc_diffs.append(np.abs(test_acc_filtered-test_acc_filtered_shap))
    
    test_mcc = matthews_corrcoef(test_labels_filtered, np.array(predictions_filtered_shap))
    mcc_diffs.append(np.abs(mcc_all_filtered-test_mcc))
    #mcc_diffs.append(mcc_all_filtered-test_mcc)
    test_eval_filtered_shap = evaluate_model_nn(np.array(predictions_filtered), np.array(probs_filtered), test_labels_filtered, labels_name)
    auc_diffs.append(np.abs(test_eval_filtered['roc'] - test_eval_filtered_shap['roc']))
    
    for c in np.unique(labels):
        c = int(c)
        cluster_diffs_auc[c].append(np.abs(test_eval_filtered[c] - test_eval_filtered_shap[c]))
        #cluster_diffs_mcc[c].append(np.abs(mcc_all[c] - mcc_all_shap[c]))

fi_filtered = pd.DataFrame({'feature': list(train_filtered.columns),
                   'importance': acc_diffs}).\
                    sort_values('importance', ascending = False)

np.save(path + "/prediction/" + analysis_type + "/nn_feature_importance_filtered_" + version  + "_" + analysis_type + ".npy",fi_filtered)

fi_filtered = pd.DataFrame({'feature': list(train_filtered.columns),
                   'importance': mcc_diffs}).\
                    sort_values('importance', ascending = False)

np.save(path + "/prediction/" + analysis_type + "/nn_feature_importance_filtered_mcc_" + version  + "_" + analysis_type + ".npy",fi_filtered)

#fi_filtered = np.load(path + "/prediction/" + analysis_type + "/nn_feature_importance_filtered_" + version  + "_" + analysis_type + ".npy", allow_pickle=True)
#fi_filtered = pd.DataFrame(fi_filtered, columns = ['feature', 'importance'])

# Fraction explained
# all with accuracy change over 1
feature_imp_filtered = fi_filtered[fi_filtered['importance'] > 0]

other_h = [i for i in filtered_h if i in LPR_h]
mental_h = [i for i in filtered_h if i in F_pheno_h]

data_names = [mental_h, other_h, np.concatenate((mbr_h,MBR_pheno_h)), family_LPR_h, geno_h, hla_pheno_h]
title_data = ['Psychiatric disorders', 'Other medical conditions', 'MBR', 'Family diagnoses', 'Genomics', 'HLA data']

colors_u = ['#EC1C1C','#E06161', '#FF9B9B', '#84C3F7', '#4387BF', '#2669A1']
#colors_u = ['tomato', 'coral', 'lightblue', 'azure', 'navy']
bar_colors = []
data_names = [mental_h, other_h, np.concatenate((mbr_h,MBR_pheno_h)), family_LPR_h, geno_h, hla_pheno_h]
for k in feature_imp_filtered.iloc[0:25]['feature']:
    j = 0
    for dn in data_names:
        if k in dn:
            bar_colors.append(colors_u[j])
            break
        j += 1

all_distributions = []
bar_colors_all = dict()
for i,dn in enumerate(data_names):
    tmp_h = [i for i in feature_imp_filtered['feature'] if i in (dn)]
    #print(len(tmp_h) / feature_imp_filtered.shape[0]* 100)
    tmp_df = feature_imp_filtered.iloc[np.where(np.isin(feature_imp_filtered['feature'],tmp_h))]
    #print(tmp_df.loc[:,'importance'].sum())
    #print(tmp_df.loc[:,'importance'].mean())
    all_distributions.append([len(tmp_h) / feature_imp_filtered.shape[0]* 100, tmp_df.loc[:,'importance'].sum(), tmp_df.loc[:,'importance'].mean()])
    bar_colors_all[title_data[i]] = colors_u[i]

f_name = path + "/prediction/" + analysis_type + "/nn_feature_importance_filtered_all_" + version  + "_" + analysis_type + '.txt'
imp = pd.DataFrame(np.array(all_distributions), index = title_data, columns = ['Percent of all', 'Sum of importance', 'Average importance'])
imp.to_csv(f_name)

tmp_pd = pd.DataFrame({'feature': list(imp.index),
                   'importance': imp['Average importance']}).\
                    sort_values('importance', ascending = False)
f_name = path + "/prediction/" + analysis_type + "/nn_feature_importance_filtered_all_" + version  + "_" + analysis_type + '.pdf'
fig = plt.figure(figsize=(14,10))
g = sns.barplot(data=tmp_pd,  y='importance', x='feature',  palette=bar_colors_all)
# Add labels to your graph
plt.ylabel('Average feature importance')
plt.xlabel('Dataset')
g.set_xticklabels(g.xaxis.get_majorticklabels(), rotation=90)
fig.subplots_adjust(bottom=0.4)
plt.savefig(f_name, format = 'pdf', dpi = 800)

feature_imp_filtered = fi_filtered[fi_filtered['importance'] > 0]
f_name = path + "/prediction/" + analysis_type + "/nn_feature_importance_filtered_" + version  + "_" + analysis_type

#bar_colors = cycle(bar_colors)
fig = plt.figure(figsize=(14,10))
plt.style.use('seaborn-whitegrid')
g = sns.barplot(data=feature_imp_filtered.iloc[0:25], y='importance', x='feature', palette=bar_colors)
# Add labels to your graph
plt.ylabel('Feature importance score')
plt.xlabel('Features')
plt.style.use('seaborn-whitegrid')
g.set_xticklabels(g.xaxis.get_majorticklabels(), rotation=90)
fig.subplots_adjust(bottom=0.4)
plt.savefig(f_name + ".pdf", format = 'pdf', dpi = 800)

# Confusion matrix
cmap = sns.diverging_palette(220, 20, sep=10, as_cmap=True)
sns.set(font_scale=1.5)
plt.style.use('seaborn-whitegrid')
f_name = path + "/prediction/" + analysis_type + "/nn_confusion_matrix_filtered_" + version  + "_" + analysis_type
conf_mat = confusion_matrix(test_labels_filtered, np.array(predictions_filtered))
conf_mat = pd.DataFrame(conf_mat, labels_name, labels_name)
conf_mat = conf_mat.sort_index()
conf_mat = conf_mat.T.sort_index()
fig = plt.figure(figsize=(14,10))
g = sns.heatmap(conf_mat, annot=True, annot_kws={"size": 16}, cmap=cmap, fmt = 'd', center = 0) # font size
bottom, top = g.get_ylim()
g.set_ylim(bottom + 0.5, top - 0.5)
plt.ylabel('True labels')
plt.xlabel('Predicted labels')
g.set_xticklabels(g.xaxis.get_majorticklabels(), rotation=30)
g.set_yticklabels(g.yaxis.get_majorticklabels(), rotation=0)
fig.subplots_adjust(bottom=0.2)
plt.savefig(f_name + ".pdf", format = 'pdf', dpi = 800)

### Distances in "new data"
between_cl_dist_raw = defaultdict(dict)
within_dists_raw = []
for c_1 in np.unique(labels):
   tmp_raw = data_filtered[labels == c_1,:]
    
   tmp_median_1_raw = np.mean(tmp_raw, axis = 0)
   
   #within_dists_raw.append(np.mean(euclideanR(tmp_raw,tmp_median_1_raw)))
   for c_2 in np.unique(labels):
      tmp_cos_2_raw = data_filtered[labels == c_2, :]
      tmp_median_2_raw = np.mean(tmp_cos_2_raw, axis = 0)
      
      dist_tmp_raw = distance.correlation(tmp_median_1_raw, tmp_median_2_raw)
      between_cl_dist_raw[c_1][c_2] = dist_tmp_raw

corr = pd.DataFrame(between_cl_dist_raw)
corr.index = labels_name
corr.columns = labels_name
corr = corr.sort_index()
corr = corr.T.sort_index()
cmap = sns.diverging_palette(220, 10, sep=5, as_cmap=True)
sns.set(font_scale=1.4)
f, ax = plt.subplots()
g = sns.clustermap(corr, cmap=cmap, center=0, xticklabels = True,
                   yticklabels = True,
                   linewidths=0, row_cluster=True, col_cluster=True, metric='correlation', figsize = (10, 10))
bottom, top = g.ax_heatmap.get_ylim()
#g.ax_heatmap.set_ylim(bottom + 0.5, top - 0.5)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
g.ax_heatmap.set_xticklabels(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=60)
f.subplots_adjust(right=0.8, bottom=0.2)
f.tight_layout()
plt.savefig(path + analysis_type + "_filtered_cluster_heatmap_dists_raw_" + version + ".pdf", bbox_inches='tight', format = 'pdf', dpi = 800)

corr = pd.DataFrame(between_cl_dist_raw)
corr.index = labels_name
corr.columns = labels_name
corr = corr.sort_index()
corr = corr.T.sort_index()
cmap = sns.diverging_palette(220, 10, sep=1, as_cmap=True)
sns.set(font_scale=1.4)
f, ax = plt.subplots()
g = sns.clustermap(corr, cmap=cmap, center=0, xticklabels = True,
                   yticklabels = True,
                   linewidths=0, row_cluster=False, col_cluster=False, metric='correlation', figsize = (10, 10), vmin=0, vmax=0.26)
bottom, top = g.ax_heatmap.get_ylim()
#g.ax_heatmap.set_ylim(bottom + 0.5, top - 0.5)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
g.ax_heatmap.set_xticklabels(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=60)
f.subplots_adjust(right=0.8, bottom=0.2)
f.tight_layout()
plt.savefig(path + analysis_type + "_filtered_cluster_heatmap_dists_raw_same_scale_" + version + ".pdf", bbox_inches='tight', format = 'pdf', dpi = 800)


#### Get dataset SHAP
# Get SHAP values

data_names = [mental_h, other_h, np.concatenate((mbr_h,MBR_pheno_h)), family_LPR_h, geno_h, hla_pheno_h]
title_data = ['Psychiatric disorders', 'Other medical conditions', 'MBR', 'Family diagnoses', 'Genomics', 'HLA data']

acc_diffs_v2 = []
mcc_diffs_v2 = []
auc_diffs_v2 = []
cluster_diffs_auc_v2 = defaultdict(list)
for dn in data_names:
    
    new_data = np.copy(test_loader_filtered.dataset.X_data)
    new_data[:,np.where(np.isin(data_filtered_h, dn))] = 0
    new_data = torch.from_numpy(new_data)
    
    dataset = ClassifierDataset(new_data, torch.from_numpy(test_labels_filtered).long())
    
    new_loader = DataLoader(dataset=dataset, batch_size=1)
    
    with torch.no_grad():
        test_epoch_acc = 0
        test_epoch_acc_clust = defaultdict(list)
        probs_filtered_shap = []
        predictions_filtered_shap = []
        best_model_filtered.eval()
        for X_val_batch, y_val_batch in new_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            
            y_val_pred = best_model_filtered(X_val_batch)
            
            test_acc, softmax_pred, correct_pred = multi_acc(y_val_pred, y_val_batch)
            
            test_epoch_acc += test_acc.item()
            predictions_filtered_shap.append(int(correct_pred))
            probs_filtered_shap.append(np.array(softmax_pred).ravel())
    
    test_acc_filtered_shap = test_epoch_acc / len(test_loader_filtered)
    acc_diffs_v2.append(np.abs(test_acc_filtered-test_acc_filtered_shap))
    
    test_mcc = matthews_corrcoef(test_labels_filtered, np.array(predictions_filtered_shap))
    mcc_diffs_v2.append(np.abs(mcc_all_filtered-test_mcc))
    #mcc_diffs.append(mcc_all_filtered-test_mcc)
    test_eval_filtered_shap = evaluate_model_nn(np.array(predictions_filtered), np.array(probs_filtered), test_labels_filtered, labels_name)
    auc_diffs_v2.append(np.abs(test_eval_filtered['roc'] - test_eval_filtered_shap['roc']))
    
    for c in np.unique(labels):
        c = int(c)
        cluster_diffs_auc_v2[c].append(np.abs(test_eval_filtered[c] - test_eval_filtered_shap[c]))

fi_filtered_v2 = pd.DataFrame({'feature': title_data,
                   'importance': acc_diffs_v2}).\
                    sort_values('importance', ascending = False)

np.save(path + "/prediction/" + analysis_type + "/nn_feature_importance_filtered_each_" + version  + "_" + analysis_type + ".npy",fi_filtered_v2)

#fi_filtered_v2 = np.load(path + "/prediction/" + analysis_type + "/nn_feature_importance_filtered_each_" + version  + "_" + analysis_type + ".npy")

f_name = path + "/prediction/" + analysis_type + "/nn_feature_importance_filtered_each_" + version  + "_" + analysis_type
fig = plt.figure(figsize=(14,10))
plt.style.use('seaborn-whitegrid')
g = sns.barplot(data=fi_filtered_v2,  y='importance', x='feature',  palette=bar_colors_all)
# Add labels to your graph
plt.style.use('seaborn-whitegrid')
plt.ylabel('Average feature importance')
plt.xlabel('Dataset')
g.set_xticklabels(g.xaxis.get_majorticklabels(), rotation=90)
fig.subplots_adjust(bottom=0.4)
plt.savefig(f_name + ".pdf", format = 'pdf', dpi = 800)

fi_filtered_v2 = pd.DataFrame({'feature': title_data,
                   'importance': mcc_diffs_v2}).\
                    sort_values('importance', ascending = False)
np.save(path + "/prediction/" + analysis_type + "/nn_feature_importance_filtered_each_mcc" + version  + "_" + analysis_type + ".npy",fi_filtered_v2)
