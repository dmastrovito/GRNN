import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from .config import config
from .model import GFR, PolynomialActivation

def format_param(name,vals):
    if name =='ds':
        return list(vals.astype(np.float32))
    elif name in ['a', 'b','poly_coeff']:
        return [list(vals.astype(np.float32))]
    elif type(vals) == np.ndarray and  len(vals) == 1:
        return float(vals[0])
    else:
        return vals
    
    



def read_file(fname):
    arr = []
    with open(fname, "r") as f:
        for x in f:
            arr.append(int(x))
    return arr

def append_to_file(fname, s):
    with open(fname, "a") as f:
        f.write(s)

def get_max_firing_rate(data):
    diffs = np.concatenate([np.diff(d["spike_times"]) for d in data])
    return np.max(1 / diffs) / 1000 # return in ms^-1

def get_activation_plot(actv, start=-100, end=270):
    currents = torch.linspace(start, end, steps=300).reshape(-1, 1)
    with torch.no_grad():
        fs = actv(currents)
    return currents.reshape(-1), fs.reshape(-1)

def plot_activation(Is, fs, actv):
    plt.figure(figsize=(8, 4.5), dpi=1000)
    plt.scatter(Is, fs)
    xs1, ys1 = get_activation_plot(actv, end=int(actv.max_current)+300)
    plt.plot(xs1, ys1, linewidth=2)
    plt.xlabel("$I$ $(pA)$")
    plt.ylabel("$g(I)$ $(ms^{-1})$")

def plot_kernel(model, cell_id, bin_size, save=False, fname=None, xlim=10):
    fig = plt.figure(constrained_layout=True, figsize=(8, 4.5), dpi=1000)
    subfigs = fig.subfigures(1, 2)
    xs = torch.linspace(0, xlim, 100)
    cs, ds = [], []
    with torch.no_grad():
        for x in xs:
            cs.append(kernel(model, x, var="a"))
            ds.append(kernel(model, x, var="b"))
    
    xs = xs * bin_size / 1000
    
    axs0 = subfigs[0].subplots(2)
    axs0[0].plot(xs,cs, linewidth=2)
    axs0[1].plot(xs,ds, linewidth=2)
    axs0[0].set_ylabel("$k_I(t)$")
    axs0[1].set_ylabel("$k_f(t)$")
    axs0[0].set_ylim([0, 3.4])
    axs0[1].set_ylim([-3.4, 0])
    axs0[1].set_xlabel("$t$ $(s)$")
    
    axs1 = subfigs[1].subplots(2)
    taus = np.array([10, 20, 50, 100, 200, 500, 1000, 2000])
    taus = taus[taus >= bin_size]
    taus = np.insert(taus, 0, 0)
    ks = [f"{i}" for i in taus]
    axs1[0].bar(ks, model.a.detach().reshape(-1))
    axs1[1].bar(ks, model.b.detach().reshape(-1))
    axs1[0].set_ylabel("$\\alpha_i$")
    axs1[1].set_ylabel("$\\beta_i$")
    axs1[1].set_xlabel("$\\tau_i$ (ms)")

    if save:
        plt.savefig(config["fig_save_path"] + f"{cell_id}/bin_size_{bin_size}/{fname}.png")
        plt.close()

def plot_predictions(model, Is, fs, bin_size, xlim=None):
    pred_fs, vs = model.predict(Is)
    pred_fs = torch.cat([torch.zeros(1), pred_fs])
    vs = torch.cat([torch.zeros(1, vs.shape[1]), vs])
    Is = torch.cat([torch.zeros(1), Is])
    fs = torch.cat([torch.zeros(1), fs])
    ts = np.arange(Is.shape[0]) * bin_size / 1000
    

    fig, axs = plt.subplots(2, figsize=(6, 2.5), dpi=100)
    
    axs[1].plot(ts, fs, label="Actual", linewidth=1)
    axs[1].plot(ts, pred_fs, label="Predicted", linewidth=1)
    axs[0].plot(ts, Is, linewidth=1)
    axs[1].legend()
    axs[1].set_ylabel("$f_t$ $(ms^{-1})$")
    axs[0].set_ylabel("$I_t$ $(pA)$")
    if xlim is not None:
        axs[0].set_xlim(xlim)
        axs[1].set_xlim(xlim)
    axs[0].set_ylim([-10, 230])
    axs[1].set_ylim([-0.003, 0.06])
    axs[0].xaxis.set_ticklabels([])
    axs[1].set_xlabel("$t$ $(s)$")
    fig.tight_layout()

def get_gfr_dataset(params, threshold=0.6):
    with open("model/labels.pickle", "rb") as f:
        labels = pickle.load(f)
    
    chosen_ids = filter(lambda x: params[x]["evr2"] > threshold, params.keys())
    
    dataset = {}
    for cell_id in chosen_ids:
        y = labels[cell_id]
        p = params[cell_id]["params"]
        
        a = p["a"].reshape(-1)
        b = p["b"].reshape(-1)
        pc = p["g"]["poly_coeff"].reshape(-1)
        gb = p["g"]["b"].reshape(-1)
        mc = p["g"]["max_current"].reshape(-1)
        mfr = p["g"]["max_firing_rate"].reshape(-1)
        x = torch.cat([a, b, pc, gb, mc, mfr])
        
        dataset[cell_id] = (x, y, params[cell_id]["evr2"])
        
    return dataset

# x: shape [batch_size, 28, 28]
# returns shape [batch_size, seq_length, in_dim]
def reshape_image(x, variant="p"):
    if variant == "p":
        x = x.reshape(x.shape[0], 24, 24)
        return x.reshape(x.shape[0], -1, 1)
    else:
        return x.reshape(x.shape[0], 28, 28)
    
# returns kernel of gfr neuron
def kernel(model, x, var="a"):
    a = model.a if var == "a" else model.b
    return torch.sum(a * torch.pow(1 - model.ds, x))

# initialize based on linear approximation of data
def activation_from_data(degree, max_current, max_firing_rate, bin_size, Is, fs):
    g = PolynomialActivation(degree, max_current, max_firing_rate, bin_size)
    
    x1, x2, y1, y2 = tuple([torch.tensor(0.0)] * 4)
    xs, ys = map(list, zip(*sorted(zip(Is.cpu(), fs.cpu()), key=lambda x: x[0])))
    i = np.argmax(ys)
    x2, y2 = xs[i], ys[i]
    for i in range(0, len(ys)):
        if ys[i] > 0.01:
            x1, y1 = (xs[i-1], ys[i-1]) if i - 1 > 0 else (xs[i], ys[i])
            break
            
    g.b = torch.nn.Parameter(x1.clone().reshape(1))
    poly_coeff = torch.randn(degree + 1) * 1e-1
    poly_coeff[1] = np.abs((y2 - y1) / (x2 - x1) * max_current)
    poly_coeff = poly_coeff.reshape(1, -1)
    g.poly_coeff = torch.nn.Parameter(poly_coeff)
    
    return g

def get_line_name(df, cell_id):
    return df[df["specimen__id"] == cell_id]["line_name"].to_numpy()[0]

def get_df(all_params, bin_size, actv_bin_size):
    params = all_params[(bin_size, actv_bin_size)]
    df2 = pd.read_csv("data/metadata.csv")
    d = {
        "cell_id": [], 
        "cre-line": [], 
        "bin_size": [], 
        "actv_bin_size": [], 
        "val_evr": [], 
        "test_evr": [],
        "train_loss": [],
        "test_loss": [],
        "params": []
    }
    for cell_id in params:
        p = params[cell_id]["params"]
        cell_type = get_line_name(df2, cell_id)
        val_evr = params[cell_id]["evr1"]
        test_evr = params[cell_id]["evr2"]
        train_loss = params[cell_id]["train_losses"][-1]
        test_loss = params[cell_id]["test_losses"][-1]

        d["cell_id"].append(cell_id)
        d["cre-line"].append(cell_type)
        d["bin_size"].append(bin_size)
        d["actv_bin_size"].append(actv_bin_size)
        d["val_evr"].append(val_evr)
        d["test_evr"].append(test_evr)
        d["train_loss"].append(train_loss)
        d["test_loss"].append(test_loss)
        d["params"].append(p)
    return pd.DataFrame.from_dict(d)

def generate_gfr_dataset():
    with open("model/best_params.pickle", "rb") as f:
        params = pickle.load(f)
    dataset = {}
    for bin_size, actv_bin_size in params:
        df = get_df(params, bin_size, actv_bin_size)
        df2 = df[(df['val_evr'] > 0.5) & (df['train_loss'] < 0.45)]
        print(f"{bin_size=}, {actv_bin_size=}, # cells: {len(df2)}")
        dataset[(bin_size, actv_bin_size)] = df2
    return dataset

def load_gfr_model(dataset, cell_id, bin_size, activation_bin_size):
    df = dataset[(bin_size, activation_bin_size)]
    if len(df[df["cell_id"] == cell_id]) == 0:
        raise Exception("Cell id not found")
    else:
        filtered_df = df[df["cell_id"] == cell_id]["params"]
        k = list(filtered_df.keys())[0]
        return GFR.from_params(filtered_df[k])
    
def to_json(dataset):
    a = []
    for bin_size, activation_bin_size in dataset:
        df = dataset[(bin_size, activation_bin_size)]
        for i in range(len(df)):
            x = df.iloc[i]
            p = x["params"]
            g = p["g"]
            d = {
                "cell_id": int(x["cell_id"]),
                "cre-line": x["cre-line"],
                "bin_size": int(x["bin_size"]),
                "actv_bin_size": int(x["actv_bin_size"]),
                "val_evr": float(x["val_evr"]),
                "test_evr": float(x["test_evr"]),
                "train_loss": float(x["train_loss"]),
                "test_loss": float(x["test_loss"]),
                "params": {
                    "a": p["a"],
                    "b": p["b"],
                    "ds": p["ds"],
                    "bin_size": p["bin_size"],
                    "g": {
                        "max_current": g["max_current"],
                        "max_firing_rate": g["max_firing_rate"],
                        "poly_coeff": g["poly_coeff"],
                        "b": g["b"],
                        "bin_size": g["bin_size"]
                    }
                }
            }
            a.append(d)
    return a

def df_from_json(json_file):
    pairs = [(10, 20), (10, 100), (20, 20), (20, 100), (50, 100), (100, 100)]
    def get_df(json_file, bin_size, actv_bin_size):
        data = json_file

        d = {
            "cell_id": [], 
            "cre-line": [], 
            "bin_size": [], 
            "actv_bin_size": [], 
            "val_evr": [], 
            "test_evr": [],
            "train_loss": [],
            "test_loss": [],
            "params": []
        }

        for x in data:
            if x["bin_size"] == bin_size and x["actv_bin_size"] == actv_bin_size:
                for key in x:
                    d[key].append(x[key])
        return pd.DataFrame.from_dict(d)
    return {(a, b): get_df(json_file, a, b) for a, b in pairs}


def load_labels(model_path, meta_path):
    with open(model_path, 'rb') as file:
        data = pickle.load(file)

    meta = pd.read_csv(meta_path)

    return pd.DataFrame(data), meta

def select_cells(meta, species = None, structure = None, layer = None, marker = None):
    '''
    species = 'Mus musculus'
    
    'structure_parent__acronym'
    'VISp','VISl','VISal','VISrl','VISli','VISpm','VISpl','VISam','VISpor','VISa'
    
    meta['structure__layer'].unique()
    array(['5', '4', '2/3', '6a', '6b', '1'], dtype=object)

    meta['line_name'].unique()
    array(['Oxtr-T2A-Cre', 'Pvalb-IRES-Cre',
       'Slc32a1-T2A-FlpO|Vipr2-IRES2-Cre', 'Cux2-CreERT2',
       'Scnn1a-Tg3-Cre', 'Htr3a-Cre_NO152', 'Rorb-IRES2-Cre',
       'Ctgf-T2A-dgCre', 'Nkx2-1-CreERT2', 'Vip-IRES-Cre',
       'Chrna2-Cre_OE25', 'Nos1-CreERT2|Sst-IRES-FlpO', 'Scnn1a-Tg2-Cre',
       'Nos1-CreERT2', 'Ntsr1-Cre_GN220', 'Esr2-IRES2-Cre',
       'Tlx3-Cre_PL56', 'Vipr2-IRES2-Cre',
       'Chrna2-Cre_OE25|Pvalb-T2A-Dre', 'Nr5a1-Cre', 'Ndnf-IRES2-dgCre',
       'Sst-IRES-Cre', 'Gad2-IRES-Cre', 'Chat-IRES-Cre-neo',
       'Slc17a6-IRES-Cre', 'Sim1-Cre_KJ18', 'Glt25d2-Cre_NF107',
       'Rbp4-Cre_KL100', 'Penk-IRES2-Cre-neo',
       'Htr3a-Cre_NO152|Pvalb-T2A-Dre', 'Esr2-IRES2-Cre|PhiC31-neo',
       'Gng7-Cre_KH71', 'Pvalb-T2A-FlpO|Vipr2-IRES2-Cre',
       'Pvalb-T2A-CreERT2', 'Esr2-IRES2-Cre-neo|PhiC31-neo',
       'Esr2-IRES2-Cre-neo', 'Vipr2-IRES2-Cre-neo'], dtype=object) 
    meta = meta[meta['donor__species'] == species]

    microns layer names:
    array(['L6CT', 'BC', 'MC', 'L3IT', 'L4IT', 'L2IT', 'L5IT', 'BPC', 'L6IT',
       'L6SP', 'NGC', 'L5ET', 'L5NP'], dtype=object)
    
'''
    if species is not None:
        meta = meta[meta['donor__species'] == species]
    if structure is not None:
        meta = meta[meta['structure_parent__acronym'] == structure]
    if layer is not None:
        meta = meta[meta['structure__layer'].str.contains(layer)]
    if marker is not None:
        meta = meta[meta['line_name'].isin(marker)]
    return meta.copy()

def get_params_by_cell_type(ids,params):
    gfr_params = [params.loc[id]['params'] for id in ids if id in params.index ]
    return gfr_params


def get_distribution(model):
    param_names = ['a', 'b','ds','bin_size']
    activation_function_params = ['max_current', 'max_firing_rate', 'poly_coeff', 'b','bin_size']
    param_dict = {}
    for name in param_names:
        param_dict[name] = np.vstack([model[i][name] for i in range(len(model))])
    param_dict['g'] = {}
    for name in activation_function_params:
        param_dict['g'][name] = np.vstack([model[i]['g'][name] for i in range(len(model))]) 
    return param_dict
    

    
def format_markers(markers):
    ms = [m if type(m) == tuple else [m] for m in markers]
    markers = ms
    return markers


def get_marker_dict(meta_path,model_path, cell_types,markers,bin_size, activation_bin_size,file=None,rebuild = False):
    if os.path.exists(file) and not rebuild:
        with open(file, 'rb') as f:
            param_dists = pickle.load(f)
        return param_dists

    model,meta = load_labels(model_path, meta_path)
    meta = meta.dropna(subset=['line_name'])
    model = model[(bin_size, activation_bin_size)].dropna()
    mouse_cells = select_cells(meta, species="Mus musculus")
    fmt_markers = format_markers(markers)
    cell_type_ids = [select_cells(mouse_cells,marker = m)['specimen__id'].tolist() for m in fmt_markers]
    param_names = ['a', 'b','ds','bin_size']
    activation_function_params = ['max_current', 'max_firing_rate', 'poly_coeff', 'b','bin_size']

    param_dists = {}
    for marker, cell_ids in zip(markers, cell_type_ids):
        print(marker)
        param_dists[marker] = {}
        param_dists[marker] =  {param_name: {} for param_name in param_names}
        param_dists[marker]['g'] = {param_name: {} for param_name in activation_function_params}
        params = get_params_by_cell_type(cell_ids, model)
        param_distributions = get_distribution(params)
        means = {k: np.mean(param_distributions[k], axis=0) for k in param_names}
        stds = {k: np.std(param_distributions[k], axis=0) for k in param_names}
        means['g'] = {k: np.mean(param_distributions['g'][k], axis=0) for k in activation_function_params}
        stds['g'] = {k: np.std(param_distributions['g'][k], axis=0) for k in activation_function_params}
        for k in param_names:
            param_dists[marker][k]['mean'] = means[k]
            param_dists[marker][k]['std'] = stds[k]
        for k in activation_function_params:
            param_dists[marker]['g'][k]['mean'] = means['g'][k]
            param_dists[marker]['g'][k]['std'] = stds['g'][k]
    
    if file is not None:
        with open(file, 'wb') as f:
            pickle.dump(param_dists, f)
    return param_dists


