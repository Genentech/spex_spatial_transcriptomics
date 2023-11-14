import numpy as np
import pandas as pd
import squidpy as sq
from multiprocessing import Pool
import glob


# Utilize complex numbers to vectorize permutation counts
def unique_perms(a):
    weight = 1j * np.arange(0, a.shape[0])
    b = a + weight[:, np.newaxis]
    u, cts = np.unique(b, return_counts=True)
    return u, cts


# Mapping functions for parallelization
def process_neighborhood(n):
    global t_perms, t_clust, tcell_perms
    ncv = np.zeros((t_perms + 1, t_clust), dtype=np.float32)

    if len(n) == 0:
        return ncv

    j, cts = unique_perms(tcell_perms[:, n])
    ncv[np.imag(j).astype(np.int32), np.real(j).astype(np.int32)] = cts

    return ncv


def pool_ncvs(argperm, argclust, argcperms):
    global t_perms, t_clust, tcell_perms
    t_perms, t_clust, tcell_perms = argperm, argclust, argcperms


# Optimized CLQ using vectorized operations
def CLQ_vec(adata, clust_col='leiden', clust_uniq=None, radius=50, n_perms=1000):
    # Calculate spatial neighbors once.
    adata.obsm['spatial'] = adata.obs[['x_coordinate', 'y_coordinate']].to_numpy()

    sq.gr.spatial_neighbors(adata, coord_type='generic', radius=radius)
    neigh_idx = adata.obsp['spatial_connectivities'].tolil()
    neighborhoods = [x + [i] for i, x in
                     enumerate(neigh_idx.rows)]  # Append self to match previous implementation? (x + [i])

    # Global frequencies.
    global_cluster_freq = adata.obs[clust_col].value_counts(normalize=True)
    if clust_uniq is not None:
        null_clusters = global_cluster_freq.index.difference(pd.Index(clust_uniq))
        for c in null_clusters:
            global_cluster_freq[c] = 0

    # Cluster identities for each cell
    cell_ids = adata.obs.loc[:, clust_col]

    # Map clusters to integers for fast vectorization in numpy
    label_dict = {x: i for i, x in enumerate(global_cluster_freq.index)}
    n_clust = len(label_dict)

    # Permute cluster identities across cells
    cell_id_perms = [[label_dict[x] for x in cell_ids]]  # 0th permutation is the observed NCV
    cell_id_perms.extend([[label_dict[x] for x in np.random.permutation(cell_ids)] for i in range(n_perms)])
    cell_id_perms = np.array(cell_id_perms)

    # Calculate neighborhood content vectors (NCVs).
    p = Pool(initializer=pool_ncvs, initargs=[n_perms, n_clust, cell_id_perms])
    temp = p.map(process_neighborhood, [n for n in neighborhoods])
    p.close()
    p.join()

    ncv = np.array(temp).transpose((1, 0, 2))
    norm_ncv = ncv / (ncv.sum(axis=2)[:, :, np.newaxis] + 1e-99)

    # Old single-threaded version.
    '''
    ncv = np.zeros((n_perms+1,n_cells,n_clust),dtype=np.float32)
    for i,cell_neighborhood in enumerate(neighborhoods):
        if len(cell_neighborhood) == 0:
            continue

        j,cts = unique_perms(cell_id_perms[:,cell_neighborhood])
        ncv[np.imag(j).astype(np.int32),i,np.real(j).astype(np.int32)] = cts
    '''

    # Read out local CLQ from NCV vectors
    local_clq = norm_ncv / np.array([global_cluster_freq[x] for x in label_dict])

    # Average local_clq over clusters to get global CLQ
    global_clq = np.array(
        [np.nanmean(local_clq[cell_id_perms == label_dict[x], :].reshape(n_perms + 1, -1, n_clust), 1) for x in
         label_dict])

    # Read out the observed local and global CLQs
    idx = [x for x in label_dict]
    lclq = pd.DataFrame(local_clq[0, :, :], columns=idx, index=adata.obs_names)
    gclq = pd.DataFrame(global_clq[:, 0, :], index=idx, columns=idx)
    ncv = pd.DataFrame(ncv[0, :, :], index=adata.obs_names, columns=idx)

    # Permutation test
    clq_perm = (global_clq[:, 1:, :] < global_clq[:, 0, :].reshape(n_clust, -1, n_clust)).sum(1) / n_perms
    clq_perm = pd.DataFrame(clq_perm, index=idx, columns=idx)

    adata.obsm['NCV'] = ncv
    adata.obsm['local_clq'] = lclq
    adata.uns['CLQ'] = {'global_clq': gclq, 'permute_test': clq_perm}

    return adata


def run(**kwargs):
    # after_phenograph_clusters on full data per image
    adata = kwargs.get('adata')

    clust_col = adata.obs.columns[-1:][0]  # column where is clustered
    radius = kwargs.get('radius')  # int  0-50-500
    n_perms = kwargs.get('n_perms')  # int  1-1000
    clust_uniq = adata.obs['cluster_phenograph'].unique()  # set(clust_col) clust_col.unique()

    return {'adata': CLQ_vec(adata, clust_col, clust_uniq, radius, n_perms)}
