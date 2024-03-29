from anndata import AnnData

clusters_to_merge = [
    {'0': 'A'}, {'1': 'A'}, {'2': 'A'}, {'19': 'B'}
]

def run(**kwargs):

    adata = kwargs.get('adata')
    clust_col = adata.obs.columns[-1:][0]
    cluster_list = kwargs.get('cluster_list')

    adata.obs[clust_col] = adata.obs[clust_col].astype(str)

    for cluster_map in clusters_to_merge:
        for old_cluster, new_cluster in cluster_map.items():
            adata.obs[clust_col] = adata.obs[clust_col].replace(old_cluster, new_cluster)
    adata.obs[clust_col] = adata.obs[clust_col].astype('category')

    return {'adata': adata}
