import squidpy as sq
import scanpy as sc


def cluster(adata, spatial_weight=0.0, resolution=1.0, method='leiden'):
    # Args:
    # adata: The anndata after preprocessing and dimensionality reduction.
    # spatial_weight: Weight of spatial edges compared to genetic edges
    # resolution: The resolution of the modularity cost function. Lower is less clusters, higher is more clusters.
    # method: The method by which we cluster data. Louvain, Leiden, TODO:: spectral Louvain, spectral Leiden

    # Clustering
    adjacency = adata.obsp['connectivities']

    # Force spatial neighbors to be close.
    if spatial_weight > 0:
        sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True)
        adjacency += adata.obsp['spatial_connectivities'] * spatial_weight

    # Scanpy
    if method == 'leiden':
        sc.tl.leiden(adata, resolution=resolution, adjacency=adjacency)
    elif method == 'louvain':
        sc.tl.louvain(adata, resolution=resolution, adjacency=adjacency)

    return adata


def run(**kwargs):
    adata = kwargs.get('adata')

    swgt = kwargs.get('spatial_weight')
    method = kwargs.get('method')
    res = kwargs.get('resolution')

    return {'adata': cluster(adata, swgt, res, method)}
