import scanpy as sc


def run(**kwargs):

    path = kwargs.get('image_path')
    adata = sc.read_h5ad(path)
    # on merge multiple files
    # if 'batch_key' not in adata.uns:
    #     adata.uns['batch_key'] = 'fov'

    return {'adata': adata}
