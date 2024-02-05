import anndata
import numpy as np


def run(**kwargs):

    adata = anndata.read(kwargs.get('image_path'))
    try:
        adata.obsm['spatial'] = np.stack([adata.obs.x_centroid, adata.obs.y_centroid]).T
    except AttributeError:
        print('No x_centroid and y_centroid found.')
    finally:
        adata.obsm['spatial'] = adata.obsm['spatial'].astype('float')

    return {'adata': adata}
