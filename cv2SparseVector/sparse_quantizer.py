import numpy as np
from sklearn import cluster

def quantize(raster, n_colors, meta=True):
    width, height, depth = raster.shape
    reshaped_raster = np.reshape(raster, (width * height, depth))

    model = cluster.KMeans(n_clusters=n_colors)
    labels = model.fit_predict(reshaped_raster)
    palette = model.cluster_centers_
    if meta:
        meta_vector = []
        for kluster in range(len(palette)):
            print("kluster {}, num kluster{}, lab {}".format(kluster, list(labels).count(kluster), palette[kluster]))
            metas = list(palette[kluster]) + [list(labels).count(kluster)] + [kluster]
            meta_vector.append(metas)
    quantized_raster = np.reshape(palette[labels], (width, height, palette.shape[1]))
    return quantized_raster, meta_vector

