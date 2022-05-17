# from cuml.cluster.dbscan import DBSCAN
from sklearn.cluster import DBSCAN

def denoise(x):
    clustering = DBSCAN(eps=0.05, min_samples=10).fit(x)  # 0.1 10 are perfect but slow
    close = clustering.labels_[x.argmax(axis=0)[2]]
    denoised_pc = x[clustering.labels_ == close]
    return denoised_pc