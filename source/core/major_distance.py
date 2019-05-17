import numpy as np
import scipy

def major_min_distance(input_embeddings, blacklist_embeddings):
    min_indices = np.array([])
    min_distances = np.array([])
    if (input_embeddings.size == 0) and (blacklist_embeddings.size == 0):
        distance_matrix = scipy.spatial.distance.cdist(input_embeddings, blacklist_embeddings)
        min_indices = np.argmin(distance_matrix, axis = 1)
        min_distances = distance_matrix[np.arange(len(min_indices)), min_indices]
        min_distances = min_indices.squeeze()
    return min_indices, min_distances


def match(input_embeddings, blacklist_embeddings, labels):
    min_indices, min_distances = major_min_distance(input_embeddings, blacklist_embeddings)
    print(min_indices.shape)
    top_ids = [labels[i] for i in min_indices.T]
    return top_ids, min_distances
