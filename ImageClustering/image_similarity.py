from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import os
import tqdm

def cluster(images_list: str, sim_threshold: float, min_community_size: int, emb_batch_size: int, cluster_batch_size: int, size: int):
    """Function that find embed all image to the embedding spac, then try to regroup all the images that are closer than sim_threshold.
    Returns only communities that are larger than min_community_size. To find duplicated images, set min_community_size to 1.

    Args:
        images_list (str): list of image files.
        sim_threshold (float): similarity threshold.
        min_community_size (int): minimum cluster size.
        emb_batch_size (int): batch size when doing image embedding. 
        cluster_batch_size (int): batch size when doing image clustering.
        size (int): resize image to this size for doing image embedding.

    Returns:
        list: list of image clusters.
        list: list of image centroids.
        list: list of duplicated images.
    """    
    # load the CLIP model
    model = SentenceTransformer('clip-ViT-B-32')

    # get all images name
    print("Images:", len(images_list))

    # compute the image embedding
    img_emb = model.encode([Image.open(filepath).resize((size, size)) for filepath in images_list], batch_size=emb_batch_size, convert_to_tensor=True, show_progress_bar=True)

    # get image indices cluster
    imgs_indice_cluster = util.community_detection(img_emb, threshold=sim_threshold, min_community_size=min_community_size, batch_size=cluster_batch_size)

    # get image cluster
    image_clusters = []
    for list_indices in imgs_indice_cluster:
        image_clusters.append([images_list[i] for i in list_indices])

    # get list of the centroid images
    image_centroids = [images_list[list_indices[0]] for list_indices in imgs_indice_cluster]

    # get list of the duplicate images
    images_duplicate = [image_name for image_name in images_list if image_name not in image_centroids]

    print("Image centroids: ", len(image_centroids))
    print("Duplicate images: ", len(images_duplicate))

    return image_clusters, image_centroids, images_duplicate