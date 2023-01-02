# Annotation format conversion
- Exemple to convert xml to txt:

```python
from AnnotationFormatConverter import vocxml_to_yolotxt

# define class name to id dict
class_name_to_id_mapping = {"smoke": 0,
                            "fire": 1}

# get the annotations
annotation_dir = 'annotations_xml/'
saved_dir = 'annotations_txt/val/'
annotations = glob.glob(annotation_dir + '*.xml')
annotations.sort()

# convert and save the annotations
for ann in tqdm.tqdm(annotations):
    info_dict = vocxml_to_yolotxt.extract_info_from_xml(ann)
    vocxml_to_yolotxt.convert_to_yolo_txt(saved_dir, info_dict, class_name_to_id_mapping)
```

- Exemple to convert txt to xml:

(script only work when image and correspondant annotation file are in a same folder).

```python
from AnnotationFormatConverter import yolotxt_to_vocxml

# define id to class mapping dict
id_to_class_name_mapping = {0: "smoke",
                            1: "fire"}

# define dataset directory that contain images and labels in the same folder
data_dir = 'data_dir'
saved_dir = 'annotation_xml'

yolotxt_to_vocxml.convert_to_voc_xml(id_to_class_name_mapping, data_dir, 'saved_dir')
```

# Image clustering

```python
from ImageClustering import image_similarity

img_list = list(glob.glob('../datasets/benchmark/images/*'))

image_clusters, image_centroids, images_duplicate = image_similarity.cluster(img_list, sim_threshold=0.93, min_community_size=1, 
                                                    emb_batch_size=64, cluster_batch_size=128, size=320)
```

# TODO
- [ ] convert yolo to coco
- [ ] annotation sanity check
- [ ] dataset statistic
- [ ] go through dataset