# Annotation format conversion

This module is designed to facilitate the conversion of various annotation formats for bounding box-based object detection.

- Example to convert xml to txt:

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

- Example to convert txt to xml:

(script only work when image and correspondent annotation file are in a same folder).

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

This module utilizes sentence transformer models to measure image similarity, enabling image clustering and dataset splitting based on those similarities.

```python
from ImageClustering import image_similarity

# get list of images
images_dir = "data_neuroo/train/images/"
images_path = sorted(os.listdir(images_dir))

# do image clustering
image_clusters, image_centroids, images_duplicate = image_similarity.cluster(root=images_dir, images_list=images_path,, sim_threshold=0.93,  
                                                    min_community_size=1, emb_batch_size=64, cluster_batch_size=128, size=320)
```

# Datatset statistic

You can utilize this module to display statistical information about your dataset.

```python
from DatasetStatistic import dataset_stistics

# define class names
names = {0: '0',
        1: '1', 
        2: '2',
        3: '3'}

# get list of labels
labels_dir = "train/labels/"
labels_path = sorted(os.listdir(labels_dir))

labels = []

# read labels
for i, label_file in enumerate(labels_path):
    with open(labels_dir + label_file, 'r') as fl:
        lines = fl.readlines()
    fl.close()

    for line in lines:
        line = line.split(" ")
        line = [float(i) for i in line]
        labels.append(line)
    
labels = np.array(labels)

# plot labels statistic
plot_labels_statistic(labels, names)
```

# Dataset split

This module facilitates the splitting of datasets for object detection while preserving the class proportions. It treats the problem as a multi-label split. However, it's worth noting that in object detection tasks, there might be multiple instances of the same object in one image (e.g., multiple persons in an image). Therefore, this approach is considered an "approximation" since it doesn't fully account for such multi-instance scenarios.

```python
from DatasetSplit import approximation_multi_label_data_split

# list label files
labels_dir = "train/labels/"
labels_path = sorted(os.listdir(labels_dir))

# split train/test
train_index, test_index = approximation_multi_label_data_split.split(labels_path=labels_path, labels_dir=labels_dir, n_splits=7, shuffle=True, random_state=0)
```

# TODO
- [ ] convert yolo to coco
- [ ] annotation sanity check
- [ ] dataset statistic
- [ ] go through dataset
- [ ] boobs annotator
- [ ] bbox clustering