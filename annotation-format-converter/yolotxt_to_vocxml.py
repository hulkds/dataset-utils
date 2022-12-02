import os
import re
from PIL import Image
from tqdm import tqdm

def get_yolotxt_annotation(yolotxt_annotation_file: str) -> list:
    """get all good annotation from annotation txt file.
    Args:
        yolotxt_annotation_file (str): path to the yolotxt annotation file.
    Returns:
        list: list of annotation values: [class_id, x_yolo, y_yolo, yolo_width, yolo_height]
    """    
    # read the annotation file
    with open(yolotxt_annotation_file, 'r') as f:
        annotations = f.readlines()

        # remove all extra space 
        annotations = [x for x in annotations if x not in ['\n', ' ']] 

    f.close()

    # inner function that help to filter anomalies annotations, like negative values or characters in the annotation
    def is_number(n):
        try:
            float(n)
            return True
        except ValueError:
            return False

    good_annotation_split = []

    for annotation in annotations:
        good_annotation = True
        # regex to find the numbers in each line of the text file
        # remove any extra space from the end of the line
        annotation = re.split("\s", annotation.rstrip())

        # make sure the array has the correct number of items
        if len(annotation) == 5:
            for each_value in annotation:
                # If a value is not a positive number less than 1, then the format is not correct, return false
                if not is_number(each_value):
                    good_annotation = False
                    break
        else:
            good_annotation = False
            continue
        
        if good_annotation:
            good_annotation_split.append(annotation)
   
    return good_annotation_split


def convert_to_voc_xml(id_to_class_name_mapping: dict, data_dir: str, vocxml_save_dir: str):
    """convert yolotxt format to vocxml.
    Attention: script only work when image and annotation is in a same folder.
    Args:
        id_to_class_name_mapping (dict): python dict that map from id to class name.
        data_dir (str): directory to the dataset (image and annotation txt).
        vocxml_save_dir (str): directory to save converted vocxml annotaion files.
        keep_bad_annotation (bool, optional): whether to keep bad annotation or delete. Defaults to True.
    """    
    if data_dir[-1] != '/':
        data_dir = data_dir + '/'

    if not os.path.exists(vocxml_save_dir):
        os.makedirs(vocxml_save_dir)

    # loop all the txt annotation files     
    for file in tqdm(os.listdir(data_dir)):
        if file.endswith(".jpg") or file.endswith('jpeg'):
            # get the image file that match the annotation file
            image_file = file
            yolotxt_annotation_file = os.path.splitext(image_file)[0] + '.txt'
                
            if not os.path.exists(data_dir + yolotxt_annotation_file):
                os.unlink(data_dir + image_file)
                continue

            else:
                # get value from annotation file
                annotations = get_yolotxt_annotation(data_dir + yolotxt_annotation_file)

                # read image
                orig_img = Image.open(data_dir + image_file)  # open the image
                image_width = orig_img.width
                image_height = orig_img.height

                # Start the XML file
                xml_file = os.path.splitext(os.path.join(vocxml_save_dir, yolotxt_annotation_file))[0] + '.xml'
                with open(xml_file, 'w') as f:
                    f.write('<annotation>\n')
                    f.write('\t<folder>XML</folder>\n')
                    f.write('\t<filename>' + image_file + '</filename>\n')
                    f.write('\t<path>' + os.getcwd() + os.sep + image_file + '</path>\n')
                    f.write('\t<source>\n')
                    f.write('\t\t<database>Unknown</database>\n')
                    f.write('\t</source>\n')
                    f.write('\t<size>\n')
                    f.write('\t\t<width>' + str(image_width) + '</width>\n')
                    f.write('\t\t<height>' + str(image_height) + '</height>\n')
                    # assuming a 3 channel color image (RGB)
                    f.write('\t\t<depth>3</depth>\n')
                    f.write('\t</size>\n')
                    f.write('\t<segmented>0</segmented>\n')

                    for annotation in annotations:
                        # assign the variables
                        class_number = int(annotation[0])
                        try:
                            object_name = id_to_class_name_mapping[class_number]
                        except KeyError:
                            print("Invalid id, must be from ", id_to_class_name_mapping.keys())
                        x_yolo = float(annotation[1])
                        y_yolo = float(annotation[2])
                        yolo_width = float(annotation[3])
                        yolo_height = float(annotation[4])

                        # Convert Yolo Format to Pascal VOC format
                        box_width = yolo_width * image_width
                        box_height = yolo_height * image_height
                        
                        if int(x_yolo * image_width - (box_width / 2)) > 0:
                            x_min = str(int(x_yolo * image_width - (box_width / 2)))
                        else: 
                            x_min = str(0)

                        if int(y_yolo * image_height - (box_height / 2)) > 0:
                            y_min = str(int(y_yolo * image_height - (box_height / 2)))
                        else:
                            y_min = str(0)

                        if int(x_yolo * image_width + (box_width / 2)) > image_width:
                            x_max = str(image_width)
                        else:
                            x_max = str(int(x_yolo * image_width + (box_width / 2)))

                        if int(y_yolo * image_height + (box_height / 2)) > image_height:
                            y_max = str(image_height)
                        else:
                            y_max = str(int(y_yolo * image_height + (box_height / 2)))

                        # write each object to the file
                        f.write('\t<object>\n')
                        f.write('\t\t<name>' + object_name + '</name>\n')
                        f.write('\t\t<pose>Unspecified</pose>\n')
                        f.write('\t\t<truncated>0</truncated>\n')
                        f.write('\t\t<difficult>0</difficult>\n')
                        f.write('\t\t<bndbox>\n')
                        f.write('\t\t\t<xmin>' + x_min + '</xmin>\n')
                        f.write('\t\t\t<ymin>' + y_min + '</ymin>\n')
                        f.write('\t\t\t<xmax>' + x_max + '</xmax>\n')
                        f.write('\t\t\t<ymax>' + y_max + '</ymax>\n')
                        f.write('\t\t</bndbox>\n')
                        f.write('\t</object>\n')

                    # Close the annotation tag once all the objects have been written to the file
                    f.write('</annotation>\n')
                    f.close()  # Close the file


if __name__ == '__main__':

    data_dir = 'data_smoke_only_split/data_smoke_split/val'

    convert_to_voc_xml('data_smoke_only_split/data_smoke_split/classes.txt', data_dir, 'data_smoke_only_split/data_smoke_split/val/xml')