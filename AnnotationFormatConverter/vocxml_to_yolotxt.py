# pylint: disable=missing-module-docstring, line-too-long, import-error, wrong-import-position

#TODO: recheck this file with typing and linting.

import glob
import os
import xml.etree.ElementTree as ET

import tqdm


def extract_info_from_xml(xml_file: str) -> dict:
    """Function to get the data from XML Annotation.

    Args:
        xml_file (str): xml annotation file to extract information.

    Returns:
        dict: python dictionary that contain information extracted from xml annotation file.
    """
    root = ET.parse(xml_file).getroot()

    # Initialise the info dict
    info_dict = {}
    info_dict["bboxes"] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name
        if elem.tag == "filename":
            info_dict["filename"] = elem.text

        # Get the image size
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))

            info_dict["image_size"] = tuple(image_size)

        # Get details of the bounding box
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text

                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)
            info_dict["bboxes"].append(bbox)

    return info_dict


def convert_to_yolo_txt(saved_folder: str, info_dict: dict, class_name_to_id_mapping: dict):
    """Convert the info dict to the required yolo format and write it to disk.
    yolo txt format: class x_center y_center width height

    Args:
        saved_folder (str): folder name to save txt annotation format.
        info_dict (dict): python dictionary that contain information extracted from xml file.
        class_name_to_id_mapping (dict): python dict key (class name): value (class id).
    """
    print_buffer = []

    # Name of the file which we have to save
    save_file_name = os.path.join(saved_folder, os.path.splitext(info_dict["filename"])[0] + ".txt")

    if len(info_dict["bboxes"]) == 0:
        with open(save_file_name, "w") as file:
            pass

    else:
        # For each bounding box
        for box in info_dict["bboxes"]:
            try:
                class_id = class_name_to_id_mapping[box["class"]]
            except KeyError:
                print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())

            # Transform the bbox co-ordinates as per the format required by YOLO v5
            b_center_x = (box["xmin"] + box["xmax"]) / 2
            b_center_y = (box["ymin"] + box["ymax"]) / 2
            b_width = box["xmax"] - box["xmin"]
            b_height = box["ymax"] - box["ymin"]

            # Normalise the co-ordinates by the dimensions of the image
            image_w, image_h, _ = info_dict["image_size"]
            b_center_x /= image_w
            b_center_y /= image_h
            b_width /= image_w
            b_height /= image_h

            # Write the bbox details to the file
            print_buffer.append(
                "{} {:.5f} {:.5f} {:.5f} {:.5f}".format(class_id, b_center_x, b_center_y, b_width, b_height)
            )

        # Save the annotation to disk
        print("\n".join(print_buffer), file=open(save_file_name, "w"))


if __name__ == "__main__":
    class_name_to_id_mapping = {"smoke": 0, "fire": 1}

    # Get the annotations
    annotations = glob.glob("annotations_xml/" + "*.xml")
    annotations.sort()

    # Convert and save the annotations
    for ann in tqdm.tqdm(annotations):
        info_dict = extract_info_from_xml(ann)
        convert_to_yolo_txt("annotations_txt/val/", info_dict, class_name_to_id_mapping)
