"""
datasets/parsers/voc_parser.py

Parse Pascal VOC XML annotations and convert them into a unified format
for conversion or visualization.
"""

import xml.etree.ElementTree as ET
from typing import List, Tuple


def parse_voc_label(xml_path: str):
    """
    Parse a Pascal VOC annotation XML file.
    Returns list of (class_name, xmin, ymin, xmax, ymax).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []

    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        objects.append((cls_name, xmin, ymin, xmax, ymax))

    return objects
