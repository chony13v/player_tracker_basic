"""
A module providing utility functions for bounding box calculations and measurements.

This module contains helper functions for working with bounding boxes, including
calculations for centers, widths, and distances between points.
"""

def get_center_of_bbox(bbox):
    """
    Calculate the center coordinates of a bounding box.

    Args:
        bbox (tuple): Bounding box coordinates in format (x1, y1, x2, y2).

    Returns:
        tuple: Center coordinates (x, y) of the bounding box.
    """
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

def get_bbox_width(bbox):
    """
    Calculate the width of a bounding box.

    Args:
        bbox (tuple): Bounding box coordinates in format (x1, y1, x2, y2).

    Returns:
        int: Width of the bounding box.
    """
    return bbox[2]-bbox[0]