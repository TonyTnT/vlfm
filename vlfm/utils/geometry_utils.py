# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import math
from typing import Tuple

import numpy as np


def rho_theta(curr_pos: np.ndarray, curr_heading: float, curr_goal: np.ndarray) -> Tuple[float, float]:
    """Calculates polar coordinates (rho, theta) relative to a given position and
    heading to a given goal position. 'rho' is the distance from the agent to the goal,
    and theta is how many radians the agent must turn (to the left, CCW from above) to
    face the goal. Coordinates are in (x, y), where x is the distance forward/backwards,
    and y is the distance to the left or right (right is negative)

    Args:
        curr_pos (np.ndarray): Array of shape (2,) representing the current position.
        curr_heading (float): The current heading, in radians. It represents how many
            radians  the agent must turn to the left (CCW from above) from its initial
            heading to reach its current heading.
        curr_goal (np.ndarray): Array of shape (2,) representing the goal position.

    Returns:
        Tuple[float, float]: A tuple of floats representing the polar coordinates
            (rho, theta).
    """
    rotation_matrix = get_rotation_matrix(-curr_heading, ndims=2)
    local_goal = curr_goal - curr_pos
    local_goal = rotation_matrix @ local_goal

    rho = np.linalg.norm(local_goal)
    theta = np.arctan2(local_goal[1], local_goal[0])

    return float(rho), float(theta)


def get_rotation_matrix(angle: float, ndims: int = 2) -> np.ndarray:
    """Returns a 2x2 or 3x3 rotation matrix for a given angle; if 3x3, the z-axis is
    rotated."""
    if ndims == 2:
        return np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
    elif ndims == 3:
        return np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
    else:
        raise ValueError("ndims must be 2 or 3")


def wrap_heading(theta: float) -> float:
    """Wraps given angle to be between -pi and pi.

    Args:
        theta (float): The angle in radians.
    Returns:
        float: The wrapped angle in radians.
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi


def calculate_vfov(hfov: float, width: int, height: int) -> float:
    """Calculates the vertical field of view (VFOV) based on the horizontal field of
    view (HFOV), width, and height of the image sensor.

    Args:
        hfov (float): The HFOV in radians.
        width (int): Width of the image sensor in pixels.
        height (int): Height of the image sensor in pixels.

    Returns:
        A float representing the VFOV in radians.
    """
    # Calculate the diagonal field of view (DFOV)
    dfov = 2 * math.atan(math.tan(hfov / 2) * math.sqrt((width**2 + height**2) / (width**2 + height**2)))

    # Calculate the vertical field of view (VFOV)
    vfov = 2 * math.atan(math.tan(dfov / 2) * (height / math.sqrt(width**2 + height**2)))

    return vfov


def within_fov_cone(
    cone_origin: np.ndarray,
    cone_angle: float,
    cone_fov: float,
    cone_range: float,
    points: np.ndarray,
) -> np.ndarray:
    """Checks if points are within a cone of a given origin, angle, fov, and range.

    Args:
        cone_origin (np.ndarray): The origin of the cone.
        cone_angle (float): The angle of the cone in radians.
        cone_fov (float): The field of view of the cone in radians.
        cone_range (float): The range of the cone.
        points (np.ndarray): The points to check.

    Returns:
        np.ndarray: The subarray of points that are within the cone.
    """
    directions = points[:, :3] - cone_origin
    dists = np.linalg.norm(directions, axis=1)
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    angle_diffs = np.mod(angles - cone_angle + np.pi, 2 * np.pi) - np.pi

    mask = np.logical_and(dists <= cone_range, np.abs(angle_diffs) <= cone_fov / 2)
    return points[mask]


def convert_to_global_frame(agent_pos: np.ndarray, agent_yaw: float, local_pos: np.ndarray) -> np.ndarray:
    """Converts a given position from the agent's local frame to the global frame.

    Args:
        agent_pos (np.ndarray): A 3D vector representing the agent's position in their
            local frame.
        agent_yaw (float): The agent's yaw in radians.
        local_pos (np.ndarray): A 3D vector representing the position to be converted in
            the agent's local frame.

    Returns:
        A 3D numpy array representing the position in the global frame.
    """
    # Append a homogeneous coordinate of 1 to the local position vector
    local_pos_homogeneous = np.append(local_pos, 1)

    # Construct the homogeneous transformation matrix
    transformation_matrix = xyz_yaw_to_tf_matrix(agent_pos, agent_yaw)

    # Perform the transformation using matrix multiplication
    global_pos_homogeneous = transformation_matrix.dot(local_pos_homogeneous)
    global_pos_homogeneous = global_pos_homogeneous[:3] / global_pos_homogeneous[-1]

    return global_pos_homogeneous


def extract_yaw(matrix: np.ndarray) -> float:
    """Extract the yaw angle from a 4x4 transformation matrix.

    Args:
        matrix (np.ndarray): A 4x4 transformation matrix.
    Returns:
        float: The yaw angle in radians.
    """
    assert matrix.shape == (4, 4), "The input matrix must be 4x4"
    rotation_matrix = matrix[:3, :3]

    # Compute the yaw angle
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    return yaw


def xyz_yaw_to_tf_matrix(xyz: np.ndarray, yaw: float) -> np.ndarray:
    """Converts a given position and yaw angle to a 4x4 transformation matrix.

    Args:
        xyz (np.ndarray): A 3D vector representing the position.
        yaw (float): The yaw angle in radians.
    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    x, y, z = xyz
    transformation_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0, x],
            [np.sin(yaw), np.cos(yaw), 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ]
    )
    return transformation_matrix


def closest_point_within_threshold(points_array: np.ndarray, target_point: np.ndarray, threshold: float) -> int:
    """Find the point within the threshold distance that is closest to the target_point.

    Args:
        points_array (np.ndarray): An array of 2D points, where each point is a tuple
            (x, y).
        target_point (np.ndarray): The target 2D point (x, y).
        threshold (float): The maximum distance threshold.

    Returns:
        int: The index of the closest point within the threshold distance.
    """
    distances = np.sqrt((points_array[:, 0] - target_point[0]) ** 2 + (points_array[:, 1] - target_point[1]) ** 2)
    within_threshold = distances <= threshold

    if np.any(within_threshold):
        closest_index = np.argmin(distances)
        return int(closest_index)

    return -1


def transform_points(transformation_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Transforms a set of points using a given transformation matrix.

    Args:
        transformation_matrix (np.ndarray): A 4x4 transformation matrix.
        points (np.ndarray): An array of points, accepects Nx3 and Nx4, where each point is a tuple (z, x, y) or (z, x, y, value).

    Returns:
        np.ndarray: An array of transformed points.
    """
    if points.shape[1] >= 4:
        # Separate the points and the category information
        points_coords = points[:, :3]
        extra_info = points[:, 3:]
    else:
        # No category information
        points_coords = points
        extra_info = None

    # Add a homogeneous coordinate of 1 to each point for matrix multiplication
    homogeneous_points = np.hstack((points_coords, np.ones((points_coords.shape[0], 1))))

    # Apply the transformation matrix to the points
    transformed_points = np.dot(transformation_matrix, homogeneous_points.T).T

    # Remove the added homogeneous coordinate and divide by the last coordinate
    transformed_points = transformed_points[:, :3] / transformed_points[:, 3:]

    if extra_info is not None:
        # Add the category information back to the transformed points
        transformed_points = np.hstack((transformed_points, extra_info))

    return transformed_points


def remap(value: float, from_low: float, from_high: float, to_low: float, to_high: float) -> float:
    """Maps a value from one range to another.

    Args:
        value (float): The value to be mapped.
        from_low (float): The lower bound of the input range.
        from_high (float): The upper bound of the input range.
        to_low (float): The lower bound of the output range.
        to_high (float): The upper bound of the output range.

    Returns:
        float: The mapped value.
    """
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low


def get_confidence(z, x, fov):
    """Generate a FOV cone with central values weighted more heavily"""
    _min_confidence = 0.25
    horizontal = abs(z)
    vertical = abs(x)
    angle = np.arctan2(vertical, horizontal)
    angle = remap(angle, 0, fov / 2, 0, np.pi / 2)
    confidence = np.cos(angle) ** 2
    confidence = remap(confidence, 0, 1, _min_confidence, 1)
    return confidence


def get_point_cloud(
    depth_image: np.ndarray,
    mask: np.ndarray,
    fx: float,
    fy: float,
    fov: float = None,
    semantic_mask: np.array = None,
    pcd_conf_threshold: float = 0.5,
) -> np.ndarray:
    """Calculates the 3D coordinates (x, y, z) of points in the depth image based on
    the horizontal field of view (HFOV), the image width and height, the depth values,
    and the pixel x and y coordinates.

    Args:
        depth_image (np.ndarray): 2D depth image.
        mask (np.ndarray): 2D binary mask identifying relevant pixels.
        fx (float): Focal length in the x direction.
        fy (float): Focal length in the y direction.

    Returns:
        np.ndarray: Array of 3D coordinates (x, y, z) of the points in the image plane.
    """
    v, u = np.where(mask)
    z = depth_image[v, u]
    x = (u - depth_image.shape[1] // 2) * z / fx
    y = (v - depth_image.shape[0] // 2) * z / fy
    if semantic_mask is not None:
        semantic_labels = semantic_mask[v, u]
        conf = get_confidence(z, x, fov)
        mask_conf = conf > pcd_conf_threshold
        cloud = np.stack(
            (z[mask_conf], -x[mask_conf], -y[mask_conf], semantic_labels[mask_conf], conf[mask_conf]), axis=-1
        )

    else:
        cloud = np.stack((z, -x, -y), axis=-1)

    return cloud


def get_fov(focal_length: float, image_height_or_width: int) -> float:
    """
    Given an fx and the image width, or an fy and the image height, returns the
    horizontal or vertical field of view, respectively.

    Args:
        focal_length: Focal length of the camera.
        image_height_or_width: Height or width of the image.

    Returns:
        Field of view in radians.
    """

    # Calculate the field of view using the formula
    fov = 2 * math.atan((image_height_or_width / 2) / focal_length)
    return fov


def pt_from_rho_theta(rho: float, theta: float) -> np.ndarray:
    """
    Given a rho and theta, computes the x and y coordinates.

    Args:
        rho: Distance from the origin.
        theta: Angle from the x-axis.

    Returns:
        numpy.ndarray: x and y coordinates.
    """
    x = rho * math.cos(theta)
    y = rho * math.sin(theta)

    return np.array([x, y])
