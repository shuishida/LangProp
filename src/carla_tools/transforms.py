import numpy as np
import carla


def transform_vec(transform: carla.Transform, vector: carla.Vector3D):
    """
    Forward transform is natively implemented as carla.Transform.transform(vector: Vector3D),
    but it is an in-place operation which leads to unwanted mutations
    """
    matrix = np.array(transform.get_matrix())
    vector = np.array([vector.x, vector.y, vector.z, 1])
    x, y, z, _ = np.dot(matrix, vector)
    return carla.Vector3D(float(x), float(y), float(z))


def inv_transform(transform: carla.Transform, vector: carla.Vector3D):
    matrix = np.array(transform.get_inverse_matrix())
    vector = np.array([vector.x, vector.y, vector.z, 1])
    x, y, z, _ = np.dot(matrix, vector)
    return carla.Vector3D(float(x), float(y), float(z))


def rotate_vec(rotation: carla.Rotation, vector: carla.Vector3D):
    transform = carla.Transform(rotation=rotation)
    return transform.transform(vector)


def rotate_in_plane(yaw: float, vector: carla.Vector3D):
    rotation = carla.Rotation(yaw=yaw)
    transform = carla.Transform(rotation=rotation)
    return transform_vec(transform, vector)


def inv_rotate_vec(rotation: carla.Rotation, vector: carla.Vector3D):
    transform = carla.Transform(rotation=rotation)
    matrix = np.array(transform.get_inverse_matrix())
    vector = np.array([vector.x, vector.y, vector.z, 1])
    x, y, z, _ = np.dot(matrix, vector)
    return carla.Vector3D(float(x), float(y), float(z))
