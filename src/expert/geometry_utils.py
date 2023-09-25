import numpy as np

"""
https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/
"""


def vec_to_numpy(carla_vector, include_z: bool = False):
    if include_z:
        return np.array([carla_vector.x, carla_vector.y, carla_vector.z])
    else:
        return np.array([carla_vector.x, carla_vector.y])


def normalize_deg(deg_angle):
    return (deg_angle + 180) % 360 - 180


def get_heading_angle(direction, curr_theta=0):
    angle = np.arctan2(direction[1], direction[0]) - curr_theta
    assert not np.isnan(angle), f"{direction}, {curr_theta}"
    return normalize_deg(np.degrees(angle))


def get_collision(p1, v1, p2, v2):
    """
    Solve for
    (x y)' = p1 + v1 * t1 = p2 + v2 * t2
    p2 - p1 = (v1 -v2) (t1 t2)'
    """
    p1, v1, p2, v2 = list(map(vec_to_numpy, (p1, v1, p2, v2)))

    A = np.stack([v1, -v2], axis=1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return None, None

    x = np.linalg.solve(A, b)   # (t1, t2)
    collides = all(x >= 0) and all(x <= 1)  # how many seconds until collision

    dist_to_collision_1 = x[0] * np.linalg.norm(v1) if collides else None
    dist_to_collision_2 = x[1] * np.linalg.norm(v2) if collides else None

    return dist_to_collision_1, dist_to_collision_2
