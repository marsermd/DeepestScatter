import numpy as np

def normalized(v):
    magnitude = np.sqrt(np.dot(v, v))
    if magnitude == 0:
        magnitude = np.finfo(v.dtype).eps
    return v / magnitude

def npVector(v):
    return np.array([v.x, v.y, v.z])

def projectionOn(v, on):
    return np.dot(v, normalized(on))

def projectOn(v, on):
    return np.dot(v, normalized(on)) * normalized(on)

def angleBetween(v1, v2):
    return np.math.acos(np.dot(normalized(v1), normalized(v2)))


def signedAngleBetween(v1, v2, normal):
    angle = angleBetween(v1, v2)
    cross = np.cross(v1, v2)
    if np.dot(normal, cross) < 0:
        angle = -angle
    return angle

def descriptorBasis(lightDirection, viewDirection):
    eZ = normalized(-lightDirection)
    eX = normalized(np.cross(eZ, viewDirection))
    eY = np.cross(eX, eZ)

    return eX, eY, eZ