import math

# Left Eye Landmark IDs
LEFT_EYE = [33, 160, 158, 133, 153, 144]

# Right Eye Landmark IDs
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def euclidean_distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )


def eye_aspect_ratio(landmarks, eye_indices):
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    vertical1 = euclidean_distance(p2, p6)
    vertical2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    ear = (vertical1 + vertical2) / (2.0 * horizontal)

    return ear