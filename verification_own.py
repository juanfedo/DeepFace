# built-in dependencies
import time
from typing import Any, Dict, Optional, Union, List

# 3rd party dependencies
import numpy as np
import json
# project dependencies
from deepface.modules import modeling, verification
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons import logger as log

logger = log.get_singletonish_logger()


def verify2(
    img1_path: Union[str, np.ndarray, List[float]],
    imagenes: [] = [],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    silent: bool = False,
    threshold: Optional[float] = None,
    anti_spoofing: bool = False,
 
) -> Dict[str, Any]:
    """
    Verify if an image pair represents the same person or different persons.

    The verification function converts facial images to vectors and calculates the similarity
    between those vectors. Vectors of images of the same person should exhibit higher similarity
    (or lower distance) than vectors of images of different persons.

    Args:
        img1_path (str or np.ndarray or List[float]): Path to the first image.
            Accepts exact image path as a string, numpy array (BGR), base64 encoded images
            or pre-calculated embeddings.

        img2_path (str or np.ndarray or  or List[float]): Path to the second image.
            Accepts exact image path as a string, numpy array (BGR), base64 encoded images
            or pre-calculated embeddings.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
            (default is opencv)

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).

        align (bool): Flag to enable face alignment (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        normalization (string): Normalize the input image before feeding it to the model.
            Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace (default is base)

        silent (boolean): Suppress or allow some log messages for a quieter analysis process
            (default is False).

        threshold (float): Specify a threshold to determine whether a pair represents the same
            person or different individuals. This threshold is used for comparing distances.
            If left unset, default pre-tuned threshold values will be applied based on the specified
            model name and distance metric (default is None).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

    Returns:
        result (dict): A dictionary containing verification results.

        - 'verified' (bool): Indicates whether the images represent the same person (True)
            or different persons (False).

        - 'distance' (float): The distance measure between the face vectors.
            A lower distance indicates higher similarity.

        - 'threshold' (float): The maximum threshold used for verification.
            If the distance is below this threshold, the images are considered a match.

        - 'model' (str): The chosen face recognition model.

        - 'similarity_metric' (str): The chosen similarity metric for measuring distances.

        - 'facial_areas' (dict): Rectangular regions of interest for faces in both images.
            - 'img1': {'x': int, 'y': int, 'w': int, 'h': int}
                    Region of interest for the first image.
            - 'img2': {'x': int, 'y': int, 'w': int, 'h': int}
                    Region of interest for the second image.

        - 'time' (float): Time taken for the verification process in seconds.
    """

    tic = time.time()

    model: FacialRecognition = modeling.build_model(model_name)
    dims = model.output_shape

    # extract faces from img1
    if isinstance(img1_path, list):
        # given image is already pre-calculated embedding
        if not all(isinstance(dim, float) for dim in img1_path):
            raise ValueError(
                "When passing img1_path as a list, ensure that all its items are of type float."
            )

        if silent is False:
            logger.warn(
                "You passed 1st image as pre-calculated embeddings."
                f"Please ensure that embeddings have been calculated for the {model_name} model."
            )

        if len(img1_path) != dims:
            raise ValueError(
                f"embeddings of {model_name} should have {dims} dimensions,"
                f" but it has {len(img1_path)} dimensions input"
            )

        img1_embeddings = [img1_path]
        img1_facial_areas = [None]
    else:
        try:
            img1_embeddings, img1_facial_areas = verification.__extract_faces_and_embeddings(
                img_path=img1_path,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=enforce_detection,
                align=align,
                expand_percentage=expand_percentage,
                normalization=normalization,
                anti_spoofing=anti_spoofing,
            )
        except ValueError as err:
            raise ValueError("Exception while processing img1_path") from err

    print (f"numero de imagenes {len(imagenes)}")
    resp_obj = {
        "verified": False
    }
    for imagen in imagenes:
        no_facial_area = {
            "x": None,
            "y": None,
            "w": None,
            "h": None,
            "left_eye": None,
            "right_eye": None,
        }

        distances = []
        facial_areas = []
        for idx, img1_embedding in enumerate(img1_embeddings):
            for idy, img2_embedding in enumerate(json.loads(imagen[2])):
                distance = verification.find_distance(img1_embedding, img2_embedding, distance_metric)
                distances.append(distance)
                facial_areas.append(
                    (img1_facial_areas[idx] or no_facial_area, json.loads(imagen[3])[idy] or no_facial_area)
                )

        # find the face pair with minimum distance
        threshold = threshold or verification.find_threshold(model_name, distance_metric)
        distance = float(min(distances))  # best distance
        facial_areas = facial_areas[np.argmin(distances)]

        toc = time.time()

        resp_obj = {
            "verified": distance <= threshold,
            "distance": distance,
            "threshold": threshold,
            "model": model_name,
            "detector_backend": detector_backend,
            "similarity_metric": distance_metric,
            "facial_areas": {"img1": facial_areas[0], "img2": facial_areas[1]},
            "time": round(toc - tic, 2),
            "nombre": imagen[1]
        }
    
    return resp_obj

