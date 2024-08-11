# built-in dependencies
import os
import time
from typing import List, Tuple, Optional

# 3rd party dependencies
import numpy as np
import pandas as pd
import cv2

# project dependencies
from deepface import DeepFace
from deepface.commons import logger as log
from deepface.modules import streaming
import verification_own

logger = log.get_singletonish_logger()

# dependency configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


IDENTIFIED_IMG_SIZE = 112
TEXT_COLOR = (255, 255, 255)

# pylint: disable=unused-variable
def analysis3(
    db_path: str,
    imagenes: [] = [],
    model_name="VGG-Face",
    detector_backend="opencv",
    distance_metric="cosine",
    enable_face_analysis=True,
    source=0,
    time_threshold=5,
    frame_threshold=5,
    anti_spoofing: bool = False,
):
    """
    Run real time face recognition and facial attribute analysis

    Args:
        db_path (string): Path to the folder containing image files. All detected faces
            in the database will be considered in the decision-making process.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
            (default is opencv).

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

        enable_face_analysis (bool): Flag to enable face analysis (default is True).

        source (Any): The source for the video stream (default is 0, which represents the
            default camera).

        time_threshold (int): The time threshold (in seconds) for face recognition (default is 5).

        frame_threshold (int): The frame threshold for face recognition (default is 5).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

    Returns:
        None
    """
    # initialize models
    #build_demography_models(enable_face_analysis=enable_face_analysis)
    streaming.build_facial_recognition_model(model_name=model_name)
    # call a dummy find function for db_path once to create embeddings before starting webcam
    #_ = search_identity(
    #    detected_face=np.zeros([224, 224, 3]),
    #    db_path=db_path,
    #    detector_backend=detector_backend,
    #    distance_metric=distance_metric,
    #    model_name=model_name,
    #)

    freezed_img = None
    freeze = False
    num_frames_with_faces = 0
    tic = time.time()

    cap = cv2.VideoCapture(source)  # webcam
    while True:
        has_frame, img = cap.read()
        if not has_frame:
            break

        # we are adding some figures into img such as identified facial image, age, gender
        # that is why, we need raw image itself to make analysis
        raw_img = img.copy()

        faces_coordinates = []
        if freeze is False:
            faces_coordinates = streaming.grab_facial_areas(
                img=img, detector_backend=detector_backend, anti_spoofing=anti_spoofing
            )

            # we will pass img to analyze modules (identity, demography) and add some illustrations
            # that is why, we will not be able to extract detected face from img clearly
            detected_faces = streaming.extract_facial_areas(img=img, faces_coordinates=faces_coordinates)

            img = streaming.highlight_facial_areas(img=img, faces_coordinates=faces_coordinates)
            img = streaming.countdown_to_freeze(
                img=img,
                faces_coordinates=faces_coordinates,
                frame_threshold=frame_threshold,
                num_frames_with_faces=num_frames_with_faces,
            )

            num_frames_with_faces = num_frames_with_faces + 1 if len(faces_coordinates) else 0

            freeze = num_frames_with_faces > 0 and num_frames_with_faces % frame_threshold == 0
            if freeze:
                # add analyze results into img - derive from raw_img
                img = streaming.highlight_facial_areas(
                    img=raw_img, faces_coordinates=faces_coordinates, anti_spoofing=anti_spoofing
                )

                result = verification_own.verify2(
                    img1_path = img,
                    imagenes = imagenes,
                    enforce_detection = False,
                    threshold=0.8
                )

                print (result)

                if result['verified']:
                    for x, y, w, h in faces_coordinates:                        
                        color = (255, 0, 0)
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
                        cv2.putText(img,result['nombre'],(x + w, y + 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,TEXT_COLOR,1,)
                    print('ENCONTRADO')

                # freeze the img after analysis
                freezed_img = img.copy()

                # start counter for freezing
                tic = time.time()
                logger.info("freezed")

        elif freeze is True and time.time() - tic > time_threshold:
            freeze = False
            freezed_img = None
            # reset counter for freezing
            tic = time.time()
            logger.info("freeze released")

        freezed_img = streaming.countdown_to_release(img=freezed_img, tic=tic, time_threshold=time_threshold)

        cv2.imshow("img", img if freezed_img is None else freezed_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
            break

    # kill open cv things
    cap.release()
    cv2.destroyAllWindows()