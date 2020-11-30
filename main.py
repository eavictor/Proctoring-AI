import cv2
import numpy as np
import face_detector
import face_landmarks
from datetime import datetime
import platform
import os


# Reduce launching time for RTX 3000 series cards.
os.environ['CUDA_CACHE_MAXSIZE'] = '2147483648'


def main(debug=False):
    # Load face detect model & face landmark model
    face_model = face_detector.get_face_detector()
    landmark_model = face_landmarks.get_landmark_model()

    cv2.namedWindow('debug_window')

    # Create camera instance for capture picture,  speed up processing on windows platform by using cv2.CAP_DSHOW.
    if platform.system().lower() == 'windows':
        camera = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    else:
        camera = cv2.VideoCapture(0)

    _, sample_img = camera.read()  # grab one frame, for getting resolution
    height, width, channel = sample_img.shape  # resolution & color channel (1 or None = GrayScale, 3 = BGR colors)

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corne
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    # Camera internals
    focal_length = width
    center = (width/2, height/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype='double')

    # Lens distance coefficient: assuming no lens distortion
    dist_coefficient = np.zeros((4, 1))

    while True:
        # start time stamp this frame
        time_start = datetime.now()

        # 1. capture image (1 frame)
        ret, frame = camera.read()
        if ret:
            # 2. detect faces
            faces = face_detector.find_faces(frame, face_model)
            if faces:
                if debug:
                    face_detector.draw_faces(frame, faces)

                for face in faces:

                    # 3. detect face landmarks
                    try:
                        landmarks = face_landmarks.detect_marks(frame, landmark_model, face)
                    except cv2.error:
                        # Skip this round if failed to detect landmarks.
                        break

                    if debug:
                        face_landmarks.draw_marks(frame, landmarks, color=(0, 255, 0))

                    frame_points = np.array([
                        landmarks[30],  # Nose tip
                        landmarks[8],   # Chin
                        landmarks[36],  # Left eye left corner
                        landmarks[45],  # Right eye right corne
                        landmarks[48],  # Left Mouth corner
                        landmarks[54]   # Right mouth corner
                    ], dtype='double')

                    # 4. get rotation & translation vector
                    success, rotation_vector, translation_vector = cv2.solvePnP(
                        model_points, frame_points, camera_matrix, dist_coefficient, flags=cv2.SOLVEPNP_UPNP
                    )

                    if debug:
                        print(' ' * 13, 'solvePnP:',
                              success, rotation_vector.tolist(), translation_vector.tolist(), end='\r')
                    # TODO: calculate iris, mouth and head's roll pitch yaw
                    # TODO: create socket client, send calculated data to unity

        cv2.imshow('debug_window', frame)

        # end time stamp of this frame
        time_delta = datetime.now() - time_start
        # calculate Frame Per Second
        fps = 1e6 / time_delta.microseconds
        print(' FPS:', fps, end='\r')

        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    if debug:
        cv2.destroyAllWindows()
    camera.release()


if __name__ == '__main__':
    main()
