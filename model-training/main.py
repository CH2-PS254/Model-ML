import enum
from typing import List, NamedTuple
import numpy as np
import os
from typing import Dict, List
import cv2
import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
import wget
import matplotlib.pyplot as plt

if 'movenet_thunder1.tflite' not in os.listdir():
    wget.download('https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite', 'movenet_thunder.tflite')

class Landmark(enum.Enum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

class Point(NamedTuple):
    x: float
    y: float

class Rectangle(NamedTuple):
    start_point: Point
    end_point: Point

class KeyPoint(NamedTuple):
    body_part: Landmark
    coordinate: Point
    score: float

class Person(NamedTuple):
    keypoints: List[KeyPoint]
    bounding_box: Rectangle
    score: float
    id: int = None

def person_from_keypoints_with_scores(keypoints_with_scores, image_height, image_width, keypoint_score_threshold=0.1):
        kpts_x, kpts_y, scores=keypoints_with_scores[:, 1], keypoints_with_scores[:, 0], keypoints_with_scores[:, 2]
        keypoints = [
        KeyPoint(Landmark(i), Point(int(kpts_x[i] * image_width), int(kpts_y[i] * image_height)), scores[i])
        for i in range(scores.shape[0])
        if scores[i] > keypoint_score_threshold
        ]
        min_x, min_y = int(np.amin(kpts_x) * image_width), int(np.amin(kpts_y) * image_height)
        max_x, max_y = int(np.amax(kpts_x) * image_width), int(np.amax(kpts_y) * image_height)
        start_point, end_point = Point(min_x, min_y), Point(max_x, max_y)
        bounding_box = Rectangle(start_point, end_point)
        scores_above_threshold = list(filter(lambda x: x > keypoint_score_threshold, scores))
        person_score = np.average(scores_above_threshold)
        return Person(keypoints, bounding_box, person_score)

class Category(NamedTuple):
    label: str
    score: float


try:
  from tflite_runtime.interpreter import Interpreter
except ImportError:
  import tensorflow as tf
  Interpreter = tf.lite.Interpreter
class Movenet:
    _MIN_CROP_KEYPOINT_SCORE = 0.2
    _TORSO_EXPANSION_RATIO = 1.9
    _BODY_EXPANSION_RATIO = 1.2

    def __init__(self, model_name: str) -> None:
        _, ext = os.path.splitext(model_name)
        if not ext:
            model_name += '.tflite'
        interpreter = Interpreter(model_path=model_name, num_threads=4)
        interpreter.allocate_tensors()

        self._input_index = interpreter.get_input_details()[0]['index']
        self._output_index = interpreter.get_output_details()[0]['index']

        self._input_height = interpreter.get_input_details()[0]['shape'][1]
        self._input_width = interpreter.get_input_details()[0]['shape'][2]

        self._interpreter = interpreter
        self._crop_region = None

    def init_crop_region(self, image_height: int,
                         image_width: int) -> Dict[(str, float)]:
        if image_width > image_height:
            x_min = 0.0
            box_width = 1.0
            y_min = (image_height / 2 - image_width / 2) / image_height
            box_height = image_width / image_height
        else:
            y_min = 0.0
            box_height = 1.0
            x_min = (image_width / 2 - image_height / 2) / image_width
            box_width = image_height / image_width

        return {
            'y_min': y_min,
            'x_min': x_min,
            'y_max': y_min + box_height,
            'x_max': x_min + box_width,
            'height': box_height,
            'width': box_width
        }

    def _torso_visible(self, keypoints: np.ndarray) -> bool:

        left_hip_score = keypoints[Landmark.LEFT_HIP.value, 2]
        right_hip_score = keypoints[Landmark.RIGHT_HIP.value, 2]
        left_shoulder_score = keypoints[Landmark.LEFT_SHOULDER.value, 2]
        right_shoulder_score = keypoints[Landmark.RIGHT_SHOULDER.value, 2]

        left_hip_visible = left_hip_score > Movenet._MIN_CROP_KEYPOINT_SCORE
        right_hip_visible = right_hip_score > Movenet._MIN_CROP_KEYPOINT_SCORE
        left_shoulder_visible = left_shoulder_score > Movenet._MIN_CROP_KEYPOINT_SCORE
        right_shoulder_visible = right_shoulder_score > Movenet._MIN_CROP_KEYPOINT_SCORE

        return ((left_hip_visible or right_hip_visible) and (left_shoulder_visible or right_shoulder_visible))

    def _determine_torso_and_body_range(self, keypoints: np.ndarray,
                                        target_keypoints: Dict[(str, float)],
                                        center_y: float,
                                        center_x: float) -> List[float]:
        torso_joints = [
            Landmark.LEFT_SHOULDER, Landmark.RIGHT_SHOULDER, Landmark.LEFT_HIP,
            Landmark.RIGHT_HIP
        ]
        max_torso_yrange = 0.0
        max_torso_xrange = 0.0
        for joint in torso_joints:
            dist_y = abs(center_y - target_keypoints[joint][0])
            dist_x = abs(center_x - target_keypoints[joint][1])
            if dist_y > max_torso_yrange:
                max_torso_yrange = dist_y
            if dist_x > max_torso_xrange:
                max_torso_xrange = dist_x

        max_body_yrange = 0.0
        max_body_xrange = 0.0
        for idx in range(len(Landmark)):
            if keypoints[Landmark(idx).value, 2] < Movenet._MIN_CROP_KEYPOINT_SCORE:
                continue
            dist_y = abs(center_y - target_keypoints[joint][0])
            dist_x = abs(center_x - target_keypoints[joint][1])
            if dist_y > max_body_yrange:
                max_body_yrange = dist_y

            if dist_x > max_body_xrange:
                max_body_xrange = dist_x

        return [
            max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange
        ]

    def _determine_crop_region(self, keypoints: np.ndarray, image_height: int,
                               image_width: int) -> Dict[(str, float)]:
        target_keypoints = {}
        for idx in range(len(Landmark)):
            target_keypoints[Landmark(idx)] = [
                keypoints[idx, 0] * image_height, keypoints[idx, 1] * image_width
            ]

        if self._torso_visible(keypoints):
            center_y = (target_keypoints[Landmark.LEFT_HIP][0] +
                        target_keypoints[Landmark.RIGHT_HIP][0]) / 2
            center_x = (target_keypoints[Landmark.LEFT_HIP][1] +
                        target_keypoints[Landmark.RIGHT_HIP][1]) / 2

            (max_torso_yrange, max_torso_xrange, max_body_yrange,
             max_body_xrange) = self._determine_torso_and_body_range(
                keypoints, target_keypoints, center_y, center_x)

            crop_length_half = np.amax([
                max_torso_xrange * Movenet._TORSO_EXPANSION_RATIO,
                max_torso_yrange * Movenet._TORSO_EXPANSION_RATIO,
                max_body_yrange * Movenet._BODY_EXPANSION_RATIO,
                max_body_xrange * Movenet._BODY_EXPANSION_RATIO
            ])

            distances_to_border = np.array(
                [center_x, image_width - center_x, center_y, image_height - center_y])
            crop_length_half = np.amin(
                [crop_length_half, np.amax(distances_to_border)])

            if crop_length_half > max(image_width, image_height) / 2:
                return self.init_crop_region(image_height, image_width)
            else:
                crop_length = crop_length_half * 2
            crop_corner = [center_y - crop_length_half, center_x - crop_length_half]
            return {
                'y_min':
                    crop_corner[0] / image_height,
                'x_min':
                    crop_corner[1] / image_width,
                'y_max': (crop_corner[0] + crop_length) / image_height,
                'x_max': (crop_corner[1] + crop_length) / image_width,
                'height': (crop_corner[0] + crop_length) / image_height -
                          crop_corner[0] / image_height,
                'width': (crop_corner[1] + crop_length) / image_width -
                         crop_corner[1] / image_width
            }
        else:
            return self.init_crop_region(image_height, image_width)

    def _crop_and_resize(
            self, image: np.ndarray, crop_region: Dict[(str, float)],
            crop_size: (int, int)) -> np.ndarray:
        y_min, x_min, y_max, x_max = [
            crop_region['y_min'], crop_region['x_min'], crop_region['y_max'],
            crop_region['x_max']
        ]

        crop_top = int(0 if y_min < 0 else y_min * image.shape[0])
        crop_bottom = int(image.shape[0] if y_max >= 1 else y_max * image.shape[0])
        crop_left = int(0 if x_min < 0 else x_min * image.shape[1])
        crop_right = int(image.shape[1] if x_max >= 1 else x_max * image.shape[1])

        padding_top = int(0 - y_min * image.shape[0] if y_min < 0 else 0)
        padding_bottom = int((y_max - 1) * image.shape[0] if y_max >= 1 else 0)
        padding_left = int(0 - x_min * image.shape[1] if x_min < 0 else 0)
        padding_right = int((x_max - 1) * image.shape[1] if x_max >= 1 else 0)
        output_image = image[crop_top:crop_bottom, crop_left:crop_right]
        output_image = cv2.copyMakeBorder(output_image, padding_top, padding_bottom,
                                          padding_left, padding_right,
                                          cv2.BORDER_CONSTANT)
        output_image = cv2.resize(output_image, (crop_size[0], crop_size[1]))

        return output_image

    def _run_detector(
            self, image: np.ndarray, crop_region: Dict[(str, float)],
            crop_size: (int, int)) -> np.ndarray:
        input_image = self._crop_and_resize(image, crop_region, crop_size=crop_size)
        input_image = input_image.astype(dtype=np.uint8)

        self._interpreter.set_tensor(self._input_index,
                                     np.expand_dims(input_image, axis=0))
        self._interpreter.invoke()

        keypoints_with_scores = self._interpreter.get_tensor(self._output_index)
        keypoints_with_scores = np.squeeze(keypoints_with_scores)

        for idx in range(len(Landmark)):
            keypoints_with_scores[idx, 0] = crop_region[
                                                'y_min'] + crop_region['height'] * keypoints_with_scores[idx, 0]
            keypoints_with_scores[idx, 1] = crop_region[
                                                'x_min'] + crop_region['width'] * keypoints_with_scores[idx, 1]

        return keypoints_with_scores

    def detect(self, input_image: np.ndarray, reset_crop_region: bool = False) -> Person:
        image_height, image_width, _ = input_image.shape
        if (self._crop_region is None) or reset_crop_region:
            self._crop_region = self.init_crop_region(image_height, image_width)

        keypoint_with_scores = self._run_detector(input_image, self._crop_region,
                                                  crop_size=(self._input_height, self._input_width))
        self._crop_region = self._determine_crop_region(keypoint_with_scores,
                                                        image_height, image_width)

        return person_from_keypoints_with_scores(keypoint_with_scores, image_height, image_width)



movenet = Movenet('movenet_thunder')


def detect(input_tensor, inference_count=3):
    movenet.detect(input_tensor.numpy(), reset_crop_region=True)

    for _ in range(inference_count - 1):
        detection = movenet.detect(input_tensor.numpy(),
                                   reset_crop_region=False)

    return detection


class Preprocessor:
    def __init__(self, images_in_folder, csvs_out_path):
        self._images_in_folder = images_in_folder
        self._csvs_out_path = csvs_out_path
        self._train_data = []
        self._test_data = []
        self.class_names = {}
        self.expected_length = 54

    def process(self, detection_threshold=0.1):
        for idx, pose_class_name in enumerate(os.listdir(self._images_in_folder)):
            self.class_names[idx] = pose_class_name

            images_in_folder = os.path.join(self._images_in_folder, pose_class_name)

            for image_name in tqdm.tqdm(os.listdir(images_in_folder)):
                image_path = os.path.join(images_in_folder, image_name)

                try:
                    image = tf.io.read_file(image_path)
                    image = tf.io.decode_image(image, channels=3)
                except:
                    continue
                if image.shape[2] != 3:
                    continue
                person = detect(image)

                min_landmark_score = min([keypoint.score for keypoint in person.keypoints])
                should_keep_image = min_landmark_score >= detection_threshold

                if should_keep_image:
                    pose_landmarks = np.array(
                        [[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
                         for keypoint in person.keypoints],
                        dtype=np.float32)

                    max_x = np.max(pose_landmarks[:, 0])
                    min_x = np.min(pose_landmarks[:, 0])
                    max_y = np.max(pose_landmarks[:, 1])
                    min_y = np.min(pose_landmarks[:, 1])
                    pose_landmarks[:, 0] = (pose_landmarks[:, 0] - min_x) / (max_x - min_x)
                    pose_landmarks[:, 1] = (pose_landmarks[:, 1] - min_y) / (max_y - min_y)

                    if len(pose_landmarks.flatten()) == self.expected_length-3:
                        self._train_data.append(
                        [image_name] + pose_landmarks.flatten().tolist() + [idx, pose_class_name])

        filtered_train_data = [data for data in self._train_data if len(data) == self.expected_length]
        self.save_to_csv(filtered_train_data, self._csvs_out_path)


    def save_to_csv(self, data, filename):
        df = pd.DataFrame(data)

        body_part_names = [part.name for part in Landmark]
        column_names = ['filename'] + [f'{part}_{coord}' for part in body_part_names for coord in ['x', 'y', 'score']]
        column_names += ['class_no', 'class_name']

        df.columns = column_names
        df.to_csv(filename, index=False)



def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.drop(['filename'], axis=1, inplace=True)
    classes = df.pop('class_name').unique()
    y = df.pop('class_no')

    X = df.astype('float64')
    y = keras.utils.to_categorical(y)

    return X, y, classes

def get_center_point(landmarks, left_bodypart, right_bodypart):
    left = tf.gather(landmarks, left_bodypart.value, axis=1)
    right = tf.gather(landmarks, right_bodypart.value, axis=1)
    center = left * 0.5 + right * 0.5
    return center

def get_pose_size(landmarks, torso_size_multiplier=2.5):
    hips_center = get_center_point(landmarks, Landmark.LEFT_HIP,
                                 Landmark.RIGHT_HIP)

    shoulders_center = get_center_point(landmarks, Landmark.LEFT_SHOULDER,
                                      Landmark.RIGHT_SHOULDER)

    torso_size = tf.linalg.norm(shoulders_center - hips_center)
    pose_center_new = get_center_point(landmarks, Landmark.LEFT_HIP,
                                     Landmark.RIGHT_HIP)
    pose_center_new = tf.expand_dims(pose_center_new, axis=1)

    pose_center_new = tf.broadcast_to(pose_center_new,
                                    [tf.size(landmarks) // (17*2), 17, 2])

    d = tf.gather(landmarks - pose_center_new, 0, axis=0,
                name="dist_to_pose_center")

    max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))


    pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)
    return pose_size

def normalize_pose_landmarks(landmarks):
    pose_center = get_center_point(landmarks, Landmark.LEFT_HIP,
                                   Landmark.RIGHT_HIP)

    pose_center = tf.expand_dims(pose_center, axis=1)

    pose_center = tf.broadcast_to(pose_center,
                                  [tf.size(landmarks) // (17 * 2), 17, 2])
    landmarks = landmarks - pose_center

    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size
    return landmarks

def landmarks_to_embedding(landmarks_and_scores):
    reshaped_inputs = keras.layers.Reshape((17, 3))(landmarks_and_scores)
    landmarks = normalize_pose_landmarks(reshaped_inputs[:, :, :2])
    embedding = keras.layers.Flatten()(landmarks)
    return embedding

def preprocess_data(X_train):
    processed_X_train = []
    for i in range(X_train.shape[0]):
        embedding = landmarks_to_embedding(tf.reshape(tf.convert_to_tensor(X_train.iloc[i]), (1, 51)))
        processed_X_train.append(tf.reshape(embedding, (34)))
    return tf.convert_to_tensor(processed_X_train)

images_in_folder = os.path.join('yoga_poses', 'train')
csvs_out_path = 'train_data.csv'
train_preprocessor = Preprocessor(images_in_folder, csvs_out_path)
train_preprocessor.process()

images_in_folder = os.path.join('yoga_poses', 'test')
csvs_out_path = 'test_data.csv'
test_preprocessor = Preprocessor(images_in_folder, csvs_out_path)
test_preprocessor.process()
X, y, class_names = load_csv('train_data.csv')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)
X_test, y_test, _ = load_csv('test_data.csv')

processed_X_train = preprocess_data(X_train)
processed_X_val = preprocess_data(X_val)
processed_X_test = preprocess_data(X_test)

processed_X_train_numpy = []
for i in range(processed_X_train.shape[0]):
    processed_X_train_numpy.append(processed_X_train[i].numpy())

df = pd.DataFrame(processed_X_train_numpy)

csv_filename = 'processed_X_train.csv'
df.to_csv(csv_filename, index=False, header=False)


model = keras.Sequential([
    keras.layers.Dense(128, activation="relu", input_shape=(34,)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("Jumlah data dalam processed_X_train:", processed_X_train.shape[0])
print("Dimensi processed_X_train:", processed_X_train.shape)

print("Jumlah data dalam processed_X_val:", processed_X_val.shape[0])
print("Dimensi processed_X_val:", processed_X_val.shape)
model.summary()
earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20)
history = model.fit(processed_X_train, y_train, epochs=200, batch_size=32, validation_data=(processed_X_val, y_val), callbacks=[earlystopping])

loss, accuracy = model.evaluate(processed_X_test, y_test)
print('LOSS:', loss)
print('ACCURACY:', accuracy)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.figure(figsize=(10, 6))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

tflite_model_name = 'pose_estimation_model.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(tflite_model_name, 'wb') as f:
    f.write(tflite_model)

print(f"Model saved to {tflite_model_name} successfully!")