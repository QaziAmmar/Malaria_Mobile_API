import cv2
import os
import numpy as np
import copy
from .seg_watershed_drwaqas import cell_segmentation
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


def test_print():
    print("test print for code")


def watershed_drwaqas_seg(image_path):
    print("watershed dr waqas code is running")
    respons_json_object = []
    annotated_img, individual_cell_images, json_object = cell_segmentation(image_path)
    prediction = check_image_malaria(individual_cell_images)
    # Merging cell segmentation and classification retuls
    for temp_json, temp_pred in zip(json_object, prediction):
        temp_json.update(temp_pred)
        respons_json_object.append(temp_json)
    print(respons_json_object)
    return respons_json_object


def red_blood_cell_segmentation(image_path):
    # Convert RGB to gray scale and improve contrast of the image
    rectangle_points = []
    # Convert RGB to gray scale and improve contrast of the image
    mean_gray = cv2.imread("mean_image.png")
    rgb_resized = cv2.imread(image_path)
    mean_gray = cv2.resize(mean_gray, (rgb_resized.shape[1], rgb_resized.shape[0]))

    # Convert RGB to gray scale and improve contrast of the image
    gray = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imge_clahe = clahe.apply(gray)

    # Subtract the Background (mean) image
    mean_subtracted = imge_clahe - mean_gray[:, :, 0]
    clone = mean_subtracted.copy()

    # Remove the pixels which are very close to the mean. 60 is selected after watching a few images
    mean_subtracted[mean_subtracted < 60] = 0

    # To separate connected cells, do the Erosion. The kernal parameters are randomly selected.
    kernel = np.ones((20, 20), np.uint8)
    mean_subtracted_erode = cv2.erode(mean_subtracted, kernel)
    kernel = np.ones((7, 7), np.uint8)
    closing = cv2.morphologyEx(mean_subtracted_erode, cv2.MORPH_CLOSE, kernel)
    _, contours_single_erode, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the regions which are still large and strongle collected and apply errosion again only to those regions
    mean_subtracted_erode_forLarge = copy.deepcopy(mean_subtracted_erode)
    area_c = []
    for c in contours_single_erode:
        (x, y, w, h) = cv2.boundingRect(c)
        area_c = w * h
        if area_c < 30000:
            mean_subtracted_erode_forLarge[y:y + h, x:x + w] = 0

    kernel = np.ones((20, 20), np.uint8)
    mean_subtracted_doubleerode_forLarge = cv2.erode(mean_subtracted_erode_forLarge, kernel)

    closing = cv2.morphologyEx(mean_subtracted_doubleerode_forLarge, cv2.MORPH_CLOSE, kernel)
    _, contours_double_erode, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Plot results with single errosion
    rgb_single_erode = copy.deepcopy(rgb_resized)
    for c in contours_single_erode:
        (x, y, w, h) = cv2.boundingRect(c)
        x1 = max(1, x - 15)
        y1 = max(1, y - 15)
        x2 = min(x + w + 15, rgb_resized.shape[1])
        y2 = min(y + h + 15, rgb_resized.shape[0])

        area_c = w * h

        rectangle_points.append({"x": x1, "y": y1, "h": y2 - y1, "w": x2 - x1})
        cv2.rectangle(img=rgb_single_erode, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 225), thickness=5)

    # Plot results with double errosion
    rgb_double_erode = copy.deepcopy(rgb_resized)

    for c in contours_single_erode:
        (x, y, w, h) = cv2.boundingRect(c)
        x1 = max(1, x - 15)
        y1 = max(1, y - 15)
        x2 = min(x + w + 15, rgb_resized.shape[1])
        y2 = min(y + h + 15, rgb_resized.shape[0])

        area_c = w * h
        if 2000 < area_c < 30000:
            rectangle_points.append({"x": x1, "y": y1, "h": y2 - y1, "w": x2 - x1})
            cv2.rectangle(img=rgb_double_erode, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 225), thickness=5)

    for c in contours_double_erode:
        (x, y, w, h) = cv2.boundingRect(c)

        x1 = max(1, x - 35)
        y1 = max(1, y - 35)
        x2 = min(x + w + 35, rgb_resized.shape[1])
        y2 = min(y + h + 35, rgb_resized.shape[0])
        area_c = w * h
        if area_c < 2000:
            continue
        else:
            rectangle_points.append({"x": x1, "y": y1, "h": y2 - y1, "w": x2 - x1})
            cv2.rectangle(img=rgb_double_erode, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 225), thickness=5)
    print(len(rectangle_points))
    # cv2.imwrite("rgb_resized.jpg", rgb_resized)
    cv2.imwrite("rgb_single_erode.png", rgb_single_erode)
    cv2.imwrite("rgb_double_erode.png", rgb_double_erode)
    return rectangle_points


def image_normalization(img):
    IMG_DIMS = (125, 125)
    img = cv2.resize(img, dsize=IMG_DIMS, interpolation=cv2.INTER_CUBIC)
    img = np.array(img, dtype=np.float32)
    return img


def check_image_malaria(images):
    # Model 1: CNN from Scratch

    img_list = []
    print("length of images" + str(len(images)))
    for temp_img in images:
        img = image_normalization(temp_img)
        img_list.append(img)

    test_data = np.array(list(img_list))

    print("img", len(test_data))
    model = get_model()
    test_img_scaled = test_data / 255.
    basic_cnn_preds = model.predict(test_img_scaled)

    train_labels = ["healthy", "malaria"]
    le = LabelEncoder()
    le.fit(train_labels)

    # train_labels_enc = le.transform(train_labels)

    basic_cnn_preds_labels = le.inverse_transform([1 if pred > 0.5 else 0
                                                   for pred in basic_cnn_preds.ravel()])
    prediction = []
    for temp_prediction, temp_confidance in zip(basic_cnn_preds_labels, basic_cnn_preds):
        prediction.append({"prediction": temp_prediction, "confidence": temp_confidance[0]})
    # prediction = {
    #     "prediction": basic_cnn_preds_labels[0],
    #     "confidence": basic_cnn_preds[0][0]
    # }

    return prediction


def get_model():
    INPUT_SHAPE = (125, 125, 3)
    inp = tf.keras.layers.Input(shape=INPUT_SHAPE)

    conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inp)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    flat = tf.keras.layers.Flatten()(pool3)

    hidden1 = tf.keras.layers.Dense(512, activation='relu')(flat)
    drop1 = tf.keras.layers.Dropout(rate=0.3)(hidden1)
    hidden2 = tf.keras.layers.Dense(512, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(rate=0.3)(hidden2)

    out = tf.keras.layers.Dense(1, activation='sigmoid')(drop2)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.summary()

    save_weight_path = "/Users/qaziammar/Documents/Thesis/Model_Result_Dataset/SavedModel/MalariaDetaction_DrMoshin/basic_cnn_IML_fineTune.h5"

    model.load_weights(save_weight_path)
    return model
