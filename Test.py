import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import numpy as np
class_names = ["Gilbert", "Twiggy"]
test_url = "https://lh4.googleusercontent.com/fbHL0z7dy5tuIhymyTeIBABo0_9EwQAEcmpxxt_4Mc-b1xxyP93myudQNK0FwxrAwIk=w2400"
test_path = tf.keras.utils.get_file('gukbegerwg', origin=test_url)
img_height = 240
img_width = 240
img = tf.keras.utils.load_img(
    test_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch


TF_MODEL_FILE_PATH = 'model.tflite' # The default path to the saved TensorFlow Lite model

interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)


classify_lite = interpreter.get_signature_runner('serving_default')
classify_lite
predictions_lite = classify_lite(rescaling_1_input=img_array)['dense_1']
score_lite = tf.nn.softmax(predictions_lite)
print(np.argmax(score_lite))
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
)