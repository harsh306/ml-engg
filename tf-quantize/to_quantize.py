"""
Convert tf-1.x and tf-2.x, models in saved_model format TO Tflite model

date: Feb 2020
author: harsh
"""
import tensorflow as tf


TFLITE_MODEL = 'path/to/model_lite'
SAVED_MODEL_DIR = 'path/to/saved_model/'


def to_quant_default():
    """
    Given a Saved model dir, this function exports
    TFlite model.
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    converter.post_training_quantize = True
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]  # you can use multiple options for model OPTIMIZE
    tflite_quant_model = converter.convert()
    file = open(TFLITE_MODEL+'model.tflite', 'wb')
    file.write(tflite_quant_model)

def to_quant_concrete_func():
    """
    Given a Saved model dir, this function exports
    TFlite model.
    """
    # optionally pass concrete_func, if you want to modify input shape
    # example for image
    model = tf.saved_model.load(SAVED_MODEL_DIR)
    w = 224
    h = 224
    concrete_func = model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape([None, w, h, 3])
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    converter.post_training_quantize = True
    converter.experimental_new_converter = True # currently in tf-nightly (2.0)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_quant_model = converter.convert()
    file = open(TFLITE_MODEL+'model.tflite', 'wb')
    file.write(tflite_quant_model)


def predict_tflite():

    input_data = None
    tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL+'model.tflite')
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    print("== Input details ==")
    print("shape:", input_details[0]['shape'])
    print("type:", input_details[0]['dtype'])
    print("\n== Output details ==")
    print("shape:", output_details[0]['shape'])
    print("type:", output_details[0]['dtype'])

    # #optionally can be used
    # tflite_interpreter.resize_tensor_input(input_details[0]['index'], (new_shape))

    tflite_interpreter.allocate_tensors()
    tflite_interpreter.set_tensor(input_details[0]['index'], input_data)

    tflite_interpreter.invoke()

    prediction = tflite_interpreter.get_tensor(output_details[0]['index'])

    return prediction

