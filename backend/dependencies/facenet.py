import tensorflow as tf
from keras_facenet import FaceNet

# Disable Mac GPU (MPS) if present due to TF1 and TF2 mismatch issues
tf.config.set_visible_devices([], "GPU")


class FaceNetModel:
    """
    FaceNet Model in use:
    - Face embeddings are extracted using FaceNet, trained on the VGGFace2 dataset, originally implemented by @davidsandberg.
    - Faces are preprocessed by tight cropping using MTCNN, implemented by @ipazc.
    - Both FaceNet and MTCNN are wrapped and provided via the keras-facenet Python library by @faustomorales.
    """

    def __init__(self):
        self.model = FaceNet(key="20180402-114759")

    def get_embeddings(self, img_file_buffer):
        return self.model.extract(img_file_buffer, threshold=0.95)

    def read_image(self, img_file_buffer):
        img = tf.io.decode_image(img_file_buffer, channels=3)
        img = tf.cast(img, tf.float32).numpy()
        if img is None or img.size == 0 or img.ndim != 3:
            raise ValueError(f"Empty or invalid image for with shape {img.shape}")
        return img


facenet_model = FaceNetModel()
