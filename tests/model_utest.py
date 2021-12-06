import unittest
import tensorflow as tf

from tfwda.model.tensorflow import Model


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        self.densenet121 = tf.keras.applications.densenet.DenseNet121(
                                include_top=True, weights='imagenet', input_tensor=None,
                                input_shape=None, pooling=None, classes=1000
                            )

    
    def test_model_initialization_s01(self):
        """
        The Model should ingest a tensorflow deep learning architecture model
        and extract relevant information about it
        """

        """ PREPARATION """


        """ EXECUTION """
        model = Model(name = "DenseNet121", architecture = self.densenet121)    


        """ VERIFICATION """
        self.assertEqual(first = "DenseNet121", second = model.name)
        self.assertEqual(first = self.densenet121, second = model.architecture)
