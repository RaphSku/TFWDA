import unittest
import tensorflow as tf
from tfwda.model.tensorflow import Model
from tfwda.model_store.tensorflow import ModelStore


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        self.densenet121 = tf.keras.applications.densenet.DenseNet121(
                                include_top=True, weights='imagenet', input_tensor=None,
                                input_shape=None, pooling=None, classes=1000
                            )


    def test_model_store_initialization_s01(self):
        """
        Check if the model store is a Singleton
        """

        """ PREPARATION """


        """ EXECUTION """
        model_store     = ModelStore()
        model_store_two = ModelStore.get_instance()


        """ VERIFICATION """
        self.assertEqual(first = model_store, second = model_store_two)