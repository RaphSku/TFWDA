import unittest
import os
import tensorflow as tf
from tfwda.model.tensorflow import Model
from tfwda.model_store.tensorflow import ModelStore
from dotenv import load_dotenv

load_dotenv()

MONGODB_CONNECTION_STRING = os.getenv('CONNECTION_STRING')
PLOT_FOLDER_PATH          = os.getenv('PLOT_FOLDER_PATH')

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
        model_store = ModelStore(db_connection_string = MONGODB_CONNECTION_STRING, database_name = "test", 
                                 path_to_dir = PLOT_FOLDER_PATH, verbosity = True)
        model_store_two = ModelStore.get_instance(db_connection_string = MONGODB_CONNECTION_STRING, database_name = "test",
                                                  path_to_dir = PLOT_FOLDER_PATH, verbosity = True)


        """ VERIFICATION """
        self.assertEqual(first = model_store, second = model_store_two)


    def test_model_store_pipe_model_s01(self):
        """
        Pipe a model through the model store
        """

        """ PREPARATION """
        model       = Model(name = "DenseNet121", architecture = self.densenet121)
        model_store = ModelStore.get_instance(db_connection_string = MONGODB_CONNECTION_STRING, database_name = "test",
                                              path_to_dir = PLOT_FOLDER_PATH, verbosity = True)


        """ EXECUTION """
        model_store.pipe_models([model])


        """ VERIFICATION """
        self.assertTrue(True)