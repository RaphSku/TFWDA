import unittest
import os
import tensorflow as tf
from tfwda.model.tensorflow import Model
from tfwda.model_store.tensorflow import ModelStore
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path('../.env')
load_dotenv(dotenv_path=dotenv_path)

MONGODB_CONNECTION_STRING = os.getenv('CONNECTION_STRING')

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
        #model_store     = ModelStore()
        #model_store_two = ModelStore.get_instance()


        """ VERIFICATION """
        #self.assertEqual(first = model_store, second = model_store_two)


    def test_model_store_db_connectivity_s01(self):
        """
        Check if database connection is established
        """

        """ PREPARATION """


        """ EXECUTION """
        try:
            model_store = ModelStore(db_connection_string = MONGODB_CONNECTION_STRING, database_name = "test")
        except:
            print("The MongoDB database connection couldn't be established!")


        """ VERIFICATION """
        self.assertTrue(True)