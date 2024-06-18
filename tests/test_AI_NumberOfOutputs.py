import unittest
from PyQt5.QtWidgets import QApplication
import sys
import os
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from view.MainView import MainWindow
from view.SelectMethod import SelectMethod
from controller.AI import AI

class TestAI_NumberOfOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)
        else:
            cls.app = QApplication.instance()
    
    def setUp(self):
        self.window = MainWindow(initial_folder=".")
        self.select_method = SelectMethod(self.window)
    
    def test_AI_Method_NumberOfOutputs(self):
        ai_instance = AI(self.window, self.window.initial_directory)
        self.assertTrue(os.path.isfile(ai_instance.model_path), f"Model file does not exist: {ai_instance.model_path}")
        
        script_dir = os.path.dirname(__file__)
        test_image_path = os.path.join(script_dir, "..", "test images", "image1.jpg")
        self.assertTrue(os.path.isfile(test_image_path), f"Test image file does not exist: {test_image_path}")
        
        prediction = ai_instance.classify_image(test_image_path)
        print(f"Prediction: {prediction}")
        self.assertEqual(len(prediction), 6, "The prediction should have 6 elements")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'app'):
            del cls.app

if __name__ == '__main__':
    unittest.main()
