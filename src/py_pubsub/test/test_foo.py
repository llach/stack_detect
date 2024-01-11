import unittest

"""Unit tests"""
import py_pubsub.publisher_member_function as publisher


class TestOperations(unittest.TestCase):

    def test_math(self):
        """Math test"""
        self.assertEqual(3+3, 6)

    def test_function_in_publisher(self):
        """Sample function test"""
        self.assertEqual(publisher.test_function(), "test function")
