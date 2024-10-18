import unittest
import json
import numpy as np
from datetime import datetime, date, timedelta, time
from uuid import UUID
from src.encoder.json import CustomJSONEncoder

class TestCustomJSONEncoder(unittest.TestCase):

    def test_encode_uuid(self):
        obj = UUID('12345678123456781234567812345678')
        encoded = json.dumps(obj, cls=CustomJSONEncoder)
        print(f"Encoded: {encoded} \nObject: {obj}")
        self.assertEqual(encoded, '"12345678-1234-5678-1234-567812345678"')

    def test_encode_datetime(self):
        obj = datetime(2023, 10, 1, 12, 0, 0)
        encoded = json.dumps(obj, cls=CustomJSONEncoder)
        print(f"Encoded: {encoded} \nObject: {obj}")
        self.assertEqual(encoded, '"2023-10-01T12:00:00"')

    def test_encode_date(self):
        obj = date(2023, 10, 1)
        encoded = json.dumps(obj, cls=CustomJSONEncoder)
        print(f"Encoded: {encoded} \nObject: {obj}")
        self.assertEqual(encoded, '"2023-10-01"')

    def test_encode_timedelta(self):
        obj = timedelta(days=1, hours=2)
        encoded = json.dumps(obj, cls=CustomJSONEncoder)
        print(f"Encoded: {encoded} \nObject: {obj}")
        self.assertEqual(encoded, '"02:00:00"')

    def test_encode_time(self):
        obj = time(12, 0, 0)
        encoded = json.dumps(obj, cls=CustomJSONEncoder)
        print(f"Encoded: {encoded} \nObject: {obj}")
        self.assertEqual(encoded, '"12:00:00"')

    def test_encode_numpy_array(self):
        obj = np.array([1, 2, 3])
        encoded = json.dumps(obj, cls=CustomJSONEncoder)
        print(f"Encoded: {encoded} \nObject: {obj}")
        self.assertEqual(encoded, '[1, 2, 3]')

    def test_encode_numpy_int64(self):
        obj = np.int64(42)
        encoded = json.dumps(obj, cls=CustomJSONEncoder)
        print(f"Encoded: {encoded} \nObject: {obj}")
        self.assertEqual(encoded, '42')

    def test_encode_numpy_float64(self):
        obj = np.float64(3.14)
        encoded = json.dumps(obj, cls=CustomJSONEncoder)
        print(f"Encoded: {encoded} \nObject: {obj}")
        self.assertEqual(encoded, '3.14')

if __name__ == '__main__':
    # unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCustomJSONEncoder)
    unittest.TextTestRunner(verbosity=2).run(suite)
    # Output:
    
