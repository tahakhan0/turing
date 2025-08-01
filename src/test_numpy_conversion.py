#!/usr/bin/env python3
"""
Test script to verify numpy data type conversion for JSON serialization.
"""

import json
import numpy as np
from face_recognition.yolo_service import convert_numpy_to_python
from face_recognition.schemas import BoundingBox, Detection

def test_numpy_conversion():
    """Test the convert_numpy_to_python function with various numpy types."""
    
    print("Testing numpy conversion function...")
    
    # Test basic numpy types
    test_cases = {
        "numpy int64": np.int64(42),
        "numpy int32": np.int32(42),
        "numpy float64": np.float64(3.14),
        "numpy float32": np.float32(3.14),
        "numpy bool": np.bool_(True),
        "numpy array": np.array([1, 2, 3, 4]),
        "nested dict with numpy": {
            "confidence": np.float64(0.85),
            "bbox": {
                "x1": np.int64(100),
                "y1": np.int64(200),
                "x2": np.int64(300),
                "y2": np.int64(400)
            }
        },
        "list with numpy": [np.int32(1), np.float32(2.5), np.bool_(False)]
    }
    
    for name, test_data in test_cases.items():
        print(f"\nTesting {name}:")
        print(f"  Original: {test_data} (type: {type(test_data)})")
        
        converted = convert_numpy_to_python(test_data)
        print(f"  Converted: {converted} (type: {type(converted)})")
        
        try:
            json_str = json.dumps(converted)
            print(f"  JSON serializable: ✓")
        except TypeError as e:
            print(f"  JSON serializable: ✗ - {e}")
    
    # Test with Pydantic models
    print("\nTesting with Pydantic models:")
    bbox = BoundingBox(
        x1=np.int64(100),
        y1=np.int64(200), 
        x2=np.int64(300),
        y2=np.int64(400)
    )
    
    detection = Detection(
        bbox=bbox,
        confidence=np.float64(0.85),
        class_name="person"
    )
    
    print(f"Detection object created: {detection}")
    
    # Convert to dict and then convert numpy types
    detection_dict = detection.model_dump()
    converted_detection = convert_numpy_to_python(detection_dict)
    
    try:
        json_str = json.dumps(converted_detection)
        print(f"Detection JSON serializable: ✓")
        print(f"JSON length: {len(json_str)} characters")
    except TypeError as e:
        print(f"Detection JSON serializable: ✗ - {e}")

if __name__ == "__main__":
    test_numpy_conversion()