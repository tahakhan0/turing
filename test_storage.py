#!/usr/bin/env python3
"""
Test script to verify the PersistentStorage system works correctly

docker-compose run --rm turing-service python test_storage.py
"""

import sys
import os
import numpy as np
import tempfile
import shutil

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from storage.persistent_storage import PersistentStorage

def test_face_encodings():
    """Test face encodings storage and retrieval"""
    print("Testing face encodings...")
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    storage = PersistentStorage(temp_dir)
    
    try:
        # Test data
        user_id = "test_user_123"
        test_encodings = {
            "names": ["Alice", "Bob", "Charlie"],
            "encodings": [
                np.array([0.1, 0.2, 0.3, 0.4]),
                np.array([0.5, 0.6, 0.7, 0.8]),
                np.array([0.9, 1.0, 1.1, 1.2])
            ]
        }
        
        # Test saving
        print(f"  Saving face encodings for user {user_id}...")
        result = storage.save_face_encodings(user_id, test_encodings)
        print(f"  Saved to: {result}")
        
        # Test loading
        print(f"  Loading face encodings for user {user_id}...")
        loaded_data = storage.load_face_encodings(user_id)
        
        if loaded_data is None:
            print("  ❌ ERROR: Failed to load face encodings")
            return False
            
        # Verify data integrity
        if loaded_data["names"] != test_encodings["names"]:
            print("  ❌ ERROR: Names don't match")
            return False
            
        if len(loaded_data["encodings"]) != len(test_encodings["encodings"]):
            print("  ❌ ERROR: Number of encodings don't match")
            return False
            
        # Check if encodings are close (account for numpy array conversion)
        for i, (original, loaded) in enumerate(zip(test_encodings["encodings"], loaded_data["encodings"])):
            if not np.allclose(original, loaded):
                print(f"  ❌ ERROR: Encoding {i} doesn't match")
                return False
                
        print("  ✅ Face encodings test passed!")
        return True
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

def test_person_embeddings():
    """Test person embeddings storage and retrieval"""
    print("Testing person embeddings...")
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    storage = PersistentStorage(temp_dir)
    
    try:
        user_id = "test_user_456"
        
        # Test adding individual embeddings
        print("  Testing individual embedding updates...")
        
        # Add Alice's embedding
        alice_features = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        result = storage.update_person_embedding(user_id, "Alice", alice_features)
        if not result:
            print("  ❌ ERROR: Failed to save Alice's embedding")
            return False
            
        # Add Bob's embedding
        bob_features = np.array([6.6, 7.7, 8.8, 9.9, 10.0])
        result = storage.update_person_embedding(user_id, "Bob", bob_features)
        if not result:
            print("  ❌ ERROR: Failed to save Bob's embedding")
            return False
            
        # Test loading
        print("  Loading person embeddings...")
        loaded_data = storage.load_person_embeddings(user_id)
        
        if loaded_data is None:
            print("  ❌ ERROR: Failed to load person embeddings")
            return False
            
        if "persons" not in loaded_data:
            print("  ❌ ERROR: 'persons' key not found in loaded data")
            return False
            
        persons = loaded_data["persons"]
        
        if "Alice" not in persons or "Bob" not in persons:
            print("  ❌ ERROR: Person names not found in loaded data")
            return False
            
        # Verify Alice's features
        if not np.allclose(alice_features, np.array(persons["Alice"])):
            print("  ❌ ERROR: Alice's features don't match")
            return False
            
        # Verify Bob's features
        if not np.allclose(bob_features, np.array(persons["Bob"])):
            print("  ❌ ERROR: Bob's features don't match")
            return False
            
        print("  ✅ Person embeddings test passed!")
        return True
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

def test_face_labels():
    """Test face labels storage and retrieval"""
    print("Testing face labels...")
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    storage = PersistentStorage(temp_dir)
    
    try:
        user_id = "test_user_789"
        test_labels = {
            "labeled_faces": [
                {
                    "video_path": "/test/video1.mp4",
                    "frame_number": 100,
                    "bbox": {"x1": 10, "y1": 20, "x2": 50, "y2": 60},
                    "person_name": "Alice",
                    "detection_type": "face"
                },
                {
                    "video_path": "/test/video2.mp4", 
                    "frame_number": 200,
                    "bbox": {"x1": 30, "y1": 40, "x2": 70, "y2": 80},
                    "person_name": "Bob",
                    "detection_type": "person"
                }
            ]
        }
        
        # Test saving
        print(f"  Saving face labels for user {user_id}...")
        result = storage.save_face_labels(user_id, test_labels)
        print(f"  Saved to: {result}")
        
        # Test loading
        print(f"  Loading face labels for user {user_id}...")
        loaded_data = storage.load_face_labels(user_id)
        
        if loaded_data is None:
            print("  ❌ ERROR: Failed to load face labels")
            return False
            
        if "labeled_faces" not in loaded_data:
            print("  ❌ ERROR: 'labeled_faces' key not found")
            return False
            
        if len(loaded_data["labeled_faces"]) != 2:
            print("  ❌ ERROR: Wrong number of labeled faces")
            return False
            
        # Check first label
        label1 = loaded_data["labeled_faces"][0]
        if label1["person_name"] != "Alice" or label1["frame_number"] != 100:
            print("  ❌ ERROR: First label data doesn't match")
            return False
            
        # Check second label
        label2 = loaded_data["labeled_faces"][1] 
        if label2["person_name"] != "Bob" or label2["frame_number"] != 200:
            print("  ❌ ERROR: Second label data doesn't match")
            return False
            
        print("  ✅ Face labels test passed!")
        return True
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

def test_user_isolation():
    """Test that different users' data is properly isolated"""
    print("Testing user isolation...")
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    storage = PersistentStorage(temp_dir)
    
    try:
        user1_id = "user_001"
        user2_id = "user_002"
        
        # Save data for user 1
        user1_features = np.array([1.0, 2.0, 3.0])
        storage.update_person_embedding(user1_id, "Alice", user1_features)
        
        # Save data for user 2
        user2_features = np.array([4.0, 5.0, 6.0])
        storage.update_person_embedding(user2_id, "Alice", user2_features)
        
        # Load data for both users
        user1_data = storage.load_person_embeddings(user1_id)
        user2_data = storage.load_person_embeddings(user2_id)
        
        if user1_data is None or user2_data is None:
            print("  ❌ ERROR: Failed to load user data")
            return False
            
        # Verify that each user has their own Alice with different features
        user1_alice = np.array(user1_data["persons"]["Alice"])
        user2_alice = np.array(user2_data["persons"]["Alice"])
        
        if np.allclose(user1_alice, user2_alice):
            print("  ❌ ERROR: User data not properly isolated")
            return False
            
        if not np.allclose(user1_alice, user1_features):
            print("  ❌ ERROR: User 1's data corrupted")
            return False
            
        if not np.allclose(user2_alice, user2_features):
            print("  ❌ ERROR: User 2's data corrupted")
            return False
            
        print("  ✅ User isolation test passed!")
        return True
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

def main():
    """Run all tests"""
    print("🧪 Testing PersistentStorage system...\n")
    
    tests = [
        test_face_encodings,
        test_person_embeddings, 
        test_face_labels,
        test_user_isolation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ ERROR: {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print(f"📊 Test Results:")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📈 Success Rate: {passed}/{passed + failed} ({100 * passed / (passed + failed):.1f}%)")
    
    if failed == 0:
        print("\n🎉 All tests passed! The PersistentStorage system is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())