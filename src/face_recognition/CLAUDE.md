# Overview
The purpose of this app is to recognize faces and people. It does so using the yolov8 model and face-recognition library

# Goals
Be able to detect faces and people, and prompt the user to label each of them. If multiple people/faces are detected
each should have their bounding boxes with different colors and allow the user to label each of them. Each label is
saved using the user's user_id which is created via the web ui in src/interfaces/face-recognition/app.js

# Important notes:
- We should be able to label each person and face that is detected.
- We should be able to save labels in bulk.
- The auto-identify feature should first look into known encodings if the given user id is recorded earlier and try to 
  identify people it already knows. 