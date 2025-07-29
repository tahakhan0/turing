Segmentation app

## What this app does?
It uses the given image and the prompt to figure out all the objects that match the prompt.

## How does it do it?
It calls this [API](https://replicate.com/adirik/grounding-dino/api) with an image and prompt to get the bounding boxes.

## Your job is to:
1. Build a client for this app in client.py file that uses the given image path. The image could be:
   - URL - use this directly but have some error handling. We can send this url to the API.
   - Local image path - You've to open this image and get a base64 encoded image. Here are some [instructions](https://replicate.com/adirik/grounding-dino/api/learn-more#option-2-local-file)
     - Within the face-recognition app when we detect objects, 
2. Here is an example to run the model [link](https://replicate.com/adirik/grounding-dino/api/learn-more#run-the-model)