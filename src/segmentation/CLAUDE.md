Segmentation app

## What this app does?
It uses the given image and the prompt to figure out all the objects that match the prompt. The segments will allow the
app to label each object around a user's home. The goal is to build a repository of areas that are labelled and accepted
by the user. The user must also define which residents can access which areas. For example - they might not want a 
kid to be around a pool area, so we must be able to record that.

## How does it do it?
It calls this [API](https://replicate.com/adirik/grounding-dino/api) with an image and prompt to get the bounding boxes.

## Your job is to:
1. Build a client for this app in client.py file that uses the given image path. The image could be:
   - URL - use this directly but have some error handling. We can send this url to the API.
   - Local image path - You've to open this image and get a base64 encoded image. Here are some [instructions](https://replicate.com/adirik/grounding-dino/api/learn-more#option-2-local-file)
2. Here is an example to run the model [link](https://replicate.com/adirik/grounding-dino/api/learn-more#run-the-model)
3. Within the face-recognition app when a user submits a video path, it uses yolo to convert that image into frames. 
   Your job is to modify that piece of code so that it also saves those frames using the `storage` module. Only 
   store unique frames so that we avoid running the segmentation model on duplicate frames That way the
   frame is saved for the user. We can use this user id parameter to fetch the image and use it for segmentation.
4. With the output you received (here is the [schema](https://replicate.com/adirik/grounding-dino/api/schema)) store 
   them using the storage module and associate with the given user id.
5. In `src/interfaces/segmentation` modify the UI so that when the user completes face-recognition, they should be able
   to get to a page that has already segmented their images and ready for the user to either approve or deny the 
   segmentations. These images will be built by us using the bounding boxes and labels returned by replicate. If 
   user denies it don't save the segments.

### Sample response from the API
```json
{
    "detections": [
      {
        "bbox": [
          19,
          204,
          408,
          563
        ],
        "label": "pink mug",
        "confidence": 0.8077122569084167
      },
      {
        "bbox": [
          545,
          263,
          952,
          650
        ],
        "label": "pink mug",
        "confidence": 0.7644544839859009
      },
      {
        "bbox": [
          416,
          60,
          764,
          380
        ],
        "label": "pink mug",
        "confidence": 0.4754282832145691
      },
      {
        "bbox": [
          909,
          161,
          1078,
          487
        ],
        "label": "pink mug",
        "confidence": 0.43150201439857483
      }
    ],
    "result_image": "https://pbxt.replicate.delivery/oJDgostFveUPEiUVnTRM9MnL7rxEMXIuy65E6K2X638PfFyRA/result.png"
  }
```