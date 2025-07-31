from src.monitoring import gemini_service
from PIL import Image
import numpy as np

def image_to_ndarray(image_path):
    """
    Reads an image from a specified path and converts it into a NumPy ndarray.

    Args:
        image_path (str): The file path to the image.

    Returns:
        numpy.ndarray: The image represented as a NumPy array, or None if an error occurs.
    """
    try:
        # Open the image file using Pillow
        img = Image.open(image_path)

        # Convert the Pillow Image object to a NumPy ndarray
        img_array = np.array(img)

        return img_array

    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    frame = image_to_ndarray("/Users/tahakhan/Desktop/i.png")
    gemini = gemini_service.GeminiAnalysisService()
    violation ={
        "person_name":"Andy",
        "segment": {},
        "violation_reason":"Unauthorized",
    }

    response = gemini.analyze_violation_frame(frame=frame,violation=violation)
    print(response)

main()