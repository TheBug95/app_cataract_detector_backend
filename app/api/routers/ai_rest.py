import requests
import base64
import json
import google.auth
from google.auth.transport.requests import Request
from PIL import Image
import io

def predict_vertex_ai_rest(image_path: str) -> dict:
    """
    Send image to Vertex AI SAM endpoint and return processed masks
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary containing processed mask data with keys:
        - 'counts': RLE encoded mask string
        - 'size': [height, width] of mask
    """
    # Get credentials
    credentials, _ = google.auth.default()
    credentials.refresh(Request())
    token = credentials.token

    # Load and preprocess image
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    # Convert to base64
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Build instance for SAM model
    instance = {"image": image_b64}
    body = {"instances": [instance]}

    # Endpoint URL (dedicated domain)
    url = (

        "https://3608891831477075968.us-east1-202589491823.prediction.vertexai.goog/"
        "v1/projects/202589491823/locations/us-east1/endpoints/3608891831477075968:predict"
    )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # Send request
    response = requests.post(url, headers=headers, json=body)

    if response.status_code != 200:
        error_msg = response.text
        try:
            error_json = response.json()
            error_msg = json.dumps(error_json, indent=2)
        except:
            pass
        raise Exception(f"Error {response.status_code}: {error_msg}")

    # Process and return mask data
    return process_sam_response(response.json())

def process_sam_response(response: dict) -> dict:
    """
    Process SAM response to extract and format mask data
    
    Args:
        response: Raw response from Vertex AI
        
    Returns:
        Processed dictionary containing mask information
    """
    processed_data = {
        "model_info": {
            "deployedModelId": response.get("deployedModelId"),
            "model": response.get("model"),
            "modelDisplayName": response.get("modelDisplayName")
        },
        "masks": []
    }
    
    for prediction in response.get("predictions", []):
        for mask_data in prediction.get("masks_rle", []):
            processed_data["masks"].append({
                "counts": mask_data["counts"],
                "size": mask_data["size"]
            })
    
    # Print verification
    print("Processed Mask Data:")
    print(json.dumps(processed_data, indent=2))
    
    return processed_data

def preprocess_image(image_bytes: bytes, max_size: int = 1024) -> bytes:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image_bytes: Image data in bytes
        max_size: Maximum dimension size
        
    Returns:
        Processed image bytes
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            else:
                img = img.convert('RGB')
        
        # Resize if needed
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        
        # Convert to JPEG
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=85)
        return output.getvalue()
    
    except Exception as e:
        raise ValueError(f"Image processing error: {str(e)}")