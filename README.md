# app_cataract_detector_backend

Backend application for cataract detection using Vertex AI's SAM (Segment Anything Model)

## Table of Contents
1. [Setup Instructions](#setup-instructions)
2. [Authentication](#authentication)
3. [Running the Application](#running-the-application)
4. [API Documentation](#api-documentation)
5. [Code Architecture](#code-architecture)
6. [Endpoint Configuration](#endpoint-configuration)

## Setup Instructions

### Virtual Environment Creation

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```bash
python3 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Authentication

Currently using temporary credentials (for development only):

```bash
gcloud auth application-default login
```

**Note:** Need to create a service account for permanent public access (consult with Miguel/Saul)

## Running the Application

**Development mode:**
```bash
uvicorn app.main:app --reload
```

**Network accessible mode:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Access the API:**
- Local: http://127.0.0.1:8000/docs
- Network: http://[YOUR_IP]:8000/docs

## API Documentation

### Endpoints

**POST /predict/**
- Processes an image through the SAM model
- Returns segmented mask data

**Parameters:**
- `image`: Uploaded image file (JPG/PNG)

**Response:**
```json
{
  "status": "success|error",
  "filename": "original_filename.jpg",
  "mask_data": {
    "model_info": {...},
    "masks": [
      {"counts": "RLE_STRING", "size": [H,W]},
      ...
    ]
  }
}
```

## Code Architecture

### main.py
The application entry point that sets up the FastAPI app and includes routers.

**Components:**
- FastAPI app configuration
- Router imports (predict, results)

### ai_rest.py
Handles communication with Vertex AI and response processing.

**Functions:**

#### `predict_vertex_ai_rest(image_path: str) -> dict`
- Sends image to Vertex AI endpoint
- Processes the SAM model response

**Parameters:**
- `image_path`: Path to temporary image file

**Returns:**
- Processed mask data dictionary

#### `process_sam_response(response: dict) -> dict`
- Extracts and formats mask data from SAM response

**Parameters:**
- `response`: Raw response from Vertex AI

**Returns:**
- Structured dictionary with model info and masks

#### `preprocess_image(image_bytes: bytes, max_size=1024) -> bytes`
- Resizes and converts images to standard format

**Parameters:**
- `image_bytes`: Raw image data
- `max_size`: Maximum dimension for resizing

**Returns:**
- Processed image bytes in JPEG format

### predict.py
Contains the FastAPI router for prediction endpoint.

**Functions:**

#### `predict(image: UploadFile)`
- Main endpoint handler

**Parameters:**
- `image`: FastAPI UploadFile object

**Returns:**
- JSONResponse with processed results

**Flow:**
1. Receives uploaded file
2. Preprocesses image
3. Sends to Vertex AI
4. Returns formatted response

## Endpoint Configuration

### Building Vertex AI Endpoint URL

```python
def build_vertex_ai_endpoint(project_id: str, endpoint_id: str, region: str = "us-central1") -> str:
    """
    Builds complete Vertex AI endpoint URL
    
    Args:
        project_id: GCP project ID (e.g. '202589491823')
        endpoint_id: Deployed endpoint ID
        region: Deployment region
    
    Returns:
        Complete endpoint URL
    """
    base_url = (
        f"https://{endpoint_id}.{region}-{project_id}.prediction.vertexai.goog/"
        f"v1/projects/{project_id}/locations/{region}/endpoints/{endpoint_id}:predict"
    )
    return base_url
```

### Current Configuration

```python
PROJECT_ID = "202589491823"
ENDPOINT_ID = "6399717325074857984"
REGION = "us-central1"

endpoint_url = build_vertex_ai_endpoint(PROJECT_ID, ENDPOINT_ID, REGION)
```

## Response Format Example

```json
{
  "model_info": {
    "deployedModelId": "5403416853798191104",
    "model": "projects/202589491823/locations/us-central1/models/segment_anything__sam_-1751722913787",
    "modelDisplayName": "segment_anything__sam_-1751722913787"
  },
  "masks": [
    {
      "counts": "Raf4b0Y<8I5K7J5K4M2N3L3N3M2N3M2N2N1O2O1N3M2N101N2N2O1N2O0O2O1N101O0O2O001N101O1O0O2O00001O000O101O00001O0O101O000O2O0000WM]FV2c9jM]FV2c9jM]FV2c9iM^FW2a9jM_FV2a9kM^FU2b9d01N100000000O100O1000001N1000001N100O101N100O2O0O101N2O0O2N101N1O2N2N2N2N2O1N3L4M2N2N2O1N2M4L4L3M4L5K4L5I:Ee]Z4",
      "size": [415, 830]
    }
  ]
}
```

## Troubleshooting

### Authentication Errors:
- Run `gcloud auth application-default login`
- Verify service account permissions

### Image Processing Errors:
- Ensure images are <10MB
- Use standard formats (JPEG/PNG)

### Endpoint Errors:
- Verify endpoint is active in GCP console
- Check project/endpoint IDs