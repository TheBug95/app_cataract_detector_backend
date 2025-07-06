# Cataract Detector Backend

Backend application for cataract detection using Vertex AI's SAM (Segment Anything Model) with advanced mask processing and classification algorithms.

## Table of Contents
1. [Setup Instructions](#setup-instructions)
2. [Authentication](#authentication)
3. [Running the Application](#running-the-application)
4. [API Documentation](#api-documentation)
5. [Code Architecture](#code-architecture)
6. [Endpoint Configuration](#endpoint-configuration)
7. [Mask Processing Pipeline](#mask-processing-pipeline)
8. [Response Examples](#response-examples)

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

### Predict Endpoint

**Endpoint:** `POST /predict/`

**Description:** Process image through Vertex AI SAM model to detect cataracts

**Parameters:**
- `image`: Uploaded image file (multipart/form-data)

**Supported Formats:**
- JPEG
- PNG (will be converted to JPEG)
- Non-transparent images recommended

**Response Types:**

#### 1. Positive Detection (Cataract Found)
```json
{
    "status": "success",
    "result": {
        "prediction": "Cataract",
        "confidence": "high",
        "best_mask": {
            "counts": "bUU36d<>E7J7K3L4L4M2N3L3N2N3M2N2N2N2N2N2N101N2N2O0O2N2O001N2O000O2O0O2O001O0O2O00001N1000001O0000000000000O100000000000000001O00000O100000000O101O000O2O0O2N100O2O0O2O0O2O0O2N2O0O2N2N3M2N2N2N2N2O1M3N3L3M4L4L4K6K5J8Gmo^3",
            "size": [417, 626]
        },
        "total_masks_processed": 13
    }
}
```

#### 2. Negative Detection (No Cataract)
```json
{
    "status": "success",
    "result": {
        "prediction": "No Cataract",
        "message": "No cataracts found in the image",
        "total_masks_processed": 13
    }
}
```

#### 3. Error Responses
```json
{
    "detail": {
        "error": "Image processing error",
        "details": "Specific error message",
        "suggestion": "Try with a JPG or non-transparent PNG image"
    }
}
```

**Response Fields:**
- `status`: Operation status ("success" or "error")
- `result.prediction`: Classification result ("Cataract" or "No Cataract")
- `result.confidence`: Confidence level (currently "high" for positive detections)
- `result.best_mask`: RLE encoded mask data for the best cataract detection
- `result.best_mask.counts`: RLE (Run Length Encoding) compressed mask string
- `result.best_mask.size`: Array with [height, width] of the mask
- `result.total_masks_processed`: Number of masks analyzed by the SAM model

## Code Architecture

### Core Components

1. **FastAPI Router** (`/predict`)
   - Handles image upload and validation
   - Manages temporary file processing
   - Coordinates between SAM model and mask processing

2. **Vertex AI Integration** (`ai_rest.py`)
   - Connects to Google Cloud Vertex AI SAM endpoint
   - Handles authentication and API communication
   - Processes SAM model responses

3. **Mask Processing Pipeline** (`maskDetector/execution.py`)
   - Analyzes masks using KDE (Kernel Density Estimation)
   - Applies cataract classification algorithms
   - Selects best mask based on confidence scores

### Processing Flow

```
Image Upload → Preprocessing → SAM Model → Mask Processing → Classification → Response
```

## Endpoint Configuration

### Vertex AI SAM Integration

The application connects to Google Cloud Vertex AI using the following configuration:

```python
# Endpoint URL (dedicated domain)
url = (
    "https://6903909377107820544.us-central1-202589491823.prediction.vertexai.goog/"
    "v1/projects/202589491823/locations/us-central1/endpoints/6903909377107820544:predict"
)
```

**Key Functions:**

#### `predict_vertex_ai_rest(image_path: str) -> dict`
Sends image to Vertex AI SAM endpoint and returns processed masks.

**Parameters:**
- `image_path`: Path to the image file

**Returns:**
- Dictionary containing processed mask data with RLE encoded masks

#### `preprocess_image(image_bytes: bytes, max_size: int = 1024) -> bytes`
Resizes image while maintaining aspect ratio and converts to JPEG format.

**Parameters:**
- `image_bytes`: Image data in bytes
- `max_size`: Maximum dimension size (default: 1024)

**Returns:**
- Processed image bytes in JPEG format

**Features:**
- Automatic RGB conversion for RGBA images
- Aspect ratio preservation
- Quality optimization (85% JPEG quality)
- Error handling for unsupported formats

## Mask Processing Pipeline

### Overview
The mask processing pipeline implements a sophisticated cataract detection algorithm using KDE (Kernel Density Estimation) and prototype-based classification.

### Key Components

#### 1. Mask Decoding
- Converts RLE (Run Length Encoding) masks to binary arrays
- Validates mask dimensions against source image
- Filters out empty or invalid masks

#### 2. Feature Extraction
- Crops regions of interest from original image
- Generates embeddings using Vision Transformer (ViT)
- Calculates bounding boxes for each mask

#### 3. Classification Algorithm
- Uses pre-trained KDE models for each embedding dimension
- Calculates log-probability scores for each mask
- Applies threshold-based classification (θ_min ≤ score ≤ θ_max)

#### 4. Best Mask Selection
- Identifies all positive classifications (cataract detections)
- Selects mask with highest confidence score
- Returns original RLE data for the best mask

### Function: `process_masks()`

```python
def process_masks(
    maskData: List[Dict[str, Any]],
    processed_image: bytes,
    k: int = 36,
    visualize: bool = True
) -> Tuple[str, Optional[np.ndarray], Optional[Dict[str, Any]]]
```

**Parameters:**
- `maskData`: List of RLE mask dictionaries from SAM
- `processed_image`: Preprocessed image bytes
- `k`: Prototype index for KDE model selection (default: 36)
- `visualize`: Enable visualization during processing

**Returns:**
- `label`: "Cataract" or "No Cataract"
- `binary_mask`: NumPy array of the best mask (if found)
- `original_data`: Original RLE data of the best mask

### Processing Steps

1. **Image Loading**: Converts processed image bytes to PIL Image
2. **RLE Decoding**: Decodes all SAM masks using `decode_sam_rle()`
3. **Model Configuration**: Loads KDE statistics for prototype k
4. **Mask Evaluation**: 
   - Extracts bounding boxes for each mask
   - Crops regions from original image
   - Generates ViT embeddings
   - Calculates log-probability scores
   - Applies binary classification
5. **Best Selection**: Chooses highest-scoring positive mask
6. **Result Formatting**: Returns classification result and mask data

### Logging and Monitoring

The pipeline includes comprehensive logging:
- Mask processing progress
- Model configuration details
- Individual mask scores and predictions
- Final classification results
- Error handling and debugging information

### Visualization (Development Mode)

When `visualize=True`, the system displays:
- Original image with mask overlays
- Cropped regions for each mask
- Classification results and confidence scores
- Bounding box visualizations

## Response Examples

### Successful Cataract Detection
```json
{
    "status": "success",
    "result": {
        "prediction": "Cataract",
        "confidence": "high",
        "best_mask": {
            "counts": "bUU36d<>E7J7K3L4L4M2N3L3N2N3M2N2N2N2N2N2N101N2N2O0O2N2O001N2O000O2O0O2O001O0O2O00001N1000001O0000000000000O100000000000000001O00000O100000000O101O000O2O0O2N100O2O0O2O0O2O0O2N2O0O2N2N3M2N2N2N2N2O1M3N3L3M4L4L4K6K5J8Gmo^3",
            "size": [417, 626]
        },
        "total_masks_processed": 13
    }
}
```

### No Cataract Detected
```json
{
    "status": "success",
    "result": {
        "prediction": "No Cataract",
        "message": "No cataracts found in the image",
        "total_masks_processed": 13
    }
}
```

### Error Cases
```json
{
    "detail": {
        "error": "Image processing error",
        "details": "Cannot identify image file",
        "suggestion": "Try with a JPG or non-transparent PNG image"
    }
}
```

## Technical Notes

### RLE Mask Format
- **counts**: Compressed binary mask using Run Length Encoding
- **size**: [height, width] array specifying mask dimensions
- Compatible with COCO mask format and pycocotools

### Performance Considerations
- Images are resized to maximum 1024px to optimize processing time
- Temporary files are automatically cleaned up after processing
- KDE models are pre-loaded for efficient mask evaluation

### Security
- No permanent file storage - all processing uses temporary files
- Authentication required for Google Cloud Vertex AI access
- Input validation for image formats and sizes

## Dependencies

Key Python packages:
- `fastapi`: Web framework and API routing
- `uvicorn`: ASGI server for FastAPI
- `google-auth`: Google Cloud authentication
- `requests`: HTTP client for Vertex AI communication
- `PIL (Pillow)`: Image processing and manipulation
- `numpy`: Numerical operations for mask processing
- `matplotlib`: Visualization and plotting
- `pycocotools`: COCO mask format handling

## Future Enhancements

- Service account authentication for production deployment
- Batch processing for multiple images
- Advanced confidence scoring algorithms
- Real-time processing optimizations
- Enhanced error handling and recovery mechanisms