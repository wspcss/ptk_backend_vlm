from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import uuid
import hashlib
import subprocess
import os
import aiohttp
import base64
import random
from pathlib import Path
from typing import Optional
import time
import cv2
from ultralytics import YOLO
from openai import AsyncOpenAI

app = FastAPI()

# Configuration
VLM_ENDPOINT = "http://192.168.1.10:8000/v1/chat/completions"
VLM_MODEL = None  # Will be fetched from /v1/models endpoint on startup
LLM_ENDPOINT = "http://192.168.1.10:9000/v1/chat/completions"
LLM_MODEL = None  # Will be fetched from /v1/models endpoint on startup
VIDEOS_DIR = Path("videos")
YOLO_MODEL = None  # YOLO model for person/vehicle detection

# Singapore areas for random location fallback
SINGAPORE_AREAS = [
    "Marina Bay", "Orchard Road", "Chinatown", "Little India",
    "Sentosa", "Clarke Quay", "Bugis",
    "Raffles Place", "Holland Village", "Tiong Bahru", "Katong",
    "Geylang", "Lavender", "Bishan", "Woodlands", "Tampines",
    "Jurong East", "Punggol", "Sengkang", "Bedok", "Yishun",
    "Ang Mo Kio", "Toa Payoh", "Queenstown", "Bukit Timah"
]



SYSTEM_PROMPT = """
You are an AI Incident Analyst. Output ONLY these two sections - NO reasoning, NO extra text.

Overview
[5-8 sentences summary should include incident category, what happened, participants, location, end status]

Chronological Timeline of Actions
00:00 - MM:SS: [Description]
MM:SS - MM:SS: [Description]
MM:SS - MM:SS: [Description]


Timeline Description Guidelines:
- Start: Initial scene state (vehicles, people, objects visible)
- Changes: New actions, movement shifts, escalations, interactions
- End: Final state or outcome of incident
- Group static periods - only new entries for meaningful state changes
- The final entry MUST end at the exact video duration provided in the user prompt.

Incident Categories: Traffic | Fire | Fighting | Unlawful Gathering | Uncategorized

RULES:
- NO PREAMBLE. Start immediately with "Overview".
- ZERO-BASE: Use a relative zero-start (00:00). Strictly ignore all on-screen CCTV/clock timestamps.
- NO ASSUMPTIONS: Describe only visible actions. Do not infer intent or events beyond the frames provided.
- NO newline between Timeline of Actions entry.
- Include in the overview section any road names or landmarks in singapore. If unable to identify, do not include
- Video may contain replay of the same event. State in your timeline if you see replay and describe the actions in the replay as well. The timeline should be strictly chronological, so if you see a replay of an event at 00:30 that originally happened at 00:10, you should include it in the timeline at 00:30 with a note that it is a replay of the event that happened at 00:10.
"""


async def get_vlm_model() -> Optional[str]:
    """Fetch the model name from the VLM endpoint"""
    try:
        models_url = VLM_ENDPOINT.replace("/chat/completions", "/models")
        async with aiohttp.ClientSession() as session:
            async with session.get(models_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    print(f"Failed to fetch models from {models_url}: HTTP {response.status}")
                    return None
                
                data = await response.json()
                
                # Extract the first model ID from the response
                if data.get("data") and len(data["data"]) > 0:
                    model_id = data["data"][0].get("id")
                    print(f"VLM models available: {[model['id'] for model in data['data']]}")
                    print(f"Successfully fetched VLM model: {model_id}")
                    return model_id
                else:
                    print("No models found in VLM endpoint response")
                    return None
    except Exception as e:
        print(f"Error fetching VLM model: {e}")
        return None


async def get_llm_model() -> Optional[str]:
    """Fetch the model name from the LLM endpoint"""
    try:
        models_url = LLM_ENDPOINT.replace("/chat/completions", "/models")
        async with aiohttp.ClientSession() as session:
            async with session.get(models_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    print(f"Failed to fetch models from {models_url}: HTTP {response.status}")
                    return None
                
                data = await response.json()
                
                # Extract the first model ID from the response
                if data.get("data") and len(data["data"]) > 0:
                    model_id = data["data"][0].get("id")
                    print(f"LLM models available: {[model['id'] for model in data['data']]}")
                    print(f"Successfully fetched LLM model: {model_id}")
                    return model_id
                else:
                    print("No models found in LLM endpoint response")
                    return None
    except Exception as e:
        print(f"Error fetching LLM model: {e}")
        return None


def load_yolo_model():
    """Load YOLO model for person/vehicle detection"""
    global YOLO_MODEL
    try:
        print("Loading YOLO model...")
        YOLO_MODEL = YOLO("yolov8n.pt")
        print("  ✓ YOLO model loaded successfully")
        return YOLO_MODEL
    except Exception as e:
        print(f"WARNING: Failed to load YOLO model: {e}")
        print("  YOLO detection will not be available")
        YOLO_MODEL = None
        return None


@app.on_event("startup")
async def startup_event():
    """Fetch the VLM and LLM models on startup, and load YOLO model"""
    global VLM_MODEL, LLM_MODEL
    VLM_MODEL = await get_vlm_model()
    if not VLM_MODEL:
        print("WARNING: Failed to fetch VLM model. The application may not function correctly.")
    
    LLM_MODEL = await get_llm_model()
    if not LLM_MODEL:
        print("WARNING: Failed to fetch LLM model. The application may not function correctly.")
    
    # Load YOLO model synchronously
    load_yolo_model()


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_video_duration(video_path: str) -> int:
    """Get video duration in seconds using ffprobe"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        return int(float(data['format']['duration']))
    except Exception as e:
        print(f"Error getting video duration: {e}")
        return 0


def extract_frames(video_path: str, video_name: str, frames_dir: str) -> bool:
    """Extract frames from video at 1fps with naming pattern: {video_name}_{seconds}.jpg"""
    try:
        # Get video duration to determine zero-padding
        duration = get_video_duration(video_path)
        
        # Determine frame pattern based on duration
        if duration > 99:
            # Use 3 digits for videos longer than 99 seconds
            frame_pattern = f"{video_name}_%03d.jpg"
        else:
            # Use 2 digits for videos 99 seconds or shorter
            frame_pattern = f"{video_name}_%02d.jpg"
        
        output_pattern = os.path.join(frames_dir, frame_pattern)
        
        # Extract frames at 1fps
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', 'fps=1',
            output_pattern,
            '-y'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return False
        
        # Verify frames were extracted
        frame_files = list(Path(frames_dir).glob("*.jpg"))
        if len(frame_files) == 0:
            print("No frames were extracted")
            return False
        
        print(f"Extracted {len(frame_files)} frames with pattern: {frame_pattern}")
        return True
        
    except subprocess.TimeoutExpired:
        print("Frame extraction timed out")
        return False
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return False


def load_frames_sorted(frames_dir: str) -> list[str]:
    """Load all frame files sorted by timestamp (extracted from filename)"""
    frame_files = list(Path(frames_dir).glob("*.jpg"))
    
    def extract_timestamp(filename: str) -> int:
        """Extract timestamp from filename like 'fight1_01.jpg' -> 1"""
        try:
            # Remove extension and split by underscore
            base = os.path.splitext(filename)[0]
            # Get the part after the last underscore (the timestamp)
            timestamp_str = base.split('_')[-1]
            return int(timestamp_str)
        except (ValueError, IndexError):
            return 0
    
    # Sort by extracted timestamp
    sorted_frames = sorted(frame_files, key=lambda f: extract_timestamp(f.name))
    return [str(f) for f in sorted_frames]


def download_video_from_url(url: str, output_path: str) -> bool:
    """Download video from URL using yt-dlp"""
    try:
        # Use yt-dlp to download video
        cmd = [
            'yt-dlp',
            '--format', 'mp4[height<=720]/best[ext=mp4]/best',  # Download best MP4, max 720p
            '--output', output_path,
            '--no-warnings',  # Suppress warnings
            '--no-playlist',  # Download single video only
            url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"yt-dlp error: {result.stderr}")
            return False
        
        print(f"Video downloaded successfully to {output_path}")
        return True
        
    except subprocess.TimeoutExpired:
        print("Video download timed out")
        return False
    except Exception as e:
        print(f"Error downloading video: {e}")
        return False


def format_time(seconds: int) -> str:
    """Convert seconds to 'Xmin:Ys' format"""
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes}min:{secs}s"


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 data URL"""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    base64_str = base64.b64encode(image_data).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_str}"


async def send_to_vlm(frames: list[str], media_uuid: str) -> str:
    """Send frames to VLM and get summary text response"""
    # Initialize OpenAI client with custom endpoint
    vlm_base_url = VLM_ENDPOINT.replace("/chat/completions", "")
    client = AsyncOpenAI(base_url=vlm_base_url, api_key="dummy")
    
    # Calculate last timestamp (frames are 1-second intervals, starting at 0s)
    num_frames = len(frames)
    last_timestamp = num_frames - 1
    time_str = format_time(last_timestamp)
    
    # Build dynamic prompt based on actual frame count
    user_prompt = f"""
    Analyze these {num_frames} frames (1s intervals). 
    **Constraint:** The video duration is exactly {last_timestamp} seconds. The timeline must start at 00:00 and conclude exactly at {time_str}.
    Video Chronological Timeline of Actions should not exceed {time_str}.
    Output NOW - no reasoning:
    Overview
    """
    
    # Build content array with prompt and frames
    content = [
        {"type": "text", "text": f"{user_prompt}\n\n:"}
    ]
    
    # Add each frame as base64 image
    for frame_path in frames:
        base64_image = image_to_base64(frame_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": base64_image}
        })
    
    print(f"\n{'='*50}")
    print(f"SYSTEM_PROMPT:\n{SYSTEM_PROMPT}")
    print(f"\n{'='*50}")
    print(f"USER_PROMPT:\n{user_prompt}")
    print(f"{'='*50}\n")

    try:
        # Call VLM using OpenAI client
        response = await client.chat.completions.create(
            model=VLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=4096,
            temperature=0.1,
            stop=["<|endofthought|>", "<|im_start|>", "<|im_end|>", "Wait,", "Actually,", "Looking at", "Let me", "I need to", "Let's", "The frames", "Refining", "Draft"],
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False}
            },
            timeout=300
        )
        
        # Extract content from response
        if response.choices and len(response.choices) > 0:
            reply = response.choices[0].message.content
            
            # Clean the response - remove markdown code blocks if present
            if reply.startswith("```json"):
                reply = reply[7:]
            if reply.startswith("```"):
                reply = reply[3:]
            if reply.endswith("```"):
                reply = reply[:-3]
            reply = reply.strip()
            
            return reply
        else:
            raise HTTPException(
                status_code=500,
                detail="No response content from VLM"
            )
            
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"VLM SERVICE UNAVAILABLE: {e}")
        print(f"{'='*50}\n")
        raise HTTPException(
            status_code=503,
            detail=f"VLM service unavailable: {str(e)}"
        )


async def send_to_llm_for_classification(summary: str) -> dict:
    """Send VLM summary to LLM for structured classification"""
    
    classification_prompt = f"""You are an incident classifier. Analyze the following incident summary and extract structured information.

Your analysis must include:
- Incident classification (Traffic/Fire/Fighting/Unlawful Gathering)
- Severity assessment (0-3, where 3 is most severe)
- Deepfake detection and authenticity score (0.0-1.0)
- Location identification (Singapore area if visible)
- Detailed summary and short summary

Required JSON output format (return ONLY this JSON, no other text):
{{
  "summary": "{summary}",
  "shortsummary": "One brief sentence describing of the incident",
  "incidentType": "Security or Traffic or Fire or Fighting or Unlawful Gathering",
  "severity": [0-3],
  "location": "Location name if visible, otherwise 'Unknown'",
  "deepfake": [true/false],
  "authenticity": [0.0-1.0]
}}

Summary to analyze:
{summary}"""
    
    # Initialize OpenAI client with custom endpoint
    llm_base_url = LLM_ENDPOINT.replace("/chat/completions", "")
    client = AsyncOpenAI(base_url=llm_base_url, api_key="dummy")
    
    try:
        # Call LLM using OpenAI client
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that returns only valid JSON, no other text."
                },
                {
                    "role": "user",
                    "content": classification_prompt
                }
            ],
            max_tokens=1024,
            temperature=0.1,
            timeout=60
        )
        
        # Extract content from response
        if response.choices and len(response.choices) > 0:
            reply = response.choices[0].message.content
            
            # Clean the response - remove markdown code blocks if present
            if reply.startswith("```json"):
                reply = reply[7:]
            if reply.startswith("```"):
                reply = reply[3:]
            if reply.endswith("```"):
                reply = reply[:-3]
            reply = reply.strip()
            
            # Parse JSON response
            try:
                result = json.loads(reply)
                
                # Validate and ensure all required fields exist
                result.setdefault("summary", summary)
                result.setdefault("shortsummary", "Unable to classify")
                result.setdefault("incidentType", "Unknown")
                result.setdefault("severity", 0)
                result.setdefault("location", "Unknown")
                result.setdefault("deepfake", False)
                result.setdefault("authenticity", 0.0)
                
                # Ensure types are correct
                result["severity"] = int(result["severity"]) if isinstance(result["severity"], (int, float, str)) else 0
                result["severity"] = max(0, min(3, result["severity"]))  # Clamp between 0-3
                result["deepfake"] = bool(result["deepfake"])
                result["authenticity"] = float(result["authenticity"]) if isinstance(result["authenticity"], (int, float, str)) else 0.0
                result["authenticity"] = max(0.0, min(1.0, result["authenticity"]))  # Clamp between 0.0-1.0
                
                # Fallback to random Singapore area if location is unknown
                if result.get("location") in ["Unknown", "", None]:
                    result["location"] = random.choice(SINGAPORE_AREAS)
                
                return result
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse LLM response as JSON: {e}")
                print(f"Raw response: {reply}")
                # Return default values on parse error
                return {
                    "summary": summary,
                    "shortsummary": "Unable to classify",
                    "incidentType": "Unknown",
                    "severity": 0,
                    "location": "Unknown",
                    "deepfake": False,
                    "authenticity": 0.0
                }
        else:
            print("No response content from LLM")
            # Return default values on error
            return {
                "summary": summary,
                "shortsummary": "Unable to classify",
                "incidentType": "Unknown",
                "severity": 0,
                "location": "Unknown",
                "deepfake": False,
                "authenticity": 0.0
            }
            
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"LLM SERVICE UNAVAILABLE: {e}")
        print(f"{'='*50}\n")
        # Return default values on error
        return {
            "summary": summary,
            "shortsummary": "Unable to classify",
            "incidentType": "Unknown",
            "severity": 0,
            "location": "Unknown",
            "deepfake": False,
            "authenticity": 0.0
        }


def detect_and_extract_entities(frames: list[str], entities_dir: str) -> int:
    """
    Detect persons and vehicles in frames using YOLO.
    Uses two-pass approach: find best frame, then extract entities from it.
    Saves cropped images to entities_dir.
    Returns total number of entities detected.
    """
    if YOLO_MODEL is None:
        print("YOLO model not available, skipping detection")
        return 0
    
    # Vehicle classes: bicycle(1), car(2), motorcycle(3), bus(5), truck(7)
    vehicle_classes = [1, 2, 3, 5, 7]
    person_class = [0]
    all_entity_classes = person_class + vehicle_classes
    
    print(f"\n{'='*50}")
    print("Running YOLO detection...")
    print(f"{'='*50}\n")
    
    # Create entities directory
    os.makedirs(entities_dir, exist_ok=True)
    
    # Clear existing entity files
    for f in os.listdir(entities_dir):
        if f.endswith('.jpg'):
            os.remove(os.path.join(entities_dir, f))
    
    frame_data = []
    
    # === PASS 1: Analyze all frames to find best frame ===
    print("Pass 1: Analyzing frames to find best frame...")
    for i, frame_path in enumerate(frames):
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"  [Frame {i+1}/{len(frames)}] Could not load {frame_path}")
            continue
        
        # Detect all entities (persons + vehicles)
        results = YOLO_MODEL(frame, conf=0.5, classes=all_entity_classes, verbose=False)
        
        total_detections = 0
        conf_sum = 0.0
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                total_detections += 1
                conf_sum += float(box.conf[0])
        
        avg_confidence = conf_sum / total_detections if total_detections > 0 else 0.0
        
        frame_data.append({
            'path': frame_path,
            'frame': frame,
            'total_detections': total_detections,
            'avg_confidence': avg_confidence
        })
        
        # print(f"  [Frame {i+1}/{len(frames)}] {os.path.basename(frame_path)}: {total_detections} detections (conf: {avg_confidence:.3f})")
    
    if not frame_data:
        print("No valid frames found for detection")
        return 0
    
    # Calculate average detections across all frames
    avg_total_detections = sum(f['total_detections'] for f in frame_data) / len(frame_data)
    
    # Filter frames above average
    above_average_frames = [f for f in frame_data if f['total_detections'] >= avg_total_detections]
    
    # Select winning frame (highest confidence among above-average frames)
    if above_average_frames:
        winning_frame_data = max(above_average_frames, key=lambda x: x['avg_confidence'])
    else:
        # Fallback: pick frame with highest confidence overall
        winning_frame_data = max(frame_data, key=lambda x: x['avg_confidence'])
    
    print(f"\nBest frame selected: {os.path.basename(winning_frame_data['path'])}")
    print(f"  Detections: {winning_frame_data['total_detections']}")
    print(f"  Avg confidence: {winning_frame_data['avg_confidence']:.3f}")
    
    # === PASS 2: Extract entities from winning frame ===
    print(f"\nPass 2: Extracting entities from best frame...")
    
    frame = winning_frame_data['frame']
    entity_count = 0
    
    # Detect and crop entities
    results = YOLO_MODEL(frame, conf=0.5, classes=all_entity_classes, verbose=False)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            
            # Crop entity
            entity_crop = frame[y1:y2, x1:x2]
            
            # Save entity image
            entity_filename = f"entity_{entity_count}.jpg"
            entity_path = os.path.join(entities_dir, entity_filename)
            cv2.imwrite(entity_path, entity_crop)
            
            entity_type = "person" if class_id == 0 else "vehicle"
            print(f"  ✓ Entity {entity_count}: {entity_type} (confidence={confidence:.2f})")
            
            entity_count += 1
    
    print(f"\nTotal entities extracted: {entity_count}")
    print(f"{'='*50}\n")
    
    return entity_count


# Upload endpoint
@app.post("/upload")
async def upload_data(post_file: UploadFile = File(...)):
    import time
    
    total_start = time.time()
    
    try:
        # Read file content
        content = await post_file.read()
        
        # Generate media_uuid based on content hash
        content_hash = hashlib.md5(content).hexdigest()
        media_uuid = str(uuid.UUID(content_hash[:32]))
        
        # Create directory for this video
        video_dir = VIDEOS_DIR / media_uuid
        frames_dir = video_dir / "frames"
        entities_dir = video_dir / "entities"
        video_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        entities_dir.mkdir(parents=True, exist_ok=True)
        
        # Save video file
        save_start = time.time()
        video_path = video_dir / "video.mp4"
        with open(video_path, 'wb') as f:
            f.write(content)
        save_time = time.time() - save_start
        
        # Get video name without extension for frame naming
        video_name = os.path.splitext(post_file.filename)[0]
        
        # Extract frames
        print(f"\n{'='*50}")
        print(f"Processing video upload:")
        print(f"  Filename: {post_file.filename}")
        print(f"  Content-Type: {post_file.content_type}")
        print(f"  File size: {len(content)} bytes")
        print(f"  Generated media_uuid: {media_uuid}")
        print(f"{'='*50}\n")
        
        extract_start = time.time()
        extract_frames(str(video_path), video_name, str(frames_dir))
        extract_time = time.time() - extract_start
        
        # Get video duration
        duration = get_video_duration(str(video_path))
        
        total_time = time.time() - total_start
        
        # Print timing information
        print(f"\n{'='*50}")
        print(f"API Call: /upload")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  - Save video: {save_time:.2f} seconds")
        print(f"  - Extract frames: {extract_time:.2f} seconds")
        print(f"  - Video duration: {duration} seconds")
        print(f"{'='*50}\n")
        
        # Return success response
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "File received and processed",
                "filename": post_file.filename,
                "content_type": post_file.content_type,
                "size": len(content),
                "media_uuid": media_uuid,
                "duration": duration,
                "deepfake": False
            }
        )
        
    except Exception as e:
        print(f"Error processing file: {type(e).__name__}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Error processing file",
                "error": str(e)
            }
        )


@app.post("/upload/")
async def upload_data_with_slash(post_file: UploadFile = File(...)):
    return await upload_data(post_file)


# Upload URL endpoint - accepts URL as query parameter
@app.post("/uploadurl")
async def upload_url(url: str):
    total_start = time.time()
    
    try:
        # Generate media_uuid based on URL hash
        url_bytes = url.encode('utf-8')
        url_hash = hashlib.md5(url_bytes).hexdigest()
        media_uuid = str(uuid.UUID(url_hash[:32]))
        
        # Create directory for this video
        video_dir = VIDEOS_DIR / media_uuid
        frames_dir = video_dir / "frames"
        entities_dir = video_dir / "entities"
        video_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        entities_dir.mkdir(parents=True, exist_ok=True)
        
        # Video path
        video_path = video_dir / "video.mp4"
        
        # Print URL information to console
        print(f"\n{'='*50}")
        print(f"Processing video upload from URL:")
        print(f"  URL: {url}")
        print(f"  Generated media_uuid: {media_uuid}")
        print(f"{'='*50}\n")
        
        # Download video
        download_start = time.time()
        success = download_video_from_url(url, str(video_path))
        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to download video from URL"
            )
        download_time = time.time() - download_start
        
        # Save video (already saved by download function)
        save_time = 0.0
        
        # Get video name from URL for frame naming
        video_name = f"video_{media_uuid[:8]}"
        
        # Extract frames
        extract_start = time.time()
        extract_frames(str(video_path), video_name, str(frames_dir))
        extract_time = time.time() - extract_start
        
        # Get video duration
        duration = get_video_duration(str(video_path))
        
        total_time = time.time() - total_start
        
        # Print timing information
        print(f"\n{'='*50}")
        print(f"API Call: /uploadurl")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  - Download video: {download_time:.2f} seconds")
        print(f"  - Save video: {save_time:.2f} seconds")
        print(f"  - Extract frames: {extract_time:.2f} seconds")
        print(f"  - Video duration: {duration} seconds")
        print(f"{'='*50}\n")
        
        # Return success response with media_uuid
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "URL processed successfully",
                "url": url,
                "media_uuid": media_uuid,
                "duration": duration,
                "deepfake": False
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing URL: {type(e).__name__}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Error processing URL",
                "error": str(e)
            }
        )


# Predict endpoint
@app.post("/predict/{media_uuid}")
async def predict(media_uuid: str):
    import time
    
    total_start = time.time()
    
    try:
        # Check if video directory exists
        video_dir = VIDEOS_DIR / media_uuid
        if not video_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Video with media_uuid {media_uuid} not found"
            )
        
        frames_dir = video_dir / "frames"
        if not frames_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Frames not found for media_uuid {media_uuid}"
            )
        
        # Load and sort frames
        load_start = time.time()
        frames = load_frames_sorted(str(frames_dir))
        load_time = time.time() - load_start
        
        if not frames:
            raise HTTPException(
                status_code=404,
                detail=f"No frames found for media_uuid {media_uuid}"
            )
        
        print(f"\n{'='*50}")
        print(f"Received predict request:")
        print(f"  media_uuid: {media_uuid}")
        print(f"  Frames to analyze: {len(frames)}")
        print(f"{'='*50}\n")
        
        # Send to VLM
        vlm_start = time.time()
        summary = await send_to_vlm(frames, media_uuid)
        vlm_time = time.time() - vlm_start
        
        # Send summary to LLM for classification
        llm_start = time.time()
        classification = await send_to_llm_for_classification(summary)
        llm_time = time.time() - llm_start
        
        # Run YOLO detection
        detect_start = time.time()
        entities_dir = video_dir / "entities"
        entities_count = detect_and_extract_entities(frames, str(entities_dir))
        detect_time = time.time() - detect_start
        
        # Extract values from classification result
        incident_type = classification.get("incidentType", "Unknown")
        severity = classification.get("severity", 0)
        location = classification.get("location", "Unknown")
        deepfake = classification.get("deepfake", False)
        authenticity = classification.get("authenticity", 0.0)
        shortsummary = classification.get("shortsummary", "Unable to classify")
        
        # Print prediction result
        print(f"\n{'='*50}")
        print(f"VLM Analysis Result:")
        print(f"  summary: {shortsummary}")
        print(f"  incidentType: {incident_type}")
        print(f"  severity: {severity}")
        print(f"  deepfake: {deepfake}")
        print(f"  authenticity: {authenticity}")
        print(f"  location: {location}")
        print(f"{'='*50}\n")
        
        total_time = time.time() - total_start
        
        # Print timing information
        print(f"\n{'='*50}")
        print(f"API Call: /predict")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  - Load frames: {load_time:.2f} seconds")
        print(f"  - VLM analysis: {vlm_time:.2f} seconds")
        print(f"  - LLM classification: {llm_time:.2f} seconds")
        print(f"  - YOLO detection: {detect_time:.2f} seconds")
        print(f"{'='*50}\n")
        
        # Return structured response
        return JSONResponse(
            status_code=200,
            content={
                "summary": summary,
                "deepfake": deepfake,
                "media_uuid": media_uuid,
                "incidentType": incident_type,
                "authenticity": authenticity,
                "severity": severity,
                "location": location,
                "shortsummary": shortsummary,
                "detections": {
                    "count": entities_count
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing predict request: {type(e).__name__}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Error processing prediction",
                "error": str(e)
            }
        )


@app.get("/video/{media_uuid}")
async def get_video(media_uuid: str):
    """Serve video file by media_uuid"""
    try:
        # Check if video directory exists
        video_path = VIDEOS_DIR / media_uuid / "video.mp4"
        
        if not video_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Video with media_uuid {media_uuid} not found"
            )
        
        # Return video file with proper content type
        return FileResponse(
            path=str(video_path),
            media_type="video/mp4",
            filename=f"{media_uuid}.mp4"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error serving video: {type(e).__name__}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Error serving video",
                "error": str(e)
            }
        )


@app.get("/thumbnails/{media_uuid}")
async def get_thumbnails_list(media_uuid: str):
    """Get list of all thumbnail frames for a video"""
    try:
        # Check if video directory exists
        video_dir = VIDEOS_DIR / media_uuid
        if not video_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Video with media_uuid {media_uuid} not found"
            )
        
        # Check if frames directory exists
        frames_dir = video_dir / "frames"
        if not frames_dir.exists():
            return JSONResponse(
                status_code=200,
                content={
                    "media_uuid": media_uuid,
                    "thumbnails": []
                }
            )
        
        # Get all frame files sorted by timestamp
        frame_files = list(frames_dir.glob("*.jpg"))
        
        # Extract timestamp from filename and sort
        def extract_timestamp(filename: str) -> int:
            """Extract timestamp from filename like 'video_uuid_01.jpg' -> 0 (0-based)"""
            try:
                base = os.path.splitext(filename)[0]
                timestamp_str = base.split('_')[-1]
                # Convert from 1-based to 0-based timestamp
                # File naming: _01, _02, _03 -> 0s, 1s, 2s
                return int(timestamp_str) - 1
            except (ValueError, IndexError):
                return 0
        
        sorted_frames = sorted(frame_files, key=lambda f: extract_timestamp(f.name))
        
        # Build thumbnail list with URLs
        thumbnails = []
        for frame_file in sorted_frames:
            timestamp = extract_timestamp(frame_file.name)
            thumbnails.append({
                "id": str(timestamp + 1),  # 1-based ID
                "filename": frame_file.name,
                "url": f"/api/thumbnails/{media_uuid}/{frame_file.name}",
                "timestamp": float(timestamp)  # Timestamp in seconds (0-based)
            })
        
        return JSONResponse(
            status_code=200,
            content={
                "media_uuid": media_uuid,
                "total_frames": len(thumbnails),
                "thumbnails": thumbnails
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting thumbnails list: {type(e).__name__}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Error getting thumbnails list",
                "error": str(e)
            }
        )


@app.get("/thumbnails/{media_uuid}/{frame_file}")
async def get_thumbnail(media_uuid: str, frame_file: str):
    """Serve individual thumbnail image file"""
    try:
        # Check if video directory exists
        video_dir = VIDEOS_DIR / media_uuid
        if not video_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Video with media_uuid {media_uuid} not found"
            )
        
        # Build frame file path
        frame_path = video_dir / "frames" / frame_file
        
        if not frame_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Frame {frame_file} not found for media_uuid {media_uuid}"
            )
        
        # Return image file with proper content type
        return FileResponse(
            path=str(frame_path),
            media_type="image/jpeg",
            filename=frame_file
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error serving thumbnail: {type(e).__name__}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Error serving thumbnail",
                "error": str(e)
            }
        )


@app.get("/entities/{media_uuid}")
async def get_entities_list(media_uuid: str):
    """Get list of all detected entity images for a video"""
    try:
        # Check if video directory exists
        video_dir = VIDEOS_DIR / media_uuid
        if not video_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Video with media_uuid {media_uuid} not found"
            )
        
        # Check if entities directory exists
        entities_dir = video_dir / "entities"
        if not entities_dir.exists():
            return JSONResponse(
                status_code=200,
                content={
                    "media_uuid": media_uuid,
                    "entities": []
                }
            )
        
        # Get all entity files sorted by number
        entity_files = list(entities_dir.glob("*.jpg"))
        
        # Extract number from filename and sort
        def extract_entity_number(filename: str) -> int:
            """Extract entity number from filename like 'entity_0.jpg' -> 0"""
            try:
                base = os.path.splitext(filename)[0]
                number_str = base.split('_')[-1]
                return int(number_str)
            except (ValueError, IndexError):
                return 0
        
        sorted_entities = sorted(entity_files, key=lambda f: extract_entity_number(f.name))
        
        # Build entity list with URLs (same format as thumbnails)
        entities = []
        for entity_file in sorted_entities:
            entity_id = extract_entity_number(entity_file.name)
            entities.append({
                "id": str(entity_id),
                "filename": entity_file.name,
                "url": f"/api/entities/{media_uuid}/{entity_file.name}"
            })
        
        return JSONResponse(
            status_code=200,
            content={
                "media_uuid": media_uuid,
                "total_entities": len(entities),
                "entities": entities
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting entities list: {type(e).__name__}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Error getting entities list",
                "error": str(e)
            }
        )


@app.get("/entities/{media_uuid}/{entity_file}")
async def get_entity(media_uuid: str, entity_file: str):
    """Serve individual entity image file"""
    try:
        # Check if video directory exists
        video_dir = VIDEOS_DIR / media_uuid
        if not video_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Video with media_uuid {media_uuid} not found"
            )
        
        # Build entity file path
        entity_path = video_dir / "entities" / entity_file
        
        if not entity_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Entity {entity_file} not found for media_uuid {media_uuid}"
            )
        
        # Return image file with proper content type
        return FileResponse(
            path=str(entity_path),
            media_type="image/jpeg",
            filename=entity_file
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error serving entity: {type(e).__name__}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Error serving entity",
                "error": str(e)
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4000)