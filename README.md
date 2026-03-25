# VLM-Powered Incident Analysis API

A Python/FastAPI backend that analyzes uploaded videos using a Vision Language Model (VLM) to detect and classify incidents. The API performs real-time video analysis instead of returning pre-defined mock data.

## Features

- **Real Video Analysis**: Uses Qwen3-VL-8B-Instruct VLM for actual video understanding
- **Automatic Frame Extraction**: Extracts frames at 1fps with intelligent naming convention
- **Deepfake Detection**: Analyzes video for AI-generated content indicators
- **Structured JSON Output**: Returns consistent, parseable JSON responses
- **Incident Classification**: Detects Security, Traffic, Fire, Fighting, and Unlawful Gathering incidents
- **Severity Assessment**: Rates incidents on a 0-3 scale
- **Location Detection**: Identifies Singapore locations when visible

## Architecture

```
Frontend → /upload → Save video → Extract frames → Return media_uuid
                            ↓
                      videos/{media_uuid}/
                          video.mp4
                          frames/{video_name}_01.jpg, _02.jpg, etc.
                            ↓
Frontend → /predict/{media_uuid} → Load frames → Send to VLM → Return JSON
```

## Installation

### Prerequisites

- Python 3.8 or higher
- ffmpeg (for video frame extraction)
- ffprobe (for video metadata)
- Access to VLM endpoint at `http://192.168.1.10:8000`

### Install ffmpeg/ffprobe

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html

### Python Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 4000
```

Or run directly:
```bash
python main.py
```

## API Endpoints

### 1. Upload Video

**POST** `/upload` or `/upload/`

Accepts video file upload via multipart/form-data, saves it permanently, extracts frames, and generates a media_uuid.

#### Example Request (Frontend)

```javascript
const formData = new FormData();
formData.append('post_file', file);

const response = await fetch("http://localhost:4000/upload/", {
  method: 'POST',
  body: formData,
});
```

#### Example Request (cURL)

```bash
curl -X POST http://localhost:4000/upload/ \
  -F "post_file=@/path/to/your/video.mp4"
```

#### Example Response

```json
{
  "status": "success",
  "message": "File received and processed",
  "filename": "fight1.mp4",
  "content_type": "video/mp4",
  "size": 5242880,
  "media_uuid": "d6eb3208-1c82-2ed5-72b7-0567826d9d9d",
  "deepfake": false
}
```

#### Video Storage

Videos are stored in `videos/{media_uuid}/`:
- Original video: `videos/{media_uuid}/video.mp4`
- Extracted frames: `videos/{media_uuid}/frames/{video_name}_{seconds}.jpg`

**Frame Naming Convention:**
- Videos ≤99 seconds: `{video_name}_01.jpg`, `{video_name}_02.jpg`, etc.
- Videos >99 seconds: `{video_name}_001.jpg`, `{video_name}_002.jpg`, etc.

Example:
- `fight1.mp4` (45s) → `fight1_01.jpg`, `fight1_02.jpg`, ..., `fight1_45.jpg`
- `traffic_long.mp4` (120s) → `traffic_long_001.jpg`, `traffic_long_002.jpg`, ..., `traffic_long_120.jpg`

### 2. Predict/Analyze Video

**POST** `/predict/{media_uuid}`

Loads the extracted frames, sends them to the VLM for analysis, and returns structured incident data.

#### Example Request (Frontend)

```javascript
const response = await fetch(`http://localhost:4000/predict/${mediaUuid}`, {
  method: 'POST'
});
```

#### Example Request (cURL)

```bash
curl -X POST http://localhost:4000/predict/d6eb3208-1c82-2ed5-72b7-0567826d9d9d
```

#### Example Response

```json
{
  "summary": "This Traffic incident involves a collision between a motorcycle and a passenger car at a signalized intersection. The incident occurred when a silver sedan (Vehicle 2) and a motorcycle (Vehicle 3) collided while navigating the junction...",
  "shortsummary": "Car and motorcycle collision; rider pinned under taxi at signalized intersection.",
  "deepfake": false,
  "media_uuid": "d6eb3208-1c82-2ed5-72b7-0567826d9d9d",
  "incidentType": "Traffic",
  "authenticity": 0.95,
  "severity": 3,
  "location": "Dover"
}
```

## Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `summary` | string | Detailed chronological analysis with timestamped events |
| `shortsummary` | string | Brief one-sentence description of the incident |
| `deepfake` | boolean | Whether the content is detected as AI-generated |
| `media_uuid` | string | The UUID provided in the request |
| `incidentType` | string | Type of incident (Security, Traffic, Fire, Fighting, Unlawful Gathering) |
| `authenticity` | float | Authenticity score (0.0-1.0, where 1.0 is definitely real) |
| `severity` | integer | Severity level (0-3, where 3 is most severe) |
| `location` | string | Location of the incident (Singapore area or Unknown) |

## Incident Types

| Type | Description |
|------|-------------|
| **Security** | Aggression, theft, suspicious activity, unauthorized access |
| **Traffic** | Collisions, reckless driving, pedestrian impacts, road obstructions |
| **Fire** | Visible flames, smoke, explosions, fire spread |
| **Fighting** | Physical altercations, assaults |
| **Unlawful Gathering** | Public demonstrations, unauthorized assemblies |

## Severity Scale

| Level | Description |
|-------|-------------|
| **0** | No incident / False alarm |
| **1** | Minor incident |
| **2** | Moderate incident |
| **3** | Severe / Critical incident |

## Console Output

### File Upload
```
==================================================
Processing video upload:
  Filename: fight1.mp4
  Content-Type: video/mp4
  File size: 5242880 bytes
  Generated media_uuid: d6eb3208-1c82-2ed5-72b7-0567826d9d9d
==================================================

Extracted 45 frames with pattern: fight1_%02d.jpg
```

### Predict Request
```
==================================================
Received predict request:
  media_uuid: d6eb3208-1c82-2ed5-72b7-0567826d9d9d
  Frames to analyze: 45
==================================================

==================================================
VLM Analysis Result:
  summary: Car and motorcycle collision; rider pinned under taxi at signalized intersection.
  incidentType: Traffic
  severity: 3
  deepfake: false
  authenticity: 0.95
==================================================
```

### VLM Service Unavailable
```
==================================================
VLM SERVICE UNAVAILABLE: Cannot connect to host 192.168.1.10:8000
==================================================
```

## Error Handling

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 404 | Video or frames not found |
| 500 | Server error or invalid VLM response |
| 503 | VLM service unavailable |

## Configuration

Edit the configuration in `main.py`:

```python
# VLM endpoint and model
VLM_ENDPOINT = "http://192.168.1.10:8000/v1/chat/completions"
VLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# Video storage directory
VIDEOS_DIR = Path("videos")
```

## Dependencies

```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
aiohttp>=3.9.0
pydantic>=2.5.0
```

## System Requirements

- Python 3.8+
- ffmpeg (for video processing)
- ffprobe (for video metadata)
- Network access to VLM endpoint

## Troubleshooting

### VLM Service Unavailable

If you see "VLM SERVICE UNAVAILABLE":
1. Check that the VLM endpoint URL is correct
2. Verify the VLM server is running at `http://192.168.1.10:8000`
3. Ensure network connectivity to the VLM server

### Frame Extraction Fails

If frames are not extracted:
1. Verify ffmpeg is installed: `ffmpeg -version`
2. Check video file format is supported
3. Ensure sufficient disk space in `videos/` directory

### Video Not Found

If `/predict/{media_uuid}` returns 404:
1. Verify the media_uuid matches what was returned from `/upload`
2. Check that `videos/{media_uuid}/` directory exists
3. Ensure frames were extracted during upload

## Limitations

- Video processing time depends on video length and VLM response time
- Large videos may cause memory issues
- Frame extraction at 1fps may miss rapid events
- VLM accuracy depends on video quality and clarity
- Location detection depends on visible landmarks

## Security Considerations

- CORS is enabled for all origins (configure for production)
- No authentication is implemented (add for production use)
- Videos are stored permanently (implement cleanup policy if needed)
- Media UUIDs are generated from MD5 hash (not cryptographically secure)

## License

This is a demonstration/development project for incident analysis.