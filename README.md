# Dog Coprophagy Watcher (Ellie Watcher)

A smart monitoring system that detects when your dog is defecating and tracks poop cleanup using computer vision and MQTT integration with Frigate.

## What It Does

This application monitors your dog through Frigate's video surveillance system and:

1. **Detects Squatting Behavior**: Uses computer vision heuristics to identify when your dog is in a defecation posture
2. **Confirms Defecation**: Monitors for the appearance of fecal matter after squatting
3. **Tracks Cleanup**: Continuously monitors until the poop is cleaned up
4. **MQTT Notifications**: Publishes real-time status updates via MQTT for integration with home automation systems

## How It Works

### Squat Detection
The system analyzes bounding boxes from Frigate's object detection using multiple heuristics:
- **Aspect Ratio**: Dogs appear shorter and wider when squatting
- **Motion Analysis**: Stationary behavior indicates potential defecation
- **Lower Body Analysis**: Edge density in the lower body region confirms posture

### Residue Detection
After detecting squatting, the system:
- Captures a background image of the area below the dog
- Monitors for new "blobs" (potential poop) using image differencing
- Confirms persistent objects to avoid false positives
- Tracks cleanup by monitoring when the detected blob disappears

### State Machine
```
IDLE → POSSIVEL_DEFECACAO → DEFECANDO → AGUARDANDO_CONFIRMACAO → DEFECACAO_CONFIRMADA → IDLE
```

## Prerequisites

- **Frigate**: Video surveillance system with object detection
- **MQTT Broker**: For receiving events from Frigate and publishing status updates
- **Python 3.11+** or Docker

## Quick Start with Docker

1. **Clone and configure**:
   ```bash
   git clone <repository>
   cd dog_coprophagy_watcher
   ```

2. **Create environment file**:
   ```bash
   cp .env.sample .env
   # Edit .env with your configuration
   ```

3. **Configure your environment** (`.env`):
   ```bash
   # MQTT Configuration
   MQTT_HOST=your-mqtt-host
   MQTT_PORT=1883
   MQTT_USER=your-username  # optional
   MQTT_PASS=your-password  # optional

   # Frigate Configuration
   FRIGATE_BASE_URL=http://frigate:5000
   CAMERA_NAME=ellie
   TOILET_ZONE=toilet_zone

   # Detection Parameters (tune these for your setup)
   SQUAT_SCORE_THRESH=0.62
   SQUAT_MIN_DURATION_S=2.0
   RESIDUE_CONFIRM_WINDOW_S=20
   RESIDUE_MIN_AREA=140
   RESIDUE_STATIC_SEC=2.0
   SNAPSHOT_FPS=4
   CHECK_INTERVAL_S=15
   ```

4. **Run with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

## Configuration Parameters

### Core Settings
- `FRIGATE_BASE_URL`: URL of your Frigate instance
- `CAMERA_NAME`: Name of the camera configured in Frigate
- `TOILET_ZONE`: Zone name in Frigate where detection should occur

### MQTT Settings
- `MQTT_HOST`: MQTT broker hostname
- `MQTT_PORT`: MQTT broker port (default: 1883)
- `MQTT_USER`/`MQTT_PASS`: Credentials for MQTT broker (optional)
- `MQTT_PREFIX`: Frigate MQTT topic prefix (default: "frigate")

### Detection Tuning
- `SQUAT_SCORE_THRESH`: Minimum confidence score for squat detection (0.0-1.0)
- `SQUAT_MIN_DURATION_S`: Minimum time dog must be squatting to confirm defecation
- `RESIDUE_CONFIRM_WINDOW_S`: Time window to look for poop after squatting ends
- `RESIDUE_MIN_AREA`: Minimum pixel area for detected blobs
- `RESIDUE_STATIC_SEC`: How long a blob must persist to be considered real poop
- `SNAPSHOT_FPS`: Frame rate for analysis when dog is detected
- `CHECK_INTERVAL_S`: How often to check for poop cleanup

## MQTT Topics

### Published Topics
- `home/ellie/state`: Current detection state
  - `"IDLE"`: No activity
  - `"POSSIVEL_DEFECACAO"`: Possible defecation detected
  - `"DEFECANDO"`: Active defecation
  - `"AGUARDANDO_CONFIRMACAO"`: Waiting for residue confirmation
  - `"DEFECACAO_CONFIRMADA"`: Defecation confirmed

- `home/ellie/poop_present`: Poop presence status
  ```json
  {
    "value": true|false,
    "zone": "toilet_zone",
    "centroid": [x, y],
    "area": 150,
    "ts": "2024-01-01T12:00:00Z"
  }
  ```

### Subscribed Topics
- `frigate/events`: Frigate detection events (filtered for dog in specified zone)

## Manual Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables**:
   ```bash
   export MQTT_HOST=your-mqtt-host
   export FRIGATE_BASE_URL=http://frigate:5000
   # ... other variables
   ```

3. **Run**:
   ```bash
   python app.py
   ```

## Troubleshooting

### Common Issues

1. **No detections**: Check that Frigate is running and the camera/zone names match
2. **False positives**: Adjust `SQUAT_SCORE_THRESH` higher
3. **Missed detections**: Lower `SQUAT_SCORE_THRESH` or adjust other parameters
4. **Residue not detected**: Tune `RESIDUE_MIN_AREA` and lighting conditions

### Debug Tips

- Monitor MQTT topics to see detection states
- Check Frigate logs for object detection events
- Adjust camera angle/lighting for better detection
- Use Frigate's snapshot feature to verify camera view

## Architecture

The system consists of:
- **MQTT Event Handler**: Processes Frigate detection events
- **Squat Detection Engine**: Computer vision analysis for posture recognition
- **Residue Monitor**: Background subtraction for poop detection
- **State Machine**: Manages detection workflow and notifications

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper documentation
4. Test thoroughly with your setup
5. Submit a pull request

## License

This project is open source. Please check the license file for details.
