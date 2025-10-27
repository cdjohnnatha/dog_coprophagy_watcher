#!/usr/bin/env bash
set -e

# Supervisor injeta as opções em /data/options.json
OPTS=/data/options.json

# Extrai opções e exporta como variáveis de ambiente para o app.py
export MQTT_HOST=$(jq -r '.mqtt_host' $OPTS)
export MQTT_PORT=$(jq -r '.mqtt_port' $OPTS)
export MQTT_USER=$(jq -r '.mqtt_user' $OPTS)
export MQTT_PASS=$(jq -r '.mqtt_pass' $OPTS)
export MQTT_PREFIX=$(jq -r '.mqtt_prefix' $OPTS)

export FRIGATE_BASE_URL=$(jq -r '.frigate_base_url' $OPTS)
export CAMERA_NAME=$(jq -r '.camera_name' $OPTS)
export TOILET_ZONE=$(jq -r '.toilet_zone' $OPTS)

export SQUAT_SCORE_THRESH=$(jq -r '.squat_score_thresh' $OPTS)
export SQUAT_MIN_DURATION_S=$(jq -r '.squat_min_duration_s' $OPTS)
export RESIDUE_CONFIRM_WINDOW_S=$(jq -r '.residue_confirm_window_s' $OPTS)
export RESIDUE_MIN_AREA=$(jq -r '.residue_min_area' $OPTS)
export RESIDUE_STATIC_SEC=$(jq -r '.residue_static_sec' $OPTS)
export SNAPSHOT_FPS=$(jq -r '.snapshot_fps' $OPTS)
export CHECK_INTERVAL_S=$(jq -r '.check_interval_s' $OPTS)

echo "[Ellie Watcher] Iniciando com:"
echo "MQTT: ${MQTT_HOST}:${MQTT_PORT} user=${MQTT_USER}"
echo "Frigate: ${FRIGATE_BASE_URL} camera=${CAMERA_NAME} zone=${TOILET_ZONE}"
echo "Heurística: squat=${SQUAT_SCORE_THRESH} min_dur=${SQUAT_MIN_DURATION_S}"

exec python -u /app/app.py