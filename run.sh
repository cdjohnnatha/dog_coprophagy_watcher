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
export PROPER_ZONE="$(jq -r '.proper_zone // .toilet_zone // ""' $OPTS)"
export MONITOR_ALL_ZONES="$(jq -r '.monitor_all_zones // true' $OPTS)"

export RESIDUE_CONFIRM_WINDOW_S=$(jq -r '.residue_confirm_window_s' $OPTS)
export RESIDUE_MIN_AREA=$(jq -r '.residue_min_area' $OPTS)
export RESIDUE_STATIC_SEC=$(jq -r '.residue_static_sec' $OPTS)
export SNAPSHOT_FPS=$(jq -r '.snapshot_fps' $OPTS)
export CHECK_INTERVAL_S=$(jq -r '.check_interval_s' $OPTS)
export ENABLE_DEBUG_WATCHER="$(jq -r '.enable_debug_watcher' $OPTS || echo 'false')"
export EVIDENCE_ENABLED="$(jq -r '.evidence_enabled // true' $OPTS)"
export EVIDENCE_DIR="$(jq -r '.evidence_dir // "/data/evidence"' $OPTS)"

# Pad reference image (persistent) + capture heuristics
export PAD_REF_IMAGE_PATH="$(jq -r '.pad_ref_image_path // "/data/pad_clean_ref.jpg"' $OPTS)"
export PAD_REF_PERSON_WINDOW_S="$(jq -r '.pad_ref_person_window_s // 180' $OPTS)"
export PAD_REF_DOG_ABSENT_S="$(jq -r '.pad_ref_dog_absent_s // 5' $OPTS)"

export PEE_ENABLED="$(jq -r '.pee_enabled // true' $OPTS)"
export PEE_CONFIRM_WINDOW_S="$(jq -r '.pee_confirm_window_s // 25' $OPTS)"
export PEE_MIN_AREA="$(jq -r '.pee_min_area // 220' $OPTS)"
export PEE_STATIC_SEC="$(jq -r '.pee_static_sec // 1.0' $OPTS)"
export PEE_DELTA_L_MIN="$(jq -r '.pee_delta_l_min // 10' $OPTS)"
export PEE_DELTA_L_BRIGHT_MIN="$(jq -r '.pee_delta_l_bright_min // 12' $OPTS)"
export PEE_DELTA_B_MIN="$(jq -r '.pee_delta_b_min // 10' $OPTS)"
export PEE_MASS_MIN="$(jq -r '.pee_mass_min // 1200' $OPTS)"
export PEE_INCREMENT_ENABLED="$(jq -r '.pee_increment_enabled // true' $OPTS)"
export PEE_INCREMENT_MASS_MIN="$(jq -r '.pee_increment_mass_min // 500' $OPTS)"
export LAST_IMAGE_MAX_AGE_S="$(jq -r '.last_image_max_age_s // 240' $OPTS)"
export PEE_COOLDOWN_S="$(jq -r '.pee_cooldown_s // 45' $OPTS)"
export PEE_MERGE_RADIUS_PX="$(jq -r '.pee_merge_radius_px // 70' $OPTS)"

export EP_COOLDOWN_S="$(jq -r '.ep_cooldown_s // 180' $OPTS)"
export MERGE_RADIUS_PX="$(jq -r '.merge_radius_px // 90' $OPTS)"

echo "[Ellie Watcher] Iniciando com:"
echo "MQTT: ${MQTT_HOST}:${MQTT_PORT} user=${MQTT_USER}"
echo "Frigate: ${FRIGATE_BASE_URL} camera=${CAMERA_NAME} zone=${TOILET_ZONE} proper_zone=${PROPER_ZONE} all_zones=${MONITOR_ALL_ZONES}"
echo "Debug: ${ENABLE_DEBUG_WATCHER}"
echo "Evidence: enabled=${EVIDENCE_ENABLED} dir=${EVIDENCE_DIR}"
echo "PadRef: path=${PAD_REF_IMAGE_PATH} person_window=${PAD_REF_PERSON_WINDOW_S}s dog_absent=${PAD_REF_DOG_ABSENT_S}s"
echo "Pee: enabled=${PEE_ENABLED} win=${PEE_CONFIRM_WINDOW_S}s min_area=${PEE_MIN_AREA} static=${PEE_STATIC_SEC}s deltaL>=${PEE_DELTA_L_MIN} deltaL+>=${PEE_DELTA_L_BRIGHT_MIN} deltaB>=${PEE_DELTA_B_MIN} mass>=${PEE_MASS_MIN}"
echo "PeeIncrement: enabled=${PEE_INCREMENT_ENABLED} mass_min=${PEE_INCREMENT_MASS_MIN} last_image_max_age_s=${LAST_IMAGE_MAX_AGE_S}"

exec python -u /app/main.py