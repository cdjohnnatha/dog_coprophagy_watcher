#!/usr/bin/env bash
# ============================================================
# Ellie ML - Video Normalizer
# Uso:
#   ./normalize_videos.sh /path/para/datasets/ellie_multihead
#
# Vai procurar pelas subpastas:
#   Poop/  Not_Poop/  Coprophagy/  Not_Coprophagy/
# e salvar tudo em:
#   normalized/{mesmas pastas}
# ============================================================

set -euo pipefail
IFS=$'\n\t'

# --- Argumentos ---
if [ $# -lt 1 ]; then
  echo "❌ Uso: $0 <caminho_base>"
  exit 1
fi

BASE_DIR="$(realpath "$1")"
OUT_DIR="$BASE_DIR/normalized"

# --- Dependências ---
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "❌ ffmpeg não encontrado no PATH. Instale-o e tente novamente."
  exit 1
fi

if ! command -v ffprobe >/dev/null 2>&1; then
  echo "❌ ffprobe não encontrado no PATH. Instale-o (geralmente vem com o ffmpeg)."
  exit 1
fi

# --- Parâmetros (configuráveis via env) ---
FPS=${FPS:-10}            # Frames por segundo (CFR)
SIZE=${SIZE:-224}         # Lado da imagem quadrada de saída
CRF=${CRF:-18}            # Qualidade x264 (menor = melhor/maior arquivo). 18 ≈ visually lossless
PRESET=${PRESET:-veryfast}
TUNE=${TUNE:-fastdecode}
THREADS=${THREADS:-0}     # 0 = auto
JOBS=${JOBS:-$( (sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN || echo 4) )}

# 1 segundo de GOP por padrão (estável para decodificação e amostragem determinística)
GOP=$((FPS))

trap 'echo; echo "⛔ Interrompido"; exit 130' INT

# --- Estado global / sumário ---
mkdir -p "$OUT_DIR"
FAILED_LOG="$OUT_DIR/_failed.txt"
: > "$FAILED_LOG"   # limpa apenas uma vez
TOTAL_SRC_ALL=0
TOTAL_OK_ALL=0
TOTAL_FAIL_ALL=0
START_TS=$(date +%s)

# --- Utilitários ---
hash_input() {
  local f="$1"
  if command -v md5 >/dev/null 2>&1; then
    md5 -q "$f" | head -c 8
  elif command -v md5sum >/dev/null 2>&1; then
    md5sum "$f" | awk '{print $1}' | head -c 8
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 1 "$f" | awk '{print $1}' | head -c 8
  else
    echo "nohash"
  fi
}

wait_for_slot() {
  # Limita concorrência usando contagem de jobs ativos
  while [ "$(jobs -pr | wc -l | tr -d ' ')" -ge "$JOBS" ]; do
    sleep 0.2
  done
}

# --- Confirma diretório base ---
if [ ! -d "$BASE_DIR" ]; then
  echo "❌ Diretório não encontrado: $BASE_DIR"
  exit 1
fi

# --- Subpastas esperadas ---
CATEGORIES=(Poop Not_Poop Coprophagy Not_Coprophagy)

echo "📁 Normalizando vídeos em: $BASE_DIR"
echo "📦 Saída: $OUT_DIR"
echo "⚙️  Config: FPS=$FPS, SIZE=${SIZE}x${SIZE}, CRF=$CRF, PRESET=$PRESET, THREADS=$THREADS, JOBS=$JOBS"
echo

# --- Loop principal ---
for d in "${CATEGORIES[@]}"; do
  SRC="$BASE_DIR/$d"
  DST="$OUT_DIR/$d"

  if [ ! -d "$SRC" ]; then
    echo "⚠️  Pasta ausente: $SRC (ignorando)"
    continue
  fi

  mkdir -p "$DST"
  echo "🎞️  Processando $d ..."

  while IFS= read -r -d '' f; do
    input_file="${f//$'\r'/}"

    if [ ! -f "$input_file" ]; then
      echo "⚠️  Arquivo não encontrado (ignorando): $input_file"
      continue
    fi

    base="$(basename "$input_file")"
    stem="${base%.*}"
    h="$(hash_input "$input_file")"
    out="$DST/${stem}__${h}.mp4"  # nome único e estável

    if [ -f "$out" ]; then
      echo "⏩ Já existe: $out (pulando)"
      continue
    fi

    wait_for_slot
    (
      if ! ffmpeg -nostdin -hide_banner -loglevel error -fflags +genpts -y -i "$input_file" \
        -an -c:v libx264 -preset "$PRESET" -tune "$TUNE" -crf "$CRF" -threads "$THREADS" \
        -g "$GOP" -keyint_min "$GOP" -sc_threshold 0 -r "$FPS" -vsync cfr \
        -movflags +faststart -map_metadata -1 \
        -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
        -vf "scale=${SIZE}:${SIZE}:force_original_aspect_ratio=decrease:eval=frame,pad=${SIZE}:${SIZE}:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p,fps=${FPS}" \
        "$out"; then
        echo "❌ Falha ao normalizar: $input_file" >&2
        printf '%s\n' "$input_file" >> "$FAILED_LOG"
        rm -f "$out"
        exit 1
      fi
      echo "✅ $d → $(basename "$out")"
    ) &
  done < <(find "$SRC" -type f \( -iname "*.mp4" -o -iname "*.mov" -o -iname "*.mkv" -o -iname "*.avi" \) -print0)

  # Espera o restante deste diretório concluir
  wait
  # Relatório breve
  total_src=$(find "$SRC" -type f \( -iname "*.mp4" -o -iname "*.mov" -o -iname "*.mkv" -o -iname "*.avi" \) | wc -l | tr -d ' ')
  total_ok=$(find "$DST" -type f -iname "*.mp4" | wc -l | tr -d ' ')
  total_fail=$(wc -l < "$FAILED_LOG" 2>/dev/null | tr -d ' ')
  echo "📊 $d: origem=$total_src, ok=$total_ok, falhas=${total_fail:-0}"

  # Acumula totais gerais (falhas: usa incremento incremental por categoria)
  TOTAL_SRC_ALL=$((TOTAL_SRC_ALL + total_src))
  TOTAL_OK_ALL=$((TOTAL_OK_ALL + total_ok))
  # Não somamos total_fail diretamente (é cumulativo no arquivo). Recalcular por categoria é caro; mantemos apenas total global ao fim.
done

# --- Sumário final ---
END_TS=$(date +%s)
DUR=$((END_TS - START_TS))
FAILED_COUNT=$(wc -l < "$FAILED_LOG" 2>/dev/null | tr -d ' ')
OUT_SIZE=$(du -sh "$OUT_DIR" 2>/dev/null | awk '{print $1}')

echo
echo "🏁 Normalização concluída!"
echo "📂 Saída final: $OUT_DIR (${OUT_SIZE:-?})"
echo "⏱️  Tempo total: ${DUR}s"
echo "📊 Totais: origem=$TOTAL_SRC_ALL, ok=$TOTAL_OK_ALL, falhas=${FAILED_COUNT:-0}"
if [ "${FAILED_COUNT:-0}" -gt 0 ]; then
  echo "❗ Arquivos com falha listados em: $FAILED_LOG"
fi

echo
echo "🏁 Normalização concluída!"
echo "📂 Saída final: $OUT_DIR"