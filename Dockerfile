ARG BUILD_FROM
FROM ${BUILD_FROM}

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# OpenCV e NumPy via APK (muito mais confiável no Alpine)
RUN apk add --no-cache \
      bash \
      python3 \
      py3-pip \
      py3-numpy \
      py3-opencv \
      ffmpeg \
      curl \
      jq \
      ca-certificates \
      libstdc++ \
      libgcc

# aliases convenientes
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# dependências puramente Python
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# (opcional) verificação em build: garante que cv2/numpy estão disponíveis
RUN python - <<'PY'
import cv2, numpy
print("cv2:", cv2.__version__, "numpy:", numpy.__version__)
PY

# app
COPY app.py /app/app.py
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

CMD ["/app/run.sh"]