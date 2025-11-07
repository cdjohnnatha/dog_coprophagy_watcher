import cv2, numpy as np, tensorflow as tf
from scripts.train_from_folders.train import load_video_clip  # usa a mesma função
PATH = "datasets/ellie_multihead/saved_model"
VIDEO = "assets/normalized/Poop/SEU_VIDEO.mp4"  # troque aqui
FRAMES, SIZE = 16, 224

model = tf.keras.models.load_model(PATH, compile=False)
x = load_video_clip(VIDEO, num_frames=FRAMES, target=SIZE)  # [T,H,W,3] float32 0..1
x = np.expand_dims(x, 0)  # [1,T,H,W,3]

pred = model.predict(x)
print("poop:", float(pred["poop"][0]), "copro:", float(pred["copro"][0]))