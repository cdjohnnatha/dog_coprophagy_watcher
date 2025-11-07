import numpy as np, tensorflow as tf
from scripts.train_from_folders.train import load_video_clip

MODEL_PATH = "datasets/ellie_multihead/ellie_multihead.tflite"
VIDEO = "assets/normalized/Poop/SEU_VIDEO.mp4"
FRAMES, SIZE = 16, 224

# Se estiver no Pi e instalou "tensorflow" (não apenas tflite-runtime), ok usar o Interpreter padrão
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()

x = load_video_clip(VIDEO, num_frames=FRAMES, target=SIZE)
x = np.expand_dims(x, 0).astype(np.float32)
interpreter.set_tensor(inp["index"], x)
interpreter.invoke()

# nomes das saídas dependem da conversão: encontremos por shape
def fetch(name_substr):
    for d in out:
        if name_substr in d["name"]:
            return interpreter.get_tensor(d["index"])
    return None

poop = fetch("poop")  # tente por nome
copro = fetch("copro")
if poop is None or copro is None:
    # fallback: imprime todos para descobrir
    vals = {d["name"]: interpreter.get_tensor(d["index"]) for d in out}
    print("saídas:", {k:v.shape for k,v in vals.items()})
else:
    print("poop:", float(poop[0][0]), "copro:", float(copro[0][0]))