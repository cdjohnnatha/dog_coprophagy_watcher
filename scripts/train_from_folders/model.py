# model.py
import tensorflow as tf
import numpy as np

# ======= Loss: BCE com máscara (-1) e peso na classe positiva =======
def masked_bce_pos_weight(pos_weight: float = 1.0):
    """
    BCE ponderada para classe positiva com máscara (-1 => ignora).
    pos_weight ~ (negativos / positivos) no split de TREINO.
    """
    pos_weight = float(pos_weight)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)              # [B,1] em {0,1} e -1 p/ ignorar
        mask   = tf.not_equal(y_true, -1.0)               # True = usa no cálculo
        y_true_safe = tf.where(mask, y_true, 0.0)         # substitui -1 por 0 só p/ BCE

        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

        # BCE ponderada na classe positiva
        per_ex = -( pos_weight * y_true_safe * tf.math.log(y_pred)
                    + (1.0 - y_true_safe) * tf.math.log(1.0 - y_pred) )

        # aplica máscara
        per_ex = tf.where(mask, per_ex, 0.0)

        denom = tf.reduce_sum(tf.cast(mask, tf.float32)) + 1e-6
        return tf.reduce_sum(per_ex) / denom

    return loss


# ======= Métrica AUC com máscara (-1) opcional =======
class MaskedAUC(tf.keras.metrics.AUC):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        mask = tf.not_equal(y_true, -1.0)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        if sample_weight is not None:
            sample_weight = tf.boolean_mask(sample_weight, mask)
        return super().update_state(y_true, y_pred, sample_weight=sample_weight)


def build_model(
    num_frames: int = 16,
    size: int = 224,
    use_metrics: bool = False,
    *,
    # taxa positiva estimada (para viés inicial das saídas)
    pos_rate_poop: float = 0.15,
    pos_rate_copr: float = 0.15,
    # pesos da classe positiva na loss (neg/pos do TREINO)
    posw_poop: float = 2.0,
    posw_copr: float = 2.0,
):
    """Modelo multihead (poop/copro) com MobileNetV2 frame-a-frame + cabeça temporal Conv1D.
       - rótulos -1 são ignorados na perda.
       - viés inicial nas Dense conforme taxa positiva estimada.
       - métricas opcionais desativadas por padrão.
    """

    inp = tf.keras.Input(shape=(num_frames, size, size, 3), name="video")

    # Normalização para MobileNetV2: [0,1] -> [-1,1]
    norm = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)
    )(inp)

    # Backbone 2D por frame
    base = tf.keras.applications.MobileNetV2(
        input_shape=(size, size, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False

    # [B, T, 1280]
    x = tf.keras.layers.TimeDistributed(base)(norm)

    # Cabeça temporal sem RNN (TFLite-friendly)
    x = tf.keras.layers.Conv1D(192, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(192, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)   # [B, 192]
    x = tf.keras.layers.Dropout(0.3)(x)

    # Viés inicial nas saídas (calibra probabilidade inicial)
    bias_poop = float(np.log(pos_rate_poop / (1.0 - pos_rate_poop)))
    bias_copr = float(np.log(pos_rate_copr / (1.0 - pos_rate_copr)))

    poop  = tf.keras.layers.Dense(
        1, activation="sigmoid",
        bias_initializer=tf.keras.initializers.Constant(bias_poop),
        name="poop"
    )(x)

    copro = tf.keras.layers.Dense(
        1, activation="sigmoid",
        bias_initializer=tf.keras.initializers.Constant(bias_copr),
        name="copro"
    )(x)

    model = tf.keras.Model(inp, {"poop": poop, "copro": copro})

    # Métricas opcionais
    metrics = None
    if use_metrics:
        metrics = {"poop": [MaskedAUC(name="auc")], "copro": [MaskedAUC(name="auc")]}

    # Compile com BCE ponderada
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss={
            "poop":  masked_bce_pos_weight(posw_poop),
            "copro": masked_bce_pos_weight(posw_copr),
        },
        metrics=metrics,
        weighted_metrics=[],
    )
    return model


def enable_partial_finetune(
    model: tf.keras.Model,
    n_unfreeze: int = 40,
    lr: float = 5e-5,
    *,
    use_metrics: bool = False,
    posw_poop: float = 2.0,
    posw_copr: float = 3.0,
):
    """Descongela parcialmente a MobileNetV2 (exceto BatchNorm) e recompila com os mesmos pesos de classe."""

    # TimeDistributed(base) é a layer de índice 1; pegue o .layer interno (o backbone)
    td = model.get_layer(index=1)
    base = getattr(td, "layer", None)
    if base is None:
        raise RuntimeError("Backbone não encontrado dentro do TimeDistributed.")

    # descongela últimas n_unfreeze (mantém BatchNorm congeladas)
    for l in base.layers[-n_unfreeze:]:
        if not isinstance(l, tf.keras.layers.BatchNormalization):
            l.trainable = True

    # métricas opcionais
    metrics = None
    if use_metrics:
        metrics = {"poop": [MaskedAUC(name="auc")], "copro": [MaskedAUC(name="auc")]}

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss={
            "poop":  masked_bce_pos_weight(posw_poop),
            "copro": masked_bce_pos_weight(posw_copr),
        },
        metrics=metrics,
        weighted_metrics=[],
    )
    return model