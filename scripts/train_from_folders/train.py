import os
import numpy as np
from dataclasses import dataclass
from typing import Optional

import tensorflow as tf

from .data_index import index_dataset, train_val_split
from .dataset import make_dataset
from .model import build_model


@dataclass
class TrainConfig:
    data_dir: str
    frames: int = 16
    size: int = 224
    batch: int = 4
    epochs: int = 12
    val_split: float = 0.20
    outdir: str = "runs/ellie_multihead"
    parallel_reads: Optional[int] = None
    debug_io: bool = False

    # ====== novas opções ======
    skip_finetune: bool = False           # pular fine-tune
    unfreeze: int = 40                    # quantas camadas do backbone descongelar
    lr_finetune: float = 5e-5             # learning rate do fine-tune
    ft_parallel_reads: int = 1            # leitura mais conservadora no fine-tune
    deterministic_ft: bool = True         # dataset determinístico no fine-tune

    # oversampling
    oversample_poop: int = 1              # replicar positivos POOP (1 = desliga)
    oversample_copro: int = 1             # replicar positivos COPRO (1 = desliga)
    augment: bool = False                 # aplicar aug no treino
    split_seed: int = 42                  # semente do split


class Heartbeat(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        if batch % 10 == 0:
            print(f"… batch {batch} começou")

    def on_train_batch_end(self, batch, logs=None):
        if batch % 10 == 0:
            print(f"✓ batch {batch} terminou – loss={logs.get('loss'):.4f}")


def create_callbacks(outdir: str):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(outdir, "ckpt_best.keras"),
            monitor="val_loss",
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(outdir, "tb")),
        Heartbeat(),
    ]


def _export_artifacts(model: tf.keras.Model, outdir: str) -> None:
    """Save SavedModel and attempt TFLite export into outdir."""
    save_dir = os.path.join(outdir, "saved_model")
    os.makedirs(outdir, exist_ok=True)
    # SavedModel
    model.save(save_dir)
    print(f"✅ Modelo salvo em: {save_dir}")
    # TFLite (best-effort)
    try:
        print("Convertendo para TFLite (com SELECT_TF_OPS)…")
        converter = tf.lite.TFLiteConverter.from_saved_model(save_dir)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tfl = converter.convert()
        tfl_path = os.path.join(outdir, "ellie_multihead.tflite")
        with open(tfl_path, "wb") as f:
            f.write(tfl)
        print(f"✅ TFLite salvo em: {tfl_path}")
    except Exception as e:
        print(f"⚠️ Falha ao converter TFLite (pode ignorar por enquanto): {e}")


def enable_partial_finetune(
    model: tf.keras.Model,
    n_unfreeze: int = 40,
    lr: float = 5e-5,
    use_metrics: bool = False,
) -> tf.keras.Model:
    """
    Descongela as últimas n_unfreeze camadas do backbone (exceto BatchNorm),
    recompila com LR específico e reutiliza as mesmas losses/metrics do modelo.
    """
    # Localiza de forma robusta o TimeDistributed que envolve a MobileNetV2
    base = None
    for l in model.layers:
        if isinstance(l, tf.keras.layers.TimeDistributed):
            sub = getattr(l, "layer", None)
            name = getattr(sub, "name", "")
            if isinstance(sub, tf.keras.Model) and "mobilenetv2" in name.lower():
                base = sub
                break
    if base is None:
        # fallback heurístico: pega o TimeDistributed cujo sub-model tem muitos params
        for l in model.layers:
            if isinstance(l, tf.keras.layers.TimeDistributed):
                sub = getattr(l, "layer", None)
                if hasattr(sub, "layers"):
                    try:
                        if sub.count_params() > 1_000_000:
                            base = sub
                            break
                    except Exception:
                        pass
    if base is None:
        raise RuntimeError("Backbone MobileNetV2 não encontrado para fine-tune.")

    # descongela últimas n_unfreeze (mantém BatchNorm congeladas)
    for l in base.layers[-n_unfreeze:]:
        if not isinstance(l, tf.keras.layers.BatchNormalization):
            l.trainable = True
    # recompila
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=model.loss,
        metrics=model.metrics if use_metrics else None
    )
    return model


def _maybe_make_deterministic(ds: tf.data.Dataset, deterministic: bool) -> tf.data.Dataset:
    if not deterministic:
        return ds
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    return ds.with_options(opts)


def run(cfg: TrainConfig) -> None:
    os.makedirs(cfg.outdir, exist_ok=True)
    # direciona caches para a pasta de saída desta execução
    os.environ["ELLIE_OUTDIR"] = cfg.outdir

    print("Indexando vídeos...")
    items = index_dataset(cfg.data_dir)
    if not items:
        raise RuntimeError("Nenhum vídeo encontrado.")
    train_items, val_items = train_val_split(items, val_ratio=cfg.val_split, seed=cfg.split_seed)
    print(f"Total: {len(items)} | train: {len(train_items)} | val: {len(val_items)}")

    # ====== Oversampling (opcional) – duplica itens positivos por cabeça ======
    def _oversample(items_in: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
        if cfg.oversample_poop <= 1 and cfg.oversample_copro <= 1:
            return items_in
        out: list[tuple[str, float, float]] = []
        for path, y_po, y_co in items_in:
            out.append((path, y_po, y_co))
            if not np.isnan(y_po) and int(y_po) == 1 and cfg.oversample_poop > 1:
                out.extend([(path, y_po, y_co)] * (cfg.oversample_poop - 1))
            if not np.isnan(y_co) and int(y_co) == 1 and cfg.oversample_copro > 1:
                out.extend([(path, y_po, y_co)] * (cfg.oversample_copro - 1))
        rng = np.random.default_rng(42)
        rng.shuffle(out)
        return out

    train_items_os = _oversample(train_items)

    # ====== Datasets para o treino com backbone congelado ======
    pr = cfg.parallel_reads if (cfg.parallel_reads and cfg.parallel_reads > 0) else None
    train_ds = make_dataset(
        train_items_os, cfg.frames, cfg.size, cfg.batch,
        shuffle=True, parallel_reads=pr, debug_io=cfg.debug_io, augment=cfg.augment
    )
    val_ds = make_dataset(
        val_items, cfg.frames, cfg.size, cfg.batch,
        shuffle=False, parallel_reads=pr, debug_io=False, augment=False
    )

    # ====== Estatísticas de classe no TREINO (para vieses e pesos da loss) ======
    def _head_stats(items):
        # Poop
        y_po = [int(v) for _, v, _ in items if not np.isnan(v)]
        pos_po = int(np.sum(np.array(y_po) == 1))
        neg_po = int(np.sum(np.array(y_po) == 0))
        # Copro
        y_co = [int(v) for _, _, v in items if not np.isnan(v)]
        pos_co = int(np.sum(np.array(y_co) == 1))
        neg_co = int(np.sum(np.array(y_co) == 0))
        return (pos_po, neg_po, pos_co, neg_co)

    pos_po, neg_po, pos_co, neg_co = _head_stats(train_items)
    # taxas positivas para viés inicial
    pos_rate_poop = (pos_po / max(1, pos_po + neg_po)) if (pos_po + neg_po) > 0 else 0.15
    pos_rate_copr = (pos_co / max(1, pos_co + neg_co)) if (pos_co + neg_co) > 0 else 0.15
    # pesos positivos ~ neg/pos (evita divisão por zero)
    posw_poop = float(neg_po / max(1, pos_po)) if pos_po > 0 else 1.0
    posw_copr = float(neg_co / max(1, pos_co)) if pos_co > 0 else 1.0

    print(
        f"Stats TREINO — POOP pos={pos_po} neg={neg_po} | COPRO pos={pos_co} neg={neg_co}\n"
        f"pos_rate: poop={pos_rate_poop:.3f} copro={pos_rate_copr:.3f} | posw: poop={posw_poop:.2f} copro={posw_copr:.2f}"
    )

    # build_model com vieses/weights calibrados pelo split de treino
    model = build_model(
        cfg.frames,
        cfg.size,
        use_metrics=False,
        pos_rate_poop=pos_rate_poop,
        pos_rate_copr=pos_rate_copr,
        posw_poop=posw_poop,
        posw_copr=posw_copr,
    )
    model.summary()

    cbs = create_callbacks(cfg.outdir)

    print("Treinando (backbone congelado)...")
    model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs, callbacks=cbs)
    print("Treinando (backbone terminado)", cfg.outdir)

    # Exporta após a primeira fase (útil se você quiser avaliar/usar já)
    print("Exportando artefatos após fase congelada…")
    _export_artifacts(model, cfg.outdir)

    # ====== Fine-tune seguro/determinístico ======
    if not cfg.skip_finetune:
        print("Fine-tuning (parcial)...")
        # recria datasets mais conservadores para o fine-tune
        ft_train = make_dataset(
            train_items, cfg.frames, cfg.size, cfg.batch,
            shuffle=True, parallel_reads=cfg.ft_parallel_reads, debug_io=False, augment=cfg.augment
        )
        ft_val = make_dataset(
            val_items, cfg.frames, cfg.size, cfg.batch,
            shuffle=False, parallel_reads=cfg.ft_parallel_reads, debug_io=False, augment=False
        )
        ft_train = _maybe_make_deterministic(ft_train, cfg.deterministic_ft)
        ft_val = _maybe_make_deterministic(ft_val, cfg.deterministic_ft)

        model = enable_partial_finetune(
            model, n_unfreeze=cfg.unfreeze, lr=cfg.lr_finetune, use_metrics=False
        )
        try:
            model.fit(
                ft_train, validation_data=ft_val,
                epochs=max(4, cfg.epochs // 3),
                callbacks=cbs
            )
            print("Fine-tuning (parcial) terminado", cfg.outdir)
        except KeyboardInterrupt:
            print("\n⛔ Fine-tune interrompido — seguindo para exportação.")
    else:
        print("⏭️  Fine-tune pulado por configuração (skip_finetune=True).")

    # ====== Exporta novamente ao final (fine-tunado) ======
    print("Exportando artefatos finais…")
    _export_artifacts(model, cfg.outdir)

    # ====== Sucesso ======
    print("✅ Training pipeline finished successfully!")
    print(
        f"📂 Saída: {cfg.outdir}\n"
        f"   ├─ ckpt_best.keras\n"
        f"   ├─ saved_model/\n"
        f"   ├─ ellie_multihead.tflite  (se a conversão passou)\n"
        f"   └─ tb/  (logs do TensorBoard)"
    )