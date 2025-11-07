import argparse
import os

from .env import FFmpegLogSuppressor, configure_environment, preimport_silence


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="pasta com Poop/ Not_Poop/ Coprophagy/ Not_Coprophagy/")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--val_split", type=float, default=0.20)
    ap.add_argument("--outdir", default="runs/ellie_multihead")
    ap.add_argument("--parallel_reads", type=int, default=0, help="0 = AUTOTUNE, ou defina N workers decodificando")
    ap.add_argument("--debug_io", action="store_true", help="Loga arquivos na leitura para depurar travas")
    ap.add_argument("--skip_finetune", action="store_true", help="Pula o fine-tune e segue para export.")
    ap.add_argument("--unfreeze", type=int, default=40, help="Qtde de layers do backbone para descongelar no fine-tune.")
    ap.add_argument("--lr_finetune", type=float, default=5e-5, help="LR do fine-tune.")
    ap.add_argument("--oversample_poop", type=int, default=1, help="Replicar positivos POOP (fator). 1 = desliga.")
    ap.add_argument("--oversample_copro", type=int, default=1, help="Replicar positivos COPRO (fator). 1 = desliga.")
    ap.add_argument("--augment", action="store_true", help="Aplica flip/brightness/contrast no TREINO.")
    ap.add_argument("--split_seed", type=int, default=42, help="Semente do split train/val.")
    return ap


def main() -> None:
    args = build_argparser().parse_args()

    configure_environment()


    with preimport_silence():
        from .train import TrainConfig, run  # deferred heavy imports under silence

    try:
        cfg = TrainConfig(
            data_dir=args.data_dir,
            frames=args.frames,
            size=args.size,
            batch=args.batch,
            epochs=args.epochs,
            val_split=args.val_split,
            outdir=args.outdir,
            parallel_reads=(args.parallel_reads if args.parallel_reads > 0 else None),
            debug_io=args.debug_io,
            oversample_poop=args.oversample_poop,
            oversample_copro=args.oversample_copro,
            augment=args.augment,
            split_seed=args.split_seed,
        )
        run(cfg)
    finally:
        FFmpegLogSuppressor.summary()


if __name__ == "__main__":
    main()


