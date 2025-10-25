import subprocess, sys

steps = [
    ("01_build_manifest.py", "Gerando manifest..."),
    ("02_split_dataset.py", "Fazendo split train/val/test..."),
    ("03_flow_baseline.py", "Calculando heurística baseline..."),
]

for script, desc in steps:
    print(f"\n=== {desc} ({script}) ===")
    try:
        subprocess.run([sys.executable, script], check=True)
    except subprocess.CalledProcessError:
        print(f"❌ Erro ao executar {script}")
        sys.exit(1)

print("\n✅ Pipeline completo! Agora copie os thresholds sugeridos para o seu arquivo .env.")