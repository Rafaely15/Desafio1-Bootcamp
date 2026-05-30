import importlib.util
import platform
import subprocess
import sys


def has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    print("=== Verificacao do ambiente ===")
    print(f"Python: {sys.version}")
    print(f"Sistema: {platform.platform()}")

    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, text=True, check=True)
        print(f"pip: OK - {result.stdout.strip()}")
    except Exception as exc:
        print(f"pip: ERRO - {exc}")

    for package, import_name in [("ultralytics", "ultralytics"), ("opencv-python", "cv2")]:
        print(f"{package}: {'OK' if has_package(import_name) else 'NAO INSTALADO'}")

    if has_package("torch"):
        import torch

        cuda = torch.cuda.is_available()
        print(f"CUDA disponivel: {'SIM' if cuda else 'NAO'}")
        print(f"Dispositivo de treino sugerido: {0 if cuda else 'cpu'}")
        if cuda:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("torch: NAO INSTALADO")
        print("Dispositivo de treino sugerido: cpu")


if __name__ == "__main__":
    main()
