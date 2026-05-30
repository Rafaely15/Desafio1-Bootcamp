import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from screw_counter import IMAGE_EXTENSIONS, contar_parafusos


def iter_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Imagem unica")
    parser.add_argument("--source", help="Pasta com imagens")
    parser.add_argument("--model", default="runs_screws/yolo11_screws/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    if not args.image and not args.source:
        raise SystemExit("Use --image ou --source.")
    if args.conf < 0.05:
        print("Aviso: confianca muito baixa pode gerar falsos positivos.")

    target = Path(args.image or args.source)
    if not target.exists():
        raise FileNotFoundError(f"Caminho inexistente: {target}")
    images = iter_images(target)
    if not images:
        raise RuntimeError("Nenhuma imagem encontrada na origem.")

    csv_path = Path(args.output) / "contagens_parafusos.csv"
    for image in images:
        try:
            row = contar_parafusos(image, args.model, args.output, args.conf, csv_path=csv_path)
            print(f"{image.name}: {row['total_parafusos']} parafusos | confianca media {row['confianca_media']:.3f}")
        except Exception as exc:
            print(f"ERRO em {image}: {exc}")

    print(f"CSV atualizado em: {csv_path}")


if __name__ == "__main__":
    main()
