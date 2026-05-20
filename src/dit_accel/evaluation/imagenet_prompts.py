"""ImageNet-1K class names and prompt template."""
import json
from pathlib import Path

PROMPT_TEMPLATE = "a photo of a {label}"


def load_class_names(path: str | None = None) -> list[str]:
    if path is None:
        here = Path(__file__).parent
        path = here / "imagenet_classes.txt"
        if not path.exists():
            try:
                import cleanfid
                cf_dir = Path(cleanfid.__file__).parent
                candidate = cf_dir / "stats" / "imagenet_class_index.json"
                if candidate.exists():
                    with open(candidate) as f:
                        idx = json.load(f)
                    return [idx[str(i)][1].replace("_", " ") for i in range(1000)]
            except ImportError:
                pass
            raise FileNotFoundError(f"imagenet_classes.txt not found at {path}.")
    with open(path) as f:
        names = [line.strip() for line in f if line.strip()]
    if len(names) != 1000:
        raise ValueError(f"expected 1000 class names, got {len(names)}")
    return names


def build_prompts(samples_per_class: int = 50, class_names: list[str] | None = None) -> list[str]:
    if class_names is None:
        class_names = load_class_names()
    prompts: list[str] = []
    for name in class_names:
        prompts.extend([PROMPT_TEMPLATE.format(label=name)] * samples_per_class)
    return prompts
