from __future__ import annotations

DEFAULT_PROMPTS = [
    "a photo of a {class}",
    "a clinical photo of a {class}",
    "a chest X-ray photo of a {class}",
]


def normalize_class_name(name: str) -> str:
    # ImageFolder names such as COVID19 are made more natural for CLIP text input.
    cleaned = name.replace("_", " ").replace("-", " ").strip()
    if cleaned.upper() == "COVID19":
        return "COVID-19"
    return cleaned.lower()


def build_prompts(class_names: list[str], templates: list[str]) -> list[list[str]]:
    return [
        [template.format(**{"class": normalize_class_name(name)}) for template in templates]
        for name in class_names
    ]
