def build_class_ids(samples_per_class: int = 50, num_classes: int = 1000) -> list[int]:
    """Flat list of class IDs aligned with build_prompts output order."""
    ids: list[int] = []
    for cls in range(num_classes):
        ids.extend([cls] * samples_per_class)
    return ids
