from __future__ import annotations

from typing import Dict


def build_prompt(filename: str, features: Dict[str, float]) -> str:
    geometry_lines = [
        f"- width: {features['width']}",
        f"- height: {features['height']}",
    ]
    color_lines = [
        f"- red_mean: {features['red_mean']}",
        f"- green_mean: {features['green_mean']}",
        f"- blue_mean: {features['blue_mean']}",
        f"- red_std: {features['red_std']}",
        f"- green_std: {features['green_std']}",
        f"- blue_std: {features['blue_std']}",
    ]
    intensity_lines = [
        f"- mean_intensity: {features['mean_intensity']}",
        f"- std_intensity: {features['std_intensity']}",
        f"- min_intensity: {features['min_intensity']}",
        f"- max_intensity: {features['max_intensity']}",
        f"- median_intensity: {features['median_intensity']}",
        f"- p10_intensity: {features['p10_intensity']}",
        f"- p90_intensity: {features['p90_intensity']}",
        f"- entropy: {features['entropy']}",
    ]
    structure_lines = [
        f"- edge_density: {features['edge_density']}",
        f"- gradient_strength: {features['gradient_strength']}",
    ]

    return f"""Statistical description for core SEM pore prompt generation

Image identifier: {filename}

Feature summary extracted from the image:

Geometry level
{chr(10).join(geometry_lines)}

Color and intensity level
{chr(10).join(color_lines)}
{chr(10).join(intensity_lines)}

Structural level
{chr(10).join(structure_lines)}

Instruction
Write a concise geological description of the pore structure implied by the statistics above.
Use one short sentence of about 15-25 words.
Emphasize pore distribution, pore connectivity, pore shape, and spatial heterogeneity.

Constraints
- Base the description only on the statistical summary.
- Do not mention that you can see an image or a mask.
- Do not infer rock type unless it is explicitly provided.
- Avoid introducing unsupported visual details.
"""
