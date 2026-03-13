"""
图像特效模块 - 模拟真实拍摄环境

添加白色光晕、光斑、亮度变化等效果，
使合成的数学题图像更接近真实手机拍摄的效果。
"""

import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import logging

logger = logging.getLogger(__name__)


def add_white_glare_effects(image, intensity_range=(0.75, 0.95),
                            count_range=(4, 8), radius_range=(150, 350)):
    """添加白色光晕效果（模拟过曝/强光照射）"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    for _ in range(random.randint(*count_range)):
        x = random.randint(0, image.width)
        y = random.randint(0, image.height)
        radius = random.randint(*radius_range)
        base_alpha = random.uniform(*intensity_range)

        # 中心高亮核心
        core_radius = radius // 3
        core_alpha = int(255 * min(0.98, base_alpha * 1.2))
        draw.ellipse([x-core_radius, y-core_radius, x+core_radius, y+core_radius],
                     fill=(255, 255, 255, core_alpha))

        # 渐变光晕
        for r in range(radius, core_radius, -5):
            dist = r - core_radius
            max_dist = radius - core_radius
            alpha = int(255 * base_alpha * (1.0 - (dist / max_dist) ** 0.8))
            alpha = max(0, min(255, alpha))
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(255, 255, 255, alpha))

    blur_radius = random.randint(8, 15)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return Image.alpha_composite(image, overlay)


def add_subtle_glare_spots(image, spot_count_range=(6, 15),
                           spot_size_range=(100, 250), opacity_range=(0.85, 0.95)):
    """添加离散光斑"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    layer = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(layer)

    for _ in range(random.randint(*spot_count_range)):
        if random.random() > 0.3:
            x, y = random.randint(0, image.width), random.randint(0, image.height // 2)
        else:
            x, y = random.randint(0, image.width), random.randint(0, image.height)

        size = random.randint(*spot_size_range)
        opacity = random.uniform(*opacity_range)

        core = size // 4
        draw.ellipse([x-core, y-core, x+core, y+core],
                     fill=(255, 255, 255, int(255 * min(0.95, opacity * 1.3))))

        for r in range(size, core, -5):
            alpha = int(255 * opacity * (1.0 - (r - core) / (size - core)))
            alpha = max(0, min(255, alpha))
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(255, 255, 255, alpha))

    layer = layer.filter(ImageFilter.GaussianBlur(radius=10))
    return Image.alpha_composite(image, layer)


def add_brightness_variation(image, variation_range=(0.95, 1.05)):
    """添加轻微亮度变化"""
    if image.mode == 'RGBA':
        alpha = image.split()[-1]
        rgb = image.convert('RGB')
    else:
        rgb, alpha = image, None

    factor = random.uniform(*variation_range)
    rgb = ImageEnhance.Brightness(rgb).enhance(factor)

    if alpha:
        rgb = rgb.convert('RGBA')
        rgb.putalpha(alpha)
    return rgb


def apply_realistic_effects(image, add_glare=True, add_brightness=True, glare_probability=0.7):
    """组合应用所有真实感效果"""
    result = image.copy()

    if add_glare and random.random() < glare_probability:
        if random.random() > 0.2:
            result = add_white_glare_effects(result)
        if random.random() > 0.15:
            result = add_subtle_glare_spots(
                result, opacity_range=(0.45, 0.75))

    if add_brightness:
        result = add_brightness_variation(result)

    return result
