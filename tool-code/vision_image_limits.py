from __future__ import annotations

import base64
import math
from pathlib import Path

from fastapi import UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from app_config import (
    QWEN_MAX_IMAGE_ASPECT_RATIO,
    QWEN_MAX_IMAGE_BASE64_BYTES,
    QWEN_MAX_IMAGE_PIXELS,
    QWEN_MAX_VISION_IMAGES,
    QWEN_MIN_IMAGE_SIDE,
    VISION_UPLOADS_DIR,
    ensure_dirs,
    now_id,
)
from app_logging import log_vision, timed_vision


class VisionImageLimitError(ValueError):
    pass


def image_base64_size(path: Path) -> int:
    return len(base64.b64encode(path.read_bytes()))


def prepare_vision_uploads(uploads: list[UploadFile]) -> tuple[list[Path], list[str]]:
    ensure_dirs()
    warnings = []
    submitted = [upload for upload in uploads if upload.filename]
    log_vision("image_prepare.batch_start", submitted_count=len(submitted))
    if len(submitted) > QWEN_MAX_VISION_IMAGES:
        warnings.append(
            f"Only the first {QWEN_MAX_VISION_IMAGES} images were used; "
            f"{len(submitted) - QWEN_MAX_VISION_IMAGES} extra image(s) were ignored."
        )
        submitted = submitted[:QWEN_MAX_VISION_IMAGES]

    saved = []
    for upload in submitted:
        try:
            with timed_vision("image_prepare.single", filename=upload.filename):
                saved.append(prepare_single_image(upload))
        except VisionImageLimitError as exc:
            warnings.append(f"{Path(upload.filename or 'image').name}: {exc}")
            log_vision("image_prepare.single_rejected", filename=upload.filename, error=str(exc))
    log_vision("image_prepare.batch_end", saved_count=len(saved), warnings=warnings)
    return saved, warnings


def prepare_single_image(upload: UploadFile) -> Path:
    safe_stem = Path(upload.filename or "image").stem or "image"
    target = VISION_UPLOADS_DIR / f"{now_id()}_{safe_stem}.jpg"

    try:
        image = Image.open(upload.file)
        image = ImageOps.exif_transpose(image)
    except UnidentifiedImageError as exc:
        raise VisionImageLimitError("not a supported image file") from exc

    width, height = image.size
    log_vision(
        "image_prepare.opened",
        filename=upload.filename,
        original_width=width,
        original_height=height,
        mode=image.mode,
    )
    validate_image_geometry(width, height)
    image = resize_to_pixel_limit(image)
    log_vision(
        "image_prepare.resized",
        filename=upload.filename,
        width=image.size[0],
        height=image.size[1],
        pixels=image.size[0] * image.size[1],
    )
    image = flatten_to_rgb(image)
    save_compressed_jpeg(image, target)
    log_vision(
        "image_prepare.saved",
        filename=upload.filename,
        target=str(target),
        file_bytes=target.stat().st_size,
        base64_bytes=image_base64_size(target),
    )
    return target


def validate_image_geometry(width: int, height: int) -> None:
    if width < QWEN_MIN_IMAGE_SIDE or height < QWEN_MIN_IMAGE_SIDE:
        raise VisionImageLimitError(
            f"image is too small; width and height must both be at least {QWEN_MIN_IMAGE_SIDE}px"
        )
    ratio = max(width / height, height / width)
    if ratio > QWEN_MAX_IMAGE_ASPECT_RATIO:
        raise VisionImageLimitError(
            f"aspect ratio is too large; maximum allowed ratio is {QWEN_MAX_IMAGE_ASPECT_RATIO}:1"
        )


def resize_to_pixel_limit(image: Image.Image) -> Image.Image:
    width, height = image.size
    pixels = width * height
    if pixels <= QWEN_MAX_IMAGE_PIXELS:
        return image
    scale = math.sqrt(QWEN_MAX_IMAGE_PIXELS / pixels)
    resized = (
        max(QWEN_MIN_IMAGE_SIDE, int(width * scale)),
        max(QWEN_MIN_IMAGE_SIDE, int(height * scale)),
    )
    return image.resize(resized, Image.Resampling.LANCZOS)


def flatten_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or ("transparency" in image.info):
        background = Image.new("RGB", image.size, "white")
        alpha = image.convert("RGBA").getchannel("A")
        background.paste(image.convert("RGB"), mask=alpha)
        return background
    return image.convert("RGB")


def save_compressed_jpeg(image: Image.Image, target: Path) -> None:
    working = image
    for quality in (88, 82, 76, 70, 64, 58):
        working.save(target, format="JPEG", quality=quality, optimize=True)
        current_size = image_base64_size(target)
        log_vision("image_prepare.compress_try", target=str(target), quality=quality, base64_bytes=current_size)
        if current_size <= QWEN_MAX_IMAGE_BASE64_BYTES:
            return

    while image_base64_size(target) > QWEN_MAX_IMAGE_BASE64_BYTES:
        width, height = working.size
        next_size = (int(width * 0.85), int(height * 0.85))
        if next_size[0] < QWEN_MIN_IMAGE_SIDE or next_size[1] < QWEN_MIN_IMAGE_SIDE:
            target.unlink(missing_ok=True)
            raise VisionImageLimitError("image cannot be compressed below the 10MB base64 limit")
        working = working.resize(next_size, Image.Resampling.LANCZOS)
        working.save(target, format="JPEG", quality=58, optimize=True)
        log_vision(
            "image_prepare.downscale_try",
            target=str(target),
            width=next_size[0],
            height=next_size[1],
            base64_bytes=image_base64_size(target),
        )
