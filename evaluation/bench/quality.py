"""CPU-friendly, frame-wise full-reference metrics for uint8 THWC video."""
import torch
import torch.nn.functional as F


def sanity(video, expected_shape=None, eps=1e-6, min_luma=0.01, max_luma=0.99):
    reasons = []
    if video.ndim != 4 or video.shape[-1] != 3 or not video.numel():
        return ["expected nonempty [T,H,W,3] video"]
    if expected_shape is not None and tuple(video.shape) != tuple(expected_shape):
        reasons.append(f"shape mismatch: {tuple(video.shape)} != {tuple(expected_shape)}")
    if not torch.isfinite(video).all():
        reasons.append("non-finite pixels")
    if video.dtype != torch.uint8:
        reasons.append("expected uint8 pixels")
    x = video.float() / 255
    if (x.flatten(1, 2).std(dim=1, unbiased=False) < eps).all(dim=-1).any():
        reasons.append("constant frame")
    luma = (x * x.new_tensor([0.299, 0.587, 0.114])).sum(-1).mean((1, 2))
    if ((luma < min_luma) | (luma > max_luma)).any():
        reasons.append("mean luminance out of range")
    return reasons


def summarize(values):
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values), "frames": values}


@torch.inference_mode()
def compare(video, reference, with_lpips=True):
    if video.shape != reference.shape or not video.numel():
        raise ValueError("shape mismatch or empty video")
    result = {"psnr": [], "ssim": []}
    perceptual = None
    if with_lpips:
        try:
            import lpips
        except ImportError:
            pass
        else:
            perceptual = lpips.LPIPS(net="alex").cpu().eval()
            result["lpips"] = []
    coords = torch.arange(11, dtype=torch.float32) - 5
    gaussian = torch.exp(-coords.square() / (2 * 1.5**2))
    gaussian /= gaussian.sum()
    window = (gaussian[:, None] * gaussian[None, :])[None, None]
    for frame, ref in zip(video.cpu(), reference.cpu()):
        x, y = (f.float().permute(2, 0, 1)[None] / 255 for f in (frame, ref))
        mse = (x - y).square().mean()
        result["psnr"].append(float(-10 * torch.log10(mse)))
        weights = x.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]
        a, b = ((f * weights).sum(1, keepdim=True) for f in (x, y))

        def blur(f):
            return F.conv2d(F.pad(f, (5, 5, 5, 5), mode="replicate"), window)

        ma, mb = blur(a), blur(b)
        va, vb, cov = blur(a * a) - ma * ma, blur(b * b) - mb * mb, blur(a * b) - ma * mb
        ssim = ((2 * ma * mb + 0.01**2) * (2 * cov + 0.03**2)
                / ((ma.square() + mb.square() + 0.01**2) * (va + vb + 0.03**2)))
        result["ssim"].append(float(ssim.mean()))
        if perceptual is not None:
            result["lpips"].append(float(perceptual(x * 2 - 1, y * 2 - 1).mean()))
    return {name: summarize(values) for name, values in result.items()}


def verdict(metrics, reasons, has_reference, min_psnr=30, min_ssim=0.90, max_lpips=0.10,
            fps=None, previous_fps=None, max_fps_drop=None):
    reasons = list(reasons)
    for name, threshold in (("psnr", min_psnr), ("ssim", min_ssim)):
        if name in metrics and not metrics[name]["min"] >= threshold:
            reasons.append(f"{name} below {threshold}")
    if "lpips" in metrics and not metrics["lpips"]["max"] <= max_lpips:
        reasons.append(f"lpips above {max_lpips}")
    if max_fps_drop is not None and previous_fps and fps < previous_fps * (1 - max_fps_drop):
        reasons.append(f"fps drop exceeds {max_fps_drop:.1%}")
    return ("fail" if reasons else "pass" if has_reference else "no-reference"), reasons
