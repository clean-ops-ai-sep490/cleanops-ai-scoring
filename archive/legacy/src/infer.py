"""
Inference Script: Before/After Cleaning Quality Report
=======================================================
Input : 2 ảnh (before.jpg, after.jpg) + environment type
Output: Visual report PNG với đầy đủ metrics theo spec

Cách chạy:
    python src/infer.py --before img/before.jpg --after img/after.jpg
    python src/infer.py --before b.jpg --after a.jpg --env RESTROOM
    python src/infer.py --demo   # Chạy demo với synthetic images

Environments:
    LOBBY_CORRIDOR, RESTROOM, BASEMENT_PARKING,
    GLASS_SURFACE, OUTDOOR_LANDSCAPE, HOSPITAL_OR
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from torchvision import transforms

from model import CleaningQualityNet, build_model


# ─── Config theo môi trường (từ spec) ────────────────────────────────────────

ENV_CONFIG = {
    "LOBBY_CORRIDOR": {
        "name": "Sảnh / Hành lang / Thang máy",
        "approve_threshold": 0.90,
        "coverage_pass": 10.0,      # dirt_coverage_after phải < 10%
        "coverage_reduction_pass": 90.0,
        "check_focus": "dirt_coverage",
        "icon": "🏢",
    },
    "RESTROOM": {
        "name": "Nhà vệ sinh",
        "approve_threshold": 0.85,
        "coverage_pass": 5.0,
        "coverage_reduction_pass": 85.0,
        "check_focus": "dirt_coverage",
        "icon": "🚻",
    },
    "BASEMENT_PARKING": {
        "name": "Tầng hầm để xe",
        "approve_threshold": 0.80,
        "coverage_pass": 15.0,
        "coverage_reduction_pass": 80.0,
        "check_focus": "detected_stains",
        "icon": "🅿️",
    },
    "GLASS_SURFACE": {
        "name": "Bề mặt kính",
        "approve_threshold": 0.90,
        "coverage_pass": 5.0,
        "coverage_reduction_pass": 92.0,
        "check_focus": "dirt_coverage",
        "icon": "🪟",
    },
    "OUTDOOR_LANDSCAPE": {
        "name": "Ngoại cảnh & Sân vườn",
        "approve_threshold": 0.80,
        "coverage_pass": 20.0,
        "coverage_reduction_pass": 75.0,
        "check_focus": "abnormal_objects",
        "icon": "🌿",
    },
    "HOSPITAL_OR": {
        "name": "Phòng mổ / Khu y tế",
        "approve_threshold": 0.95,   # Cực gắt theo spec
        "coverage_pass": 1.0,
        "coverage_reduction_pass": 99.0,
        "check_focus": "dirt_coverage",
        "icon": "🏥",
    },
}


# ─── Inference Engine ─────────────────────────────────────────────────────────

class QualityInferenceEngine:
    """
    Engine chạy inference cho Before/After pair.
    """
    
    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    def __init__(self, model_path: str = None, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.model = build_model(mode="single", freeze=False).to(self.device)
        
        if model_path and Path(model_path).exists():
            print(f"[*] Load model từ {model_path}")
            ckpt = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            print(f"    Epoch: {ckpt.get('epoch', '?')} | Best val loss: {ckpt.get('best_val_loss', '?'):.4f}")
        else:
            print("[WARN] Không tìm thấy model, dùng untrained weights (chỉ để demo pipeline)")
        
        self.model.eval()
    
    def preprocess(self, img_path: str) -> torch.Tensor:
        img = Image.open(img_path).convert("RGB")
        return self.TRANSFORM(img).unsqueeze(0).to(self.device)
    
    @torch.no_grad()
    def analyze_single(self, img_path: str) -> dict:
        """Phân tích 1 ảnh đơn lẻ."""
        x = self.preprocess(img_path)
        raw = self.model(x)
        
        return {
            "detected_stains": round((raw["stains_norm"].item() * 20)),
            "dirt_coverage": round(raw["coverage_norm"].item() * 100, 1),
            "abnormal_objects": round((raw["objects_norm"].item() * 10)),
            "quality_score": round(raw["quality_norm"].item() * 100, 1),
        }
    
    def compute_confidence(self, before: dict, after: dict, env: str) -> float:
        """
        Tính Confidence Score theo rule-based logic kết hợp với model score.
        Logic theo spec:
          - Cải thiện rõ → confidence cao
          - Môi trường đặc biệt (HOSPITAL_OR) → ngưỡng gắt hơn
        """
        quality_improvement = after["quality_score"] - before["quality_score"]
        stain_reduction = before["detected_stains"] - after["detected_stains"]
        coverage_reduction = before["dirt_coverage"] - after["dirt_coverage"]
        objects_reduction = before["abnormal_objects"] - after["abnormal_objects"]
        
        # Base confidence từ improvement
        base = min(100, max(0,
            quality_improvement * 0.6 +
            (stain_reduction / max(before["detected_stains"], 1)) * 20 +
            (coverage_reduction / max(before["dirt_coverage"], 1)) * 20
        ))
        
        # Penalty nếu sau vẫn còn nhiều vết bẩn
        if after["dirt_coverage"] > 30:
            base *= 0.7
        if after["detected_stains"] > 5:
            base *= 0.8
        if after["abnormal_objects"] > 2:
            base *= 0.85
        
        # Hospital mode: confidence bị giảm mạnh nếu không đạt
        if env == "HOSPITAL_OR":
            if after["dirt_coverage"] > 5:
                base = min(base, 30)
        
        return round(min(100, max(5, base)), 1)
    
    def get_verdict(self, confidence: float, env: str, after: dict) -> str:
        """
        Bẻ luồng workflow theo spec:
          > 90% → AUTO_APPROVE
          < 50% → AUTO_REJECT
          50-90% → MANUAL_REVIEW
        """
        cfg = ENV_CONFIG[env]
        
        if confidence >= cfg["approve_threshold"] * 100:
            # Extra check cho hospital
            if env == "HOSPITAL_OR":
                if after["dirt_coverage"] > 1:
                    return "AUTO_REJECT"
            return "AUTO_APPROVE"
        elif confidence < 50:
            return "AUTO_REJECT"
        else:
            return "MANUAL_REVIEW"
    
    def analyze_pair(self, before_path: str, after_path: str, env: str = "LOBBY_CORRIDOR") -> dict:
        """
        Phân tích cặp Before/After và tổng hợp kết quả.
        """
        before = self.analyze_single(before_path)
        after = self.analyze_single(after_path)
        
        coverage_reduction = (
            (before["dirt_coverage"] - after["dirt_coverage"]) / 
            max(before["dirt_coverage"], 0.1) * 100
        )
        
        confidence = self.compute_confidence(before, after, env)
        verdict = self.get_verdict(confidence, env, after)
        
        # Issues detection
        issues = []
        positives = []
        
        if after["dirt_coverage"] > ENV_CONFIG[env]["coverage_pass"]:
            issues.append(f"dirt_coverage sau vẫn cao: {after['dirt_coverage']:.1f}% (ngưỡng: {ENV_CONFIG[env]['coverage_pass']}%)")
        
        if after["detected_stains"] > 3:
            issues.append(f"Còn {after['detected_stains']} vết bẩn cục bộ sau khi dọn")
        
        if after["abnormal_objects"] > 0:
            issues.append(f"Còn {after['abnormal_objects']} dị vật / rác sót lại")
        
        if coverage_reduction >= ENV_CONFIG[env]["coverage_reduction_pass"]:
            positives.append(f"Diện tích bẩn giảm {coverage_reduction:.0f}% (đạt ngưỡng {ENV_CONFIG[env]['coverage_reduction_pass']}%)")
        
        if before["detected_stains"] - after["detected_stains"] >= 3:
            positives.append(f"Giảm {before['detected_stains'] - after['detected_stains']} vết bẩn cục bộ")
        
        if after["quality_score"] > 70:
            positives.append(f"Quality Score đạt {after['quality_score']:.0f}/100")
        
        return {
            "environment": env,
            "environment_name": ENV_CONFIG[env]["name"],
            "before": before,
            "after": after,
            "coverage_reduction": round(coverage_reduction, 1),
            "confidence_score": confidence,
            "quality_score_after": after["quality_score"],
            "verdict": verdict,
            "issues": issues,
            "positives": positives,
        }


# ─── Visual Report Generator ──────────────────────────────────────────────────

VERDICT_COLORS = {
    "AUTO_APPROVE": "#3d9e3d",
    "MANUAL_REVIEW": "#e07b00",
    "AUTO_REJECT": "#c0392b",
}
VERDICT_LABELS = {
    "AUTO_APPROVE": "✓  TỰ ĐỘNG DUYỆT",
    "MANUAL_REVIEW": "⚠  CẦN GIÁM SÁT VIÊN",
    "AUTO_REJECT": "✕  ĐÁNH TRƯỢT",
}
VERDICT_BG = {
    "AUTO_APPROVE": "#f0faf0",
    "MANUAL_REVIEW": "#fff8ee",
    "AUTO_REJECT": "#fdf0ef",
}


def render_score_bar(ax, value: float, max_val: float, label: str, color: str, unit: str = ""):
    """Vẽ một bar metric đơn."""
    ax.set_xlim(0, max_val)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")
    
    pct = value / max_val
    bar_color = color
    
    # Background bar
    ax.barh(0, max_val, height=0.35, color="#f0f0f0", edgecolor="none")
    # Value bar
    ax.barh(0, value, height=0.35, color=bar_color, edgecolor="none")
    # Label
    ax.text(-max_val * 0.02, 0, label, va="center", ha="right", fontsize=9, color="#555")
    # Value text
    ax.text(value + max_val * 0.02, 0, f"{value:.1f}{unit}", va="center", ha="left", fontsize=9, fontweight="bold", color=bar_color)


def generate_report(
    before_path: str,
    after_path: str,
    result: dict,
    output_path: str = "outputs/quality_report.png"
):
    """
    Tạo visual report PNG đầy đủ.
    Layout:
        Header (title + verdict)
        Row 1: Ảnh Before | Ảnh After
        Row 2: Metrics grid (4 tham số)
        Row 3: Score bars (quality, confidence, coverage reduction)
        Row 4: Contamination breakdown | Issues & Positives
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    before_img = Image.open(before_path).convert("RGB")
    after_img = Image.open(after_path).convert("RGB")
    
    r = result
    verdict = r["verdict"]
    env_cfg = ENV_CONFIG[r["environment"]]
    
    fig = plt.figure(figsize=(14, 18), facecolor="white")
    fig.patch.set_facecolor("white")
    
    gs = gridspec.GridSpec(
        5, 4,
        figure=fig,
        hspace=0.45, wspace=0.35,
        top=0.96, bottom=0.03, left=0.05, right=0.97,
        height_ratios=[0.8, 3, 1.2, 1.2, 2.2]
    )
    
    # ── Header ────────────────────────────────────────────────────────────────
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis("off")
    
    verdict_color = VERDICT_COLORS[verdict]
    verdict_bg = VERDICT_BG[verdict]
    
    ax_header.add_patch(FancyBboxPatch(
        (0, 0.05), 1, 0.9,
        transform=ax_header.transAxes,
        boxstyle="round,pad=0.02",
        facecolor=verdict_bg,
        edgecolor=verdict_color, linewidth=2
    ))
    
    ax_header.text(0.5, 0.72, "AI QUALITY SCORING REPORT",
                   ha="center", va="center", fontsize=15, fontweight="bold",
                   color="#222", transform=ax_header.transAxes)
    
    ax_header.text(0.5, 0.32,
                   f"{env_cfg['name']}   |   {VERDICT_LABELS[verdict]}",
                   ha="center", va="center", fontsize=12,
                   color=verdict_color, fontweight="bold",
                   transform=ax_header.transAxes)
    
    # ── Ảnh Before / After ────────────────────────────────────────────────────
    ax_before = fig.add_subplot(gs[1, :2])
    ax_after = fig.add_subplot(gs[1, 2:])
    
    for ax, img, label, color in [
        (ax_before, before_img, "BEFORE", "#c0392b"),
        (ax_after, after_img, "AFTER", "#3d9e3d")
    ]:
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(label, fontsize=13, fontweight="bold", color=color, pad=6)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(2)
    
    # ── Metrics Grid (4 tham số) ──────────────────────────────────────────────
    metrics = [
        ("Vết bẩn\n(detected_stains)",
         r["before"]["detected_stains"], r["after"]["detected_stains"], 20, "#e74c3c"),
        ("Mảng bám/Bụi\n(dirt_coverage %)",
         r["before"]["dirt_coverage"], r["after"]["dirt_coverage"], 100, "#e67e22"),
        ("Dị vật\n(abnormal_objects)",
         r["before"]["abnormal_objects"], r["after"]["abnormal_objects"], 10, "#8e44ad"),
        ("Quality Score",
         r["before"]["quality_score"], r["after"]["quality_score"], 100, "#2980b9"),
    ]
    
    for col, (label, b_val, a_val, max_v, color) in enumerate(metrics):
        ax = fig.add_subplot(gs[2, col])
        ax.axis("off")
        
        delta = a_val - b_val
        delta_pct = (delta / max(abs(b_val), 0.1)) * 100
        
        # Card background
        ax.add_patch(FancyBboxPatch(
            (0.05, 0.05), 0.9, 0.9,
            transform=ax.transAxes,
            boxstyle="round,pad=0.02",
            facecolor="#f8f9fa", edgecolor="#dee2e6", linewidth=0.8
        ))
        
        ax.text(0.5, 0.82, label, ha="center", va="center",
                fontsize=8, color="#666", transform=ax.transAxes)
        
        ax.text(0.28, 0.48, str(b_val if isinstance(b_val, int) else f"{b_val:.0f}"),
                ha="center", va="center", fontsize=20, fontweight="bold",
                color="#c0392b", transform=ax.transAxes)
        
        ax.text(0.5, 0.48, "→", ha="center", va="center",
                fontsize=14, color="#aaa", transform=ax.transAxes)
        
        ax.text(0.72, 0.48, str(a_val if isinstance(a_val, int) else f"{a_val:.0f}"),
                ha="center", va="center", fontsize=20, fontweight="bold",
                color="#3d9e3d" if delta <= 0 else "#c0392b",
                transform=ax.transAxes)
        
        delta_str = f"{'↓' if delta <= 0 else '↑'} {abs(delta_pct):.0f}%"
        delta_color = "#3d9e3d" if delta <= 0 else "#c0392b"
        ax.text(0.5, 0.18, delta_str, ha="center", va="center",
                fontsize=10, color=delta_color, fontweight="bold",
                transform=ax.transAxes)
    
    # ── Score Bars ────────────────────────────────────────────────────────────
    score_data = [
        ("Quality Score (sau)", r["quality_score_after"], 100, "#2980b9", ""),
        ("Confidence Score", r["confidence_score"], 100,
         VERDICT_COLORS[verdict], "%"),
        ("Coverage Reduction", r["coverage_reduction"], 100, "#16a085", "%"),
    ]
    
    for col, (label, val, max_v, color, unit) in enumerate(score_data):
        ax = fig.add_subplot(gs[3, col])
        render_score_bar(ax, val, max_v, label, color, unit)
    
    # Ngưỡng decision
    ax_thresh = fig.add_subplot(gs[3, 3])
    ax_thresh.axis("off")
    thresh_text = (
        "Ngưỡng bẻ luồng:\n"
        f"  > {ENV_CONFIG[r['environment']]['approve_threshold']*100:.0f}%  →  Auto-approve\n"
        "  50–90%  →  Manual review\n"
        "  < 50%   →  Auto-reject"
    )
    ax_thresh.text(0.05, 0.7, thresh_text, va="top", fontsize=8.5, color="#555",
                   fontfamily="monospace", transform=ax_thresh.transAxes,
                   linespacing=1.8)
    
    # ── Issues ────────────────────────────────────────────────
    ax_issues = fig.add_subplot(gs[4, :])
    
    # Issues & Positives
    ax_issues.axis("off")
    ax_issues.text(0.5, 0.97, "Nhận xét chi tiết",
                   ha="center", va="top", fontsize=11, fontweight="bold", color="#333",
                   transform=ax_issues.transAxes)
    
    y_pos = 0.85
    for note in r.get("positives", []):
        ax_issues.text(0.04, y_pos, f"✓  {note}",
                       ha="left", va="top", fontsize=9, color="#3d9e3d",
                       transform=ax_issues.transAxes, wrap=True)
        y_pos -= 0.14
    
    for issue in r.get("issues", []):
        if y_pos < 0.05:
            break
        ax_issues.text(0.04, y_pos, f"⚠  {issue}",
                       ha="left", va="top", fontsize=9, color="#e07b00",
                       transform=ax_issues.transAxes, wrap=True)
        y_pos -= 0.14
    
    if not r.get("issues") and not r.get("positives"):
        ax_issues.text(0.5, 0.5, "Không có ghi chú đặc biệt",
                       ha="center", va="center", fontsize=9, color="#aaa",
                       transform=ax_issues.transAxes)
    
    plt.savefig(output_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[OK] Report đã lưu → {output_path}")
    return output_path


# ─── Demo với synthetic images ────────────────────────────────────────────────

def create_demo_images(out_dir: str = "outputs/demo"):
    """Tạo ảnh demo khi không có ảnh thật."""
    import numpy as np
    from PIL import ImageDraw, ImageFilter
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    
    # BEFORE: sàn bẩn với nhiều stains
    arr_before = np.ones((480, 640, 3), dtype=np.uint8) * np.array([195, 190, 183])
    arr_before += rng.randint(-10, 10, arr_before.shape).astype(np.int8)
    arr_before = np.clip(arr_before, 0, 255).astype(np.uint8)
    img_before = Image.fromarray(arr_before)
    draw = ImageDraw.Draw(img_before)
    for _ in range(15):
        x, y = rng.randint(0, 580), rng.randint(0, 420)
        w, h = rng.randint(10, 60), rng.randint(8, 40)
        color = tuple(rng.randint(50, 100, 3).tolist())
        draw.ellipse([x, y, x+w, y+h], fill=color)
    img_before = img_before.filter(ImageFilter.GaussianBlur(0.5))
    before_path = f"{out_dir}/demo_before.jpg"
    img_before.save(before_path, quality=92)
    
    # AFTER: sàn sạch
    arr_after = np.ones((480, 640, 3), dtype=np.uint8) * np.array([215, 212, 205])
    arr_after += rng.randint(-5, 5, arr_after.shape).astype(np.int8)
    arr_after = np.clip(arr_after, 0, 255).astype(np.uint8)
    img_after = Image.fromarray(arr_after)
    draw2 = ImageDraw.Draw(img_after)
    # Chỉ 1-2 stain nhỏ còn sót
    for _ in range(2):
        x, y = rng.randint(0, 580), rng.randint(0, 420)
        draw2.ellipse([x, y, x+10, y+8], fill=(160, 150, 140))
    after_path = f"{out_dir}/demo_after.jpg"
    img_after.save(after_path, quality=92)
    
    print(f"[OK] Demo images: {before_path}, {after_path}")
    return before_path, after_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cleaning Quality Inference")
    parser.add_argument("--before", type=str, default=None)
    parser.add_argument("--after", type=str, default=None)
    parser.add_argument("--env", type=str, default="LOBBY_CORRIDOR",
                        choices=list(ENV_CONFIG.keys()))
    parser.add_argument("--model", type=str, default="models/checkpoint_best.pth")
    parser.add_argument("--output", type=str, default="outputs/quality_report.png")
    parser.add_argument("--demo", action="store_true", help="Chạy demo với synthetic images")
    args = parser.parse_args()
    
    # Demo mode
    if args.demo or (args.before is None or args.after is None):
        print("[*] Chạy demo mode...")
        before_path, after_path = create_demo_images()
    else:
        before_path, after_path = args.before, args.after
    
    # Check files
    for p in [before_path, after_path]:
        if not Path(p).exists():
            print(f"[ERROR] File không tồn tại: {p}")
            sys.exit(1)
    
    # Load engine
    engine = QualityInferenceEngine(model_path=args.model)
    
    # Phân tích
    print(f"\n[*] Phân tích cặp Before/After...")
    print(f"    Before: {before_path}")
    print(f"    After:  {after_path}")
    print(f"    Môi trường: {args.env} ({ENV_CONFIG[args.env]['name']})")
    
    result = engine.analyze_pair(before_path, after_path, args.env)
    
    # Print kết quả
    print(f"\n{'='*55}")
    print(f"  KẾT QUẢ PHÂN TÍCH")
    print(f"{'='*55}")
    print(f"  Môi trường: {result['environment_name']}")
    print()
    print(f"  {'Tham số':<28} {'Before':>8} {'After':>8}")
    print(f"  {'-'*48}")
    print(f"  {'detected_stains':<28} {result['before']['detected_stains']:>8} {result['after']['detected_stains']:>8}")
    print(f"  {'dirt_coverage (%)':<28} {result['before']['dirt_coverage']:>8.1f} {result['after']['dirt_coverage']:>8.1f}")
    print(f"  {'abnormal_objects':<28} {result['before']['abnormal_objects']:>8} {result['after']['abnormal_objects']:>8}")
    print(f"  {'quality_score':<28} {result['before']['quality_score']:>8.1f} {result['after']['quality_score']:>8.1f}")
    print()
    print(f"  Coverage Reduction:  {result['coverage_reduction']:.1f}%")
    print(f"  Confidence Score:    {result['confidence_score']:.1f}%")
    print()
    
    v = result["verdict"]
    v_color = {"AUTO_APPROVE": "\033[92m", "MANUAL_REVIEW": "\033[93m", "AUTO_REJECT": "\033[91m"}
    reset = "\033[0m"
    print(f"  VERDICT: {v_color.get(v, '')}{VERDICT_LABELS[v]}{reset}")
    
    if result["positives"]:
        print(f"\n  Điểm tốt:")
        for p in result["positives"]:
            print(f"    ✓ {p}")
    
    if result["issues"]:
        print(f"\n  Vấn đề:")
        for issue in result["issues"]:
            print(f"    ⚠ {issue}")
    
    print(f"{'='*55}\n")
    
    # Render visual report
    generate_report(before_path, after_path, result, args.output)
    
    # Lưu JSON
    json_path = args.output.replace(".png", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in result.items() if k not in ["before", "after"]}, f, indent=2, ensure_ascii=False)
    print(f"[OK] JSON kết quả → {json_path}")


if __name__ == "__main__":
    main()
