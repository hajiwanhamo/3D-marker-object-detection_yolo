import argparse
import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np


ORIGINAL_PATH = Path("/Users/hajiwan/Desktop/object_detection/new_tema/pointnet/make_top_id_projection.py")


def load_original_module():
    spec = importlib.util.spec_from_file_location(
        "make_top_id_projection_original_v2",
        ORIGINAL_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"원본 모듈 로드 실패: {ORIGINAL_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_wrapper_args(argv):
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--body_layer_mode",
        type=str,
        default="original",
        choices=["original", "keep_upper_down_expand"],
    )
    parser.add_argument("--body_base_top_candidate_q", type=float, default=0.85)
    parser.add_argument("--body_base_height_band_m", type=float, default=0.03)
    parser.add_argument("--body_base_adaptive_max_band_m", type=float, default=0.080)
    parser.add_argument("--body_down_expand_m", type=float, default=0.0)

    wrapper_args, remaining_args = parser.parse_known_args(argv)
    return wrapper_args, remaining_args


def make_patched_selector(original_select_top_id_layer, wrapper_args):
    def patched_select_top_id_layer(h, args):
        if wrapper_args.body_layer_mode == "original":
            return original_select_top_id_layer(h, args)

        if wrapper_args.body_layer_mode != "keep_upper_down_expand":
            raise ValueError(f"지원하지 않는 body_layer_mode: {wrapper_args.body_layer_mode}")

        # ------------------------------------------------------------
        # 1. 기존 adaptive 방식으로 기준 layer 계산
        # ------------------------------------------------------------
        base_args = copy.copy(args)

        base_args.height_mode = "adaptive"
        base_args.top_candidate_q = float(wrapper_args.body_base_top_candidate_q)
        base_args.height_band_m = float(wrapper_args.body_base_height_band_m)

        if hasattr(base_args, "adaptive_max_band_m"):
            base_args.adaptive_max_band_m = float(wrapper_args.body_base_adaptive_max_band_m)

        base_mask, base_center, base_band, base_info = original_select_top_id_layer(h, base_args)

        base_center = float(base_center)
        base_band = float(base_band)

        # 기존 기준 상단/하단
        base_upper = base_center + base_band
        base_lower = base_center - base_band

        # ------------------------------------------------------------
        # 2. 상단 고정 + 하단만 확장
        # ------------------------------------------------------------
        down = float(wrapper_args.body_down_expand_m)
        if down < 0:
            raise ValueError("--body_down_expand_m must be >= 0")

        new_upper = base_upper
        new_lower = base_lower - down

        new_center = (new_upper + new_lower) / 2.0
        new_band = (new_upper - new_lower) / 2.0

        mask = (h >= new_lower) & (h <= new_upper)
        selected = h[mask]

        if selected.size == 0:
            print("[KEEP_UPPER_WARN] selected point가 0개라서 base layer로 fallback")
            mask = base_mask
            selected = h[mask]
            new_upper = base_upper
            new_lower = base_lower
            new_center = base_center
            new_band = base_band
            down = 0.0

        center_shift = new_center - base_center

        # ------------------------------------------------------------
        # 3. 반드시 로그에 출력
        # ------------------------------------------------------------
        print(
            "[KEEP_UPPER_APPLIED] "
            f"down={down:.6f} | "
            f"base_center={base_center:.6f} | "
            f"base_band={base_band:.6f} | "
            f"base_lower={base_lower:.6f} | "
            f"base_upper={base_upper:.6f} | "
            f"new_center={new_center:.6f} | "
            f"new_band={new_band:.6f} | "
            f"new_lower={new_lower:.6f} | "
            f"new_upper={new_upper:.6f} | "
            f"center_shift={center_shift:.6f} | "
            f"selected_points={int(np.count_nonzero(mask))}"
        )

        layer_info = {
            "height_mode": "keep_upper_down_expand",
            "selection_rule": "upper_fixed_lower_expanded",
            "body_down_expand_m": float(down),
            "base_id_height_center": float(base_center),
            "base_height_band_m_used": float(base_band),
            "base_lower": float(base_lower),
            "base_upper_fixed": float(base_upper),
            "id_height_center": float(new_center),
            "height_band_m": float(new_band),
            "height_band_m_used": float(new_band),
            "new_lower_expanded": float(new_lower),
            "new_upper_fixed": float(new_upper),
            "center_shift_m": float(center_shift),
            "selected_height_min": float(selected.min()) if selected.size > 0 else None,
            "selected_height_max": float(selected.max()) if selected.size > 0 else None,
        }

        return mask, float(new_center), float(new_band), layer_info

    return patched_select_top_id_layer


def main():
    wrapper_args, remaining_args = parse_wrapper_args(sys.argv[1:])

    module = load_original_module()

    if not hasattr(module, "select_top_id_layer"):
        raise AttributeError("원본 make_top_id_projection.py에 select_top_id_layer 함수가 없음")

    module.select_top_id_layer = make_patched_selector(
        original_select_top_id_layer=module.select_top_id_layer,
        wrapper_args=wrapper_args,
    )

    print("=" * 80)
    print("[WRAPPER_V2] keep-upper projection wrapper")
    print(f"[WRAPPER_V2] original = {ORIGINAL_PATH}")
    print(f"[WRAPPER_V2] body_layer_mode = {wrapper_args.body_layer_mode}")
    print(f"[WRAPPER_V2] body_base_top_candidate_q = {wrapper_args.body_base_top_candidate_q}")
    print(f"[WRAPPER_V2] body_base_height_band_m = {wrapper_args.body_base_height_band_m}")
    print(f"[WRAPPER_V2] body_base_adaptive_max_band_m = {wrapper_args.body_base_adaptive_max_band_m}")
    print(f"[WRAPPER_V2] body_down_expand_m = {wrapper_args.body_down_expand_m}")
    print("=" * 80)

    # 원본 argparse가 wrapper 인자를 모르게 하기 위해 제거
    sys.argv = [str(ORIGINAL_PATH)] + remaining_args

    module.main()


if __name__ == "__main__":
    main()
