"""Tracking-side view of the shared TCP/contact adapter and moving-part prompt artifacts.

Loaded once per sequence in ``slide_window_tracking``.  Maps each tracking-sequence index
(after optional pingpong) to its original FreeArt frame, rescales full-resolution pixel
evidence to the resized tracking camera (``u_track = K_track @ p_cam``), and exposes
per-window slices.  Every feature is opt-in; with all flags off no artifact is read.

Frame kinds:
    forward        original frame i
    reversed_copy  pingpong replay of original frame j (image identity verified)
    duplicate      turning-point or ``duplicate_last_frame`` repeat; never used for priors
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from samuele.tcp_contact_evidence import load_tcp_contact_artifact, pinhole, sha256_file  # noqa: E402

KIND_FORWARD, KIND_REVERSED, KIND_DUPLICATE = "forward", "reversed_copy", "duplicate"
SEED_ARTIFACT = "moving_seed_maps.npz"


def build_sequence_index_map(num_original, pingpong, exclude_boundary):
    """Explicit replica of ``load_images`` pingpong ordering."""
    original = list(range(num_original))
    kinds = [KIND_FORWARD] * num_original
    if pingpong and num_original > 1:
        if exclude_boundary:
            back = list(reversed(range(num_original - 1)))
            original += back
            kinds += [KIND_REVERSED] * len(back)
        else:
            back = list(reversed(range(num_original)))
            original += back
            kinds += [KIND_DUPLICATE] + [KIND_REVERSED] * (len(back) - 1)
    return dict(original=np.asarray(original, dtype=np.int64), kind=kinds)


def verify_sequence_map(rgbs, sequence_map):
    """Reversed copies are usable only if their image equals the mapped original image."""
    usable = np.array([kind != KIND_DUPLICATE for kind in sequence_map["kind"]])
    if len(rgbs) != len(usable):
        raise ValueError(f"Sequence map length {len(usable)} != loaded frames {len(rgbs)}")
    for index, (orig, kind) in enumerate(zip(sequence_map["original"], sequence_map["kind"])):
        if kind == KIND_REVERSED and not np.array_equal(rgbs[index], rgbs[int(orig)]):
            usable[index] = False
    return usable


def _resolve_artifact(path_value, input_dir, default_relative):
    if path_value in (None, ""):
        return Path(input_dir) / default_relative
    path = Path(path_value)
    return path if path.is_absolute() else Path(input_dir) / path


class TrackingPriorContext:
    def __init__(self, args, resized_loader, input_dir, num_original):
        self.phase_a = bool(getattr(args, "tcp_supervision_enabled", False))
        self.phase_b = bool(getattr(args, "moving_prompt_enabled", False))
        self.phase_c = bool(getattr(args, "contact_surface_enabled", False))
        if getattr(args, "contact_multiview_refine_enabled", False):
            raise NotImplementedError(
                "contact_multiview_refine_enabled: the optional post-ICP multiview refinement is not implemented; "
                "validate per-window Phase C first (see docs/tcp_contact_tracking.md).")
        self.input_dir = Path(input_dir)
        self.num_original = int(num_original)
        self.sequence_map = build_sequence_index_map(
            num_original, bool(getattr(args, "pingpong", True)), not bool(getattr(args, "include_turn_frame", False)))
        self.usable = verify_sequence_map(resized_loader.data["rgbs"], self.sequence_map)
        self.K_track = np.asarray(resized_loader.data["cam_K"], dtype=np.float64)
        self.height, self.width = resized_loader.data["rgbs"][0].shape[:2]
        self.report = dict(sequence_length=len(self.usable), num_original=self.num_original,
                           reversed_copies=int(sum(k == KIND_REVERSED for k in self.sequence_map["kind"])),
                           duplicates=int(sum(k == KIND_DUPLICATE for k in self.sequence_map["kind"])),
                           unusable_frames=int((~self.usable).sum()))
        self.tcp = None
        self.seeds = None
        if self.phase_a or self.phase_c:
            self._load_tcp(args)
        if self.phase_b:
            self._load_seeds(args)

    @property
    def enabled(self):
        return self.phase_a or self.phase_b or self.phase_c

    # ------------------------------------------------------------------ loading
    def _check_conversion(self, manifest, label):
        conversion = self.input_dir / "metadata/freeart_conversion.json"
        expected = manifest.get("freeart_conversion_sha256")
        if conversion.is_file() and expected and sha256_file(conversion) != expected:
            raise ValueError(f"{label} was built for a different FreeArt conversion than {conversion}")

    def _load_tcp(self, args):
        path = _resolve_artifact(getattr(args, "tcp_contact_artifact", None), self.input_dir, "metadata/tcp_contact")
        arrays, manifest = load_tcp_contact_artifact(path)
        self._check_conversion(manifest, f"TCP artifact {path}")
        if len(arrays["local_frame_index"]) != self.num_original:
            raise ValueError(f"TCP artifact has {len(arrays['local_frame_index'])} frames, "
                             f"tracking input has {self.num_original}")
        K_full = arrays["K_fullres"]
        size = (self.width, self.height)
        seq = self.sequence_map["original"]
        track_point = getattr(args, "tcp_track_point", "tcp") or "tcp"
        if track_point == "tcp":
            p_tcp = arrays["p_tcp_cam_m"][seq]
        elif track_point == "secondary_feature":
            if "p_feature2_cam_m" not in arrays:
                raise ValueError("tcp_track_point=secondary_feature needs secondary_feature_local_tool_m in the tool "
                                 f"geometry used to build {path}")
            p_tcp = arrays["p_feature2_cam_m"][seq]
        else:
            raise ValueError(f"Unknown tcp_track_point {track_point!r}; use 'tcp' or 'secondary_feature'")
        p_contact = arrays["p_contact_cam_m"][seq]
        uv_tcp, valid_tcp = pinhole(self.K_track, p_tcp, size)
        uv_contact, valid_contact = pinhole(self.K_track, p_contact, size)
        self.tcp = dict(uv=uv_tcp, valid_projection=valid_tcp, z=p_tcp[:, 2], p_cam=p_tcp,
                        contact_uv=uv_contact, contact_valid_projection=valid_contact, contact_p_cam=p_contact,
                        contact_active=arrays["contact_active"][seq].astype(bool),
                        source_frame_index=arrays["source_frame_index"][seq],
                        scale=float(self.K_track[0, 0] / K_full[0, 0]), path=str(path),
                        manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                        contact_point_calibrated=bool(manifest.get("tool_geometry", {}).get("contact_point_calibrated")),
                        contact_source=manifest.get("contact", {}).get("contact_source"))
        state = arrays.get("contact_state")
        if state is None:  # artifacts written before the three-state contract
            state = np.where(arrays["contact_active"].astype(bool), 1, -1).astype(np.int8)
        self.tcp["contact_state"] = state[seq]
        if "T_base_ee" in arrays and "T_base_camera" in arrays:   # tool axis for contact_exclude_tool_radius_m
            T_cam_base = np.linalg.inv(np.asarray(arrays["T_base_camera"], dtype=np.float64))
            ee_cam = np.einsum("ij,nj->ni", T_cam_base[:3, :3], np.asarray(arrays["T_base_ee"])[:, :3, 3]) + T_cam_base[:3, 3]
            self.tcp["tool_tip_cam"] = np.asarray(arrays["p_tcp_cam_m"])[seq]   # calibrated tip, not the Phase A point
            self.tcp["tool_ee_cam"] = ee_cam[seq]
        counts = {name: int((state == code).sum()) for name, code in (("contact", 1), ("absent", 0), ("uncertain", -1))}
        self.tcp["track_point"] = track_point
        self.report.update(tcp_artifact=str(path), contact_active_frames=int(arrays["contact_active"].sum()),
                           phase_a_track_point=track_point,
                           contact_state_counts=counts, contact_source=self.tcp["contact_source"],
                           contact_point_calibrated=self.tcp["contact_point_calibrated"],
                           tracking_scale=self.tcp["scale"])
        if self.phase_c and not bool(arrays["contact_active"].any()):
            raise ValueError(
                f"Phase C enabled but the TCP artifact {path} has no confirmed contact frame "
                f"(contact source: {self.tcp['contact_source']}, states: {counts}). Export force-based contact "
                "annotations (samuele/export_force_contact_annotations.py) or disable Phase C.")

    def _load_seeds(self, args):
        path = _resolve_artifact(getattr(args, "moving_prompt_artifact", None), self.input_dir,
                                 f"metadata/moving_prompts/{SEED_ARTIFACT}")
        if path.is_dir():
            path = path / SEED_ARTIFACT
        if not path.is_file():
            raise FileNotFoundError(f"Phase B enabled but moving-part seed artifact is missing: {path}")
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        manifest_path = path.with_suffix(".json")
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        self._check_conversion(manifest, f"Seed artifact {path}")
        offsets = arrays["frame_offsets"]
        if len(offsets) != self.num_original + 1:
            raise ValueError(f"Seed artifact frame count {len(offsets) - 1} != tracking input {self.num_original}")
        K_full = arrays["K_fullres"]
        scale_x = self.K_track[0, 0] / K_full[0, 0]
        scale_y = self.K_track[1, 1] / K_full[1, 1]
        self.seeds = dict(points=arrays["points_xy"].astype(np.float64), offsets=offsets,
                          valid=arrays["valid"].astype(bool), scale=(scale_x, scale_y), path=str(path))
        self.report.update(seed_artifact=str(path), seed_frames=int(self.seeds["valid"].sum()),
                           seed_points=int(len(self.seeds["points"])))
        if not bool(self.seeds["valid"].any()) or len(self.seeds["points"]) == 0:
            raise ValueError(
                f"Phase B enabled but {path} contains no usable moving-part seed (no confirmed, visible contact "
                "frame produced a positive prompt). Refusing to report a successful no-op.")

    # ------------------------------------------------------------------ per-window access
    def window_indices(self, current_frame, length, duplicated_last):
        seq = list(range(current_frame, current_frame + length))
        usable = [bool(self.usable[i]) for i in seq]
        if duplicated_last:
            seq.append(seq[-1])
            usable.append(False)
        return np.asarray(seq, dtype=np.int64), np.asarray(usable, dtype=bool)

    def window_tcp(self, current_frame, length, duplicated_last):
        seq, usable = self.window_indices(current_frame, length, duplicated_last)
        return dict(uv=self.tcp["uv"][seq], z=self.tcp["z"][seq],
                    usable=usable & self.tcp["valid_projection"][seq],
                    source_frame_index=self.tcp["source_frame_index"][seq], sequence_index=seq)

    def window_contact(self, current_frame, length, duplicated_last):
        """Camera-frame contact points for each window frame and whether contact is usable."""
        seq, usable = self.window_indices(current_frame, length, duplicated_last)
        p = self.tcp["contact_p_cam"][seq]
        valid = usable & self.tcp["contact_active"][seq] & np.isfinite(p).all(axis=1) & (p[:, 2] > 0)
        return p, valid, seq

    def seed_map(self, sequence_index):
        """Boolean positive moving-object seed map at tracking resolution, or None."""
        if self.seeds is None or not self.usable[sequence_index]:
            return None
        frame = int(self.sequence_map["original"][sequence_index])
        if not self.seeds["valid"][frame]:
            return None
        start, end = self.seeds["offsets"][frame], self.seeds["offsets"][frame + 1]
        if end <= start:
            return None
        pts = self.seeds["points"][start:end]
        x = np.clip(np.rint(pts[:, 0] * self.seeds["scale"][0]).astype(np.int64), 0, self.width - 1)
        y = np.clip(np.rint(pts[:, 1] * self.seeds["scale"][1]).astype(np.int64), 0, self.height - 1)
        seed = np.zeros((self.height, self.width), dtype=bool)
        seed[y, x] = True
        return seed

    def write_report(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        flags = dict(tcp_supervision_enabled=self.phase_a, moving_prompt_enabled=self.phase_b,
                     contact_surface_enabled=self.phase_c)
        (Path(output_dir) / "tracking_priors.json").write_text(json.dumps(dict(flags=flags, **self.report), indent=2) + "\n")
