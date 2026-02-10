from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import networkx as nx
import numpy as np
import trimesh
from dotenv import load_dotenv
from shapely.geometry import Polygon
from skimage.morphology import skeletonize

load_dotenv()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def create_mask(
    path_to_json: str,
    path_to_img: str,
    output_dir: str,
    include_classes: Tuple[str, ...] = (
        "internal-walls",
        "external-walls",
    ),
    min_conf: float = 0.0,
    fill_value: int = 255,
) -> str:
    """Rasterize only wall classes from predictions into a binary mask."""

    json_path = Path(path_to_json)
    img_path = Path(path_to_img)
    out_dir = Path(output_dir)
    ensure_dir(out_dir)

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    h, w = img.shape[:2]

    try:
        data = json.loads(json_path.read_text())
    except Exception as exc:
        raise ValueError(f"Could not parse JSON: {json_path}") from exc

    mask = np.zeros((h, w), dtype=np.uint8)

    for det in data.get("predictions", []):
        cls = det.get("class")
        conf = float(det.get("confidence", 0.0))
        if cls not in include_classes or conf < min_conf:
            continue

        pts = det.get("points") or []
        if pts and len(pts) >= 3:
            poly = np.array([[p["x"], p["y"]] for p in pts], dtype=np.float32)
            poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
            poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
            poly_i = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [poly_i], color=int(fill_value))
            continue

        # Fallback: draw bounding box if polygon points are missing
        width = det.get("width")
        height = det.get("height")
        center_x = det.get("x")
        center_y = det.get("y")
        if None in (width, height, center_x, center_y):
            continue
        half_w = float(width) / 2.0
        half_h = float(height) / 2.0
        x1 = max(0.0, float(center_x) - half_w)
        y1 = max(0.0, float(center_y) - half_h)
        x2 = min(float(w - 1), float(center_x) + half_w)
        y2 = min(float(h - 1), float(center_y) + half_h)
        rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        rect = np.round(rect).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [rect], color=int(fill_value))

    out_path = out_dir / f"{img_path.stem}_wall.png"
    if not cv2.imwrite(str(out_path), mask):
        raise IOError(f"Failed to write mask to: {out_path}")
    return str(out_path)


def reconstruct_from_mask_and_predictions(
    mask_path: Path,
    predictions_path: Path,
    output_dir: Path,
    *,
    image_stem: Optional[str] = None,
    wall_height: float = 3.0,
    wall_thickness: float = 0.1,
    pixel_size: float = 0.01,
    wall_threshold: float = 0.5,
    hough_threshold: int = 45,
    min_line_length: int = 40,
    max_line_gap: int = 6,
    merge_distance: int = 4,
    snap_tolerance: float = 5.0,
    door_height: float = 2.1,
    window_height: float = 1.2,
    sill_height: float = 0.9,
    floor_thickness: float = 0.05,
    door_frame_thickness: float = 0.05,
    door_frame_depth: float = 0.25,
    window_frame_thickness: float = 0.04,
    window_frame_depth: float = 0.2,
    visualize_doors: bool = True,
    door_panel_depth: float = 0.15,
    mirror_gltf: bool = False,
    filter_min_wall_px: int = 7,
) -> Tuple[Path, Path]:
    CLASS_NAME_TO_ID = {"door": 1, "window": 2}
    GEOM_EPS = 1e-6
    BROWN_RGBA = np.array([150, 75, 0, 255], dtype=np.uint8)
    DARK_DOOR_RGBA = np.array([90, 45, 0, 255], dtype=np.uint8)

    @dataclass
    class AxisAlignedBox:
        x1: float
        y1: float
        z1: float
        x2: float
        y2: float
        z2: float

        def __post_init__(self) -> None:
            if self.x1 > self.x2:
                self.x1, self.x2 = self.x2, self.x1
            if self.y1 > self.y2:
                self.y1, self.y2 = self.y2, self.y1
            if self.z1 > self.z2:
                self.z1, self.z2 = self.z2, self.z1

        def extent(self) -> Tuple[float, float, float]:
            return self.x2 - self.x1, self.y2 - self.y1, self.z2 - self.z1

        def center(self) -> Tuple[float, float, float]:
            ex, ey, ez = self.extent()
            return self.x1 + ex / 2.0, self.y1 + ey / 2.0, self.z1 + ez / 2.0

        def intersects(self, other: "AxisAlignedBox", tol: float = GEOM_EPS) -> bool:
            return not (
                self.x2 <= other.x1 + tol
                or self.x1 >= other.x2 - tol
                or self.y2 <= other.y1 + tol
                or self.y1 >= other.y2 - tol
                or self.z2 <= other.z1 + tol
                or self.z1 >= other.z2 - tol
            )

        def within(self, other: "AxisAlignedBox", tol: float = GEOM_EPS) -> bool:
            return (
                self.x1 >= other.x1 - tol
                and self.x2 <= other.x2 + tol
                and self.y1 >= other.y1 - tol
                and self.y2 <= other.y2 + tol
                and self.z1 >= other.z1 - tol
                and self.z2 <= other.z2 + tol
            )

        def to_mesh(self, tol: float = GEOM_EPS) -> Optional[trimesh.Trimesh]:
            ex, ey, ez = self.extent()
            if ex <= tol or ey <= tol or ez <= tol:
                return None
            mesh = trimesh.creation.box(extents=[ex, ey, ez])
            mesh.apply_translation(self.center())
            return mesh

    @dataclass
    class OpeningCutout:
        box: AxisAlignedBox
        label: Optional[int]
        horizontal: Optional[bool] = None
        normal_center: Optional[float] = None

    @dataclass
    class FrameParams:
        thickness: float
        depth: float

        def enabled(self) -> bool:
            return self.thickness > GEOM_EPS

    def log_step(message: str) -> None:
        print(f"[step] {message}", flush=True)

    def normalize_label(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        label = name.strip().lower()
        if label in CLASS_NAME_TO_ID:
            return label
        if label.endswith("s") and label[:-1] in CLASS_NAME_TO_ID:
            return label[:-1]
        return label

    def apply_face_color(mesh: trimesh.Trimesh, rgba: np.ndarray) -> None:
        if len(mesh.faces) == 0:
            return
        color = np.asarray(rgba, dtype=np.uint8)
        mesh.visual.face_colors = np.tile(color, (len(mesh.faces), 1))

    def subtract_axis_aligned_box(
        box: AxisAlignedBox, cutout: AxisAlignedBox, tol: float = GEOM_EPS
    ) -> List[AxisAlignedBox]:
        if not box.intersects(cutout, tol):
            return [box]
        xs = sorted({box.x1, box.x2, cutout.x1, cutout.x2})
        ys = sorted({box.y1, box.y2, cutout.y1, cutout.y2})
        zs = sorted({box.z1, box.z2, cutout.z1, cutout.z2})
        residuals: List[AxisAlignedBox] = []
        for xi in range(len(xs) - 1):
            x_start, x_end = xs[xi], xs[xi + 1]
            if x_end - x_start <= tol:
                continue
            for yi in range(len(ys) - 1):
                y_start, y_end = ys[yi], ys[yi + 1]
                if y_end - y_start <= tol:
                    continue
                for zi in range(len(zs) - 1):
                    z_start, z_end = zs[zi], zs[zi + 1]
                    if z_end - z_start <= tol:
                        continue
                    candidate = AxisAlignedBox(
                        x_start, y_start, z_start, x_end, y_end, z_end
                    )
                    if candidate.within(cutout, tol):
                        continue
                    if candidate.within(box, tol):
                        residuals.append(candidate)
        return residuals

    def wall_segment_to_mesh(
        p1: np.ndarray,
        p2: np.ndarray,
        thickness: float,
        height: float,
        cutouts: Optional[Sequence[AxisAlignedBox]] = None,
    ) -> Optional[trimesh.Trimesh]:
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < GEOM_EPS:
            return None
        direction /= length
        normal = np.array([-direction[1], direction[0]])
        offset = normal * (thickness / 2)
        corners = [
            tuple(p1 + offset),
            tuple(p2 + offset),
            tuple(p2 - offset),
            tuple(p1 - offset),
        ]
        polygon = Polygon(corners)
        if not polygon.is_valid or polygon.area == 0:
            return None
        base_mesh = trimesh.creation.extrude_polygon(polygon, height)
        if not cutouts:
            return base_mesh
        horizontal = abs(direction[0]) >= abs(direction[1])
        if horizontal:
            x1 = min(p1[0], p2[0])
            x2 = max(p1[0], p2[0])
            center_y = (p1[1] + p2[1]) / 2.0
            y1 = center_y - thickness / 2.0
            y2 = center_y + thickness / 2.0
        else:
            y1 = min(p1[1], p2[1])
            y2 = max(p1[1], p2[1])
            center_x = (p1[0] + p2[0]) / 2.0
            x1 = center_x - thickness / 2.0
            x2 = center_x + thickness / 2.0
        base_box = AxisAlignedBox(x1, y1, 0.0, x2, y2, height)
        relevant = [cut for cut in cutouts if base_box.intersects(cut)]
        if not relevant:
            return base_mesh
        solids: List[AxisAlignedBox] = [base_box]
        for cut in relevant:
            updated: List[AxisAlignedBox] = []
            for solid in solids:
                updated.extend(subtract_axis_aligned_box(solid, cut))
            solids = updated
        meshes: List[trimesh.Trimesh] = []
        for solid in solids:
            mesh = solid.to_mesh()
            if mesh:
                meshes.append(mesh)
        if not meshes:
            return None
        return trimesh.util.concatenate(meshes)

    def graph_to_mesh(
        graph: nx.Graph,
        thickness: float,
        height: float,
        pixel_size_local: float,
        opening_cutouts: Optional[Sequence[AxisAlignedBox]] = None,
    ) -> Optional[trimesh.Trimesh]:
        meshes: List[trimesh.Trimesh] = []
        for u, v, _ in graph.edges(data=True):
            pixel_p1 = np.array(graph.nodes[u]["pos"], dtype=float)
            pixel_p2 = np.array(graph.nodes[v]["pos"], dtype=float)
            p1 = pixel_p1 * pixel_size_local
            p2 = pixel_p2 * pixel_size_local
            mesh = wall_segment_to_mesh(
                p1,
                p2,
                thickness,
                height,
                cutouts=opening_cutouts,
            )
            if mesh:
                meshes.append(mesh)
        if not meshes:
            return None
        return trimesh.util.concatenate(meshes)

    def build_floor_mesh(
        wall_mesh: trimesh.Trimesh, thickness: float
    ) -> trimesh.Trimesh:
        if thickness <= 0:
            raise ValueError("Floor thickness must be positive")
        min_corner, max_corner = wall_mesh.bounds
        size_x = max(1e-3, max_corner[0] - min_corner[0])
        size_y = max(1e-3, max_corner[1] - min_corner[1])
        floor = trimesh.creation.box(extents=[size_x, size_y, thickness])
        center_x = (min_corner[0] + max_corner[0]) / 2
        center_y = (min_corner[1] + max_corner[1]) / 2
        floor.apply_translation([center_x, center_y, thickness / 2])
        return floor

    def build_opening_cutouts(
        detections: List[Dict[str, float]],
        pixel_size_local: float,
        thickness_local: float,
        door_height_local: float,
        window_height_local: float,
        sill_height_local: float,
    ) -> List[OpeningCutout]:
        cutouts: List[OpeningCutout] = []
        min_span = thickness_local + 0.1
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            width = abs(x2 - x1) * pixel_size_local
            depth_extent = abs(y2 - y1) * pixel_size_local
            center_x = ((x1 + x2) / 2.0) * pixel_size_local
            center_y = ((y1 + y2) / 2.0) * pixel_size_local
            label = det.get("label")
            if label == CLASS_NAME_TO_ID["door"]:
                z1 = 0.0
                z2 = door_height_local
            else:
                z1 = sill_height_local
                z2 = sill_height_local + window_height_local
            z2 = max(z1 + GEOM_EPS, z2)
            span_x = max(width, min_span)
            span_y = max(depth_extent, min_span)
            cutouts.append(
                OpeningCutout(
                    AxisAlignedBox(
                        center_x - span_x / 2.0,
                        center_y - span_y / 2.0,
                        z1,
                        center_x + span_x / 2.0,
                        center_y + span_y / 2.0,
                        z2,
                    ),
                    det.get("label"),
                )
            )
        return cutouts

    def build_opening_frames(
        cutouts: Sequence[OpeningCutout],
        door_params: FrameParams,
        window_params: FrameParams,
        wall_thickness_local: float,
    ) -> Tuple[List[trimesh.Trimesh], float]:
        frames: List[trimesh.Trimesh] = []
        total_volume = 0.0
        for cut in cutouts:
            params = (
                door_params if cut.label == CLASS_NAME_TO_ID["door"] else window_params
            )
            if not params.enabled():
                continue
            base = cut.box
            span_x = base.x2 - base.x1
            span_y = base.y2 - base.y1
            if span_x <= GEOM_EPS or span_y <= GEOM_EPS:
                continue
            depth_offset = max(params.depth, 0.0) * wall_thickness_local
            horizontal = cut.horizontal
            if horizontal is None:
                diff_x = abs(span_x - wall_thickness_local)
                diff_y = abs(span_y - wall_thickness_local)
                if diff_y + GEOM_EPS < diff_x:
                    horizontal = True
                elif diff_x + GEOM_EPS < diff_y:
                    horizontal = False
                else:
                    horizontal = span_x >= span_y
            if horizontal:
                wall_min = base.x1
                wall_max = base.x2
                normal_center = (
                    cut.normal_center
                    if cut.normal_center is not None
                    else (base.y1 + base.y2) / 2.0
                )
                normal_half = min(span_y, wall_thickness_local) / 2.0
                base_normal_min = normal_center - normal_half
                base_normal_max = normal_center + normal_half
                outer_x1 = wall_min - params.thickness
                outer_x2 = wall_max + params.thickness
                outer_y1 = base_normal_min - depth_offset
                outer_y2 = base_normal_max + depth_offset
                inner_x1 = wall_min
                inner_x2 = wall_max
                inner_y1 = base_normal_min
                inner_y2 = base_normal_max
            else:
                wall_min = base.y1
                wall_max = base.y2
                normal_center = (
                    cut.normal_center
                    if cut.normal_center is not None
                    else (base.x1 + base.x2) / 2.0
                )
                normal_half = min(span_x, wall_thickness_local) / 2.0
                base_normal_min = normal_center - normal_half
                base_normal_max = normal_center + normal_half
                outer_y1 = wall_min - params.thickness
                outer_y2 = wall_max + params.thickness
                outer_x1 = base_normal_min - depth_offset
                outer_x2 = base_normal_max + depth_offset
                inner_y1 = wall_min
                inner_y2 = wall_max
                inner_x1 = base_normal_min
                inner_x2 = base_normal_max
            z1 = base.z1
            z2 = base.z2
            components: List[AxisAlignedBox] = []
            if horizontal:
                components.append(
                    AxisAlignedBox(outer_x1, outer_y1, z1, inner_x1, outer_y2, z2)
                )
                components.append(
                    AxisAlignedBox(inner_x2, outer_y1, z1, outer_x2, outer_y2, z2)
                )
            else:
                components.append(
                    AxisAlignedBox(outer_x1, outer_y1, z1, outer_x2, inner_y1, z2)
                )
                components.append(
                    AxisAlignedBox(outer_x1, inner_y2, z1, outer_x2, outer_y2, z2)
                )
            head_height = min(params.thickness, z2 - z1)
            components.append(
                AxisAlignedBox(
                    outer_x1, outer_y1, z2 - head_height, outer_x2, outer_y2, z2
                )
            )
            if cut.label != CLASS_NAME_TO_ID["door"]:
                sill_height_local = min(params.thickness, z2 - z1)
                components.append(
                    AxisAlignedBox(
                        outer_x1,
                        outer_y1,
                        z1,
                        outer_x2,
                        outer_y2,
                        z1 + sill_height_local,
                    )
                )
            for component in components:
                dx = max(component.x2 - component.x1, 0.0)
                dy = max(component.y2 - component.y1, 0.0)
                dz = max(component.z2 - component.z1, 0.0)
                volume = dx * dy * dz
                if volume <= GEOM_EPS:
                    continue
                mesh = component.to_mesh()
                if mesh:
                    apply_face_color(mesh, BROWN_RGBA)
                    frames.append(mesh)
                    total_volume += volume
        return frames, total_volume

    def build_door_panels(
        cutouts: Sequence[OpeningCutout],
        wall_thickness_local: float,
        depth_fraction: float,
    ) -> List[trimesh.Trimesh]:
        panels: List[trimesh.Trimesh] = []
        depth = max(depth_fraction, 0.0) * wall_thickness_local
        min_depth = max(wall_thickness_local * 0.02, GEOM_EPS)
        for cut in cutouts:
            if cut.label != CLASS_NAME_TO_ID["door"]:
                continue
            base = cut.box
            horizontal = cut.horizontal if cut.horizontal is not None else True
            normal_center = cut.normal_center
            if horizontal:
                center = (
                    normal_center
                    if normal_center is not None
                    else (base.y1 + base.y2) / 2.0
                )
                half = max(depth / 2.0, min_depth / 2.0)
                y1 = center - half
                y2 = center + half
                panel_box = AxisAlignedBox(base.x1, y1, base.z1, base.x2, y2, base.z2)
            else:
                center = (
                    normal_center
                    if normal_center is not None
                    else (base.x1 + base.x2) / 2.0
                )
                half = max(depth / 2.0, min_depth / 2.0)
                x1 = center - half
                x2 = center + half
                panel_box = AxisAlignedBox(x1, base.y1, base.z1, x2, base.y2, base.z2)
            mesh = panel_box.to_mesh()
            if mesh:
                apply_face_color(mesh, DARK_DOOR_RGBA)
                panels.append(mesh)
        return panels

    def generate_skeleton(
        prob_map: np.ndarray, threshold: float, min_thickness_px: int
    ) -> np.ndarray:
        binary = (prob_map > threshold).astype(np.uint8)
        if min_thickness_px > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (min_thickness_px, min_thickness_px)
            )
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        skeleton = skeletonize(binary).astype(np.uint8) * 255
        return skeleton

    def detect_lines(
        skeleton_image: np.ndarray,
        threshold_local: int,
        min_line_length_local: int,
        max_line_gap_local: int,
    ) -> List[Tuple[int, int, int, int]]:
        lines = cv2.HoughLinesP(
            skeleton_image,
            rho=1,
            theta=np.pi / 180,
            threshold=threshold_local,
            minLineLength=min_line_length_local,
            maxLineGap=max_line_gap_local,
        )
        if lines is None:
            return []
        return [tuple(map(int, line[0])) for line in lines]

    def snap_and_merge_lines(
        lines: List[Tuple[int, int, int, int]], merge_distance_local: int
    ) -> List[Tuple[int, int, int, int]]:
        snapped: List[Tuple[int, int, int, int]] = []
        for x1, y1, x2, y2 in lines:
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) >= abs(dy):
                y = int(round((y1 + y2) / 2))
                snapped.append((min(x1, x2), y, max(x1, x2), y))
            else:
                x = int(round((x1 + x2) / 2))
                snapped.append((x, min(y1, y2), x, max(y1, y2)))
        horizontals = sorted(
            [s for s in snapped if s[1] == s[3]], key=lambda seg: (seg[1], seg[0])
        )
        verticals = sorted(
            [s for s in snapped if s[0] == s[2]], key=lambda seg: (seg[0], seg[1])
        )
        merged: List[Tuple[int, int, int, int]] = []
        for group, horizontal in [(horizontals, True), (verticals, False)]:
            current: Optional[List[int]] = None
            for seg in group:
                if current is None:
                    current = list(seg)
                    continue
                if horizontal:
                    same_line = abs(seg[1] - current[1]) <= merge_distance_local
                    overlap = seg[0] <= current[2] + merge_distance_local
                    if same_line and overlap:
                        current[2] = max(current[2], seg[2])
                        current[0] = min(current[0], seg[0])
                        current[1] = current[3] = int(round((current[1] + seg[1]) / 2))
                    else:
                        merged.append(tuple(current))
                        current = list(seg)
                else:
                    same_line = abs(seg[0] - current[0]) <= merge_distance_local
                    overlap = seg[1] <= current[3] + merge_distance_local
                    if same_line and overlap:
                        current[3] = max(current[3], seg[3])
                        current[1] = min(current[1], seg[1])
                        current[0] = current[2] = int(round((current[0] + seg[0]) / 2))
                    else:
                        merged.append(tuple(current))
                        current = list(seg)
            if current is not None:
                merged.append(tuple(current))
        return merged

    def build_wall_graph(
        segments: List[Tuple[int, int, int, int]], snap_tolerance_local: float
    ) -> nx.Graph:
        graph = nx.Graph()
        points: List[np.ndarray] = []

        def snap_point(point: np.ndarray) -> Tuple[int, np.ndarray]:
            for idx, existing in enumerate(points):
                if np.linalg.norm(existing - point) <= snap_tolerance_local:
                    return idx, existing
            points.append(point)
            return len(points) - 1, point

        for seg in segments:
            p1 = np.array([seg[0], seg[1]], dtype=float)
            p2 = np.array([seg[2], seg[3]], dtype=float)
            if np.linalg.norm(p2 - p1) < GEOM_EPS:
                continue
            id1, pt1 = snap_point(p1)
            id2, pt2 = snap_point(p2)
            graph.add_node(id1, pos=pt1)
            graph.add_node(id2, pos=pt2)
            length = float(np.linalg.norm(pt2 - pt1))
            graph.add_edge(id1, id2, length=length)
        return graph

    def wall_segments_world(
        graph: nx.Graph, pixel_size_local: float
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        segments: List[Tuple[np.ndarray, np.ndarray]] = []
        for u, v in graph.edges():
            p1 = np.array(graph.nodes[u]["pos"], dtype=float) * pixel_size_local
            p2 = np.array(graph.nodes[v]["pos"], dtype=float) * pixel_size_local
            segments.append((p1, p2))
        return segments

    def annotate_cutout_orientations(
        cutouts: Sequence[OpeningCutout],
        segments_world: Sequence[Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        if not cutouts or not segments_world:
            return

        def dist_sq(point: np.ndarray, seg: Tuple[np.ndarray, np.ndarray]) -> float:
            a, b = seg
            ab = b - a
            denom = float(np.dot(ab, ab))
            if denom <= GEOM_EPS:
                return float(np.sum((point - a) ** 2))
            t = float(np.dot(point - a, ab) / denom)
            t = max(0.0, min(1.0, t))
            proj = a + t * ab
            return float(np.sum((point - proj) ** 2))

        for cut in cutouts:
            center = np.array(
                [
                    (cut.box.x1 + cut.box.x2) / 2.0,
                    (cut.box.y1 + cut.box.y2) / 2.0,
                ],
                dtype=float,
            )
            best_dist: Optional[float] = None
            best_horizontal: Optional[bool] = None
            best_center: Optional[float] = None
            for seg in segments_world:
                distance = dist_sq(center, seg)
                if best_dist is None or distance < best_dist:
                    best_dist = distance
                    p1, p2 = seg
                    if abs(p2[0] - p1[0]) >= abs(p2[1] - p1[1]):
                        best_horizontal = True
                        best_center = (p1[1] + p2[1]) / 2.0
                    else:
                        best_horizontal = False
                        best_center = (p1[0] + p2[0]) / 2.0
            cut.horizontal = best_horizontal
            cut.normal_center = best_center

    def parse_opening_detections(pred_path: Path) -> List[Dict[str, float]]:
        data = json.loads(pred_path.read_text())
        detections: List[Dict[str, float]] = []
        for pred in data.get("predictions", []):
            label_raw = pred.get("class") or pred.get("label")
            normalized = normalize_label(label_raw)
            label_id = CLASS_NAME_TO_ID.get(normalized) if normalized else None
            if label_id not in CLASS_NAME_TO_ID.values():
                continue
            bbox: Optional[Tuple[float, float, float, float]] = None
            width = pred.get("width")
            height = pred.get("height")
            center_x = pred.get("x")
            center_y = pred.get("y")
            if None not in (width, height, center_x, center_y):
                w = float(width)
                h = float(height)
                cx = float(center_x)
                cy = float(center_y)
                bbox = (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
            elif pred.get("points"):
                xs = [float(pt.get("x", 0.0)) for pt in pred["points"]]
                ys = [float(pt.get("y", 0.0)) for pt in pred["points"]]
                if xs and ys:
                    bbox = (min(xs), min(ys), max(xs), max(ys))
            if bbox is None:
                continue
            detections.append(
                {
                    "bbox": list(bbox),
                    "score": float(pred.get("confidence", 0.0)),
                    "label": label_id,
                    "label_name": normalized,
                }
            )
        return detections

    ensure_dir(output_dir)
    if image_stem is None:
        image_stem = Path(mask_path).stem
    target_dir = output_dir / image_stem
    ensure_dir(target_dir)

    log_step(f"Loading binary mask from {mask_path}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    log_step(f"Mask resolution: {mask.shape[1]}x{mask.shape[0]} pixels")
    prob_map = (mask.astype(np.float32) / 255.0).clip(0.0, 1.0)

    min_wall_px = max(1, int(round(wall_thickness / max(pixel_size, GEOM_EPS))))
    if filter_min_wall_px and filter_min_wall_px > 0:
        min_wall_px = max(1, int(filter_min_wall_px))
    if min_wall_px % 2 == 0:
        min_wall_px += 1
    log_step(
        f"Generating skeleton from mask (threshold={wall_threshold}, min_px={min_wall_px})"
    )
    skeleton = generate_skeleton(prob_map, wall_threshold, min_wall_px)
    log_step("Detecting lines via Hough transform")
    lines = detect_lines(skeleton, hough_threshold, min_line_length, max_line_gap)
    log_step(f"Detected {len(lines)} raw line segments")
    segments = snap_and_merge_lines(lines, merge_distance)
    log_step(f"Merged segments down to {len(segments)} snapped walls")
    wall_graph = build_wall_graph(segments, snap_tolerance)
    log_step(
        f"Wall graph has {wall_graph.number_of_nodes()} nodes and {wall_graph.number_of_edges()} edges"
    )

    log_step(f"Parsing opening detections from {predictions_path}")
    detections = parse_opening_detections(predictions_path)
    opening_cutouts = build_opening_cutouts(
        detections,
        pixel_size,
        wall_thickness,
        door_height,
        window_height,
        sill_height,
    )
    wall_segments = wall_segments_world(wall_graph, pixel_size)
    annotate_cutout_orientations(opening_cutouts, wall_segments)

    log_step("Converting wall graph to mesh")
    wall_mesh = graph_to_mesh(
        wall_graph,
        wall_thickness,
        wall_height,
        pixel_size,
        opening_cutouts=[cut.box for cut in opening_cutouts],
    )
    if wall_mesh is None:
        raise RuntimeError("Unable to build wall mesh; mask may be empty")

    door_params = FrameParams(door_frame_thickness, door_frame_depth)
    window_params = FrameParams(window_frame_thickness, window_frame_depth)
    frame_meshes, frame_volume = build_opening_frames(
        opening_cutouts, door_params, window_params, wall_thickness
    )
    if frame_meshes:
        log_step(
            f"Adding {len(frame_meshes)} frame meshes (~{frame_volume:.4f} volume units)"
        )
        wall_mesh = trimesh.util.concatenate([wall_mesh, *frame_meshes])

    if visualize_doors:
        door_panels = build_door_panels(
            opening_cutouts, wall_thickness, door_panel_depth
        )
        if door_panels:
            log_step(f"Adding {len(door_panels)} door panel meshes")
            wall_mesh = trimesh.util.concatenate([wall_mesh, *door_panels])

    if floor_thickness > 0:
        # Mirror reconstruct() behavior from main.py by adding an optional floor slab.
        log_step(f"Adding floor slab with thickness {floor_thickness}")
        floor_mesh = build_floor_mesh(wall_mesh, floor_thickness)
        wall_mesh = trimesh.util.concatenate([wall_mesh, floor_mesh])

    obj_path = target_dir / f"{image_stem}.obj"
    glb_path = target_dir / f"{image_stem}.glb"
    log_step(f"Exporting OBJ to {obj_path}")
    wall_mesh.export(obj_path, include_color=True)
    mtl_path = obj_path.with_suffix(".mtl")
    if not mtl_path.exists():
        log_step("[warn] OBJ exporter did not emit an MTL file; colors may be missing")

    log_step(f"Exporting glTF to {glb_path}")
    glb_mesh = wall_mesh
    if mirror_gltf:
        log_step("Mirroring glTF mesh across Y axis")
        glb_mesh = wall_mesh.copy()
        mirror_transform = np.eye(4)
        mirror_transform[1, 1] = -1.0
        glb_mesh.apply_transform(mirror_transform)
    glb_mesh.export(glb_path)
    log_step(
        f"Finished exports: OBJ={obj_path.name}, MTL={mtl_path.name if mtl_path.exists() else 'n/a'}, glTF={glb_path.name}"
    )

    return obj_path, glb_path


def fetch_roboflow_predictions(
    image_path: Path,
    api_key: str,
    model_id: str,
    api_url: str = "https://serverless.roboflow.com",
) -> Dict:
    if not api_key:
        raise ValueError("Roboflow API key is required")
    if not model_id:
        raise ValueError("Roboflow model ID is required")
    try:
        from inference_sdk import InferenceHTTPClient
    except ImportError as exc:
        raise RuntimeError(
            "inference-sdk is missing. Install requirements or add inference-sdk to your environment."
        ) from exc

    print(f"[step] Calling Roboflow model '{model_id}' via {api_url}")
    client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
    result = client.infer(str(image_path), model_id=model_id)
    predictions = result.get("predictions") or result.get("preds") or []
    result["predictions"] = predictions
    print(f"[step] Roboflow returned {len(predictions)} predictions")
    return result


def reconstruct_from_image_with_roboflow(
    image_path: Path,
    output_dir: Path,
    *,
    roboflow_api_key: str,
    roboflow_model_id: str,
    roboflow_api_url: str = "https://serverless.roboflow.com",
    **reconstruct_kwargs,
) -> Tuple[Path, Path]:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    image_stem = image_path.stem
    target_dir = output_dir / image_stem
    ensure_dir(target_dir)

    if not roboflow_api_key:
        raise ValueError("ROBOFLOW_API_KEY is not set")

    predictions = fetch_roboflow_predictions(
        image_path, roboflow_api_key, roboflow_model_id, api_url=roboflow_api_url
    )
    predictions_path = target_dir / "predictions.json"
    predictions_path.write_text(json.dumps(predictions, indent=2))

    mask_path = Path(
        create_mask(
            path_to_json=str(predictions_path),
            path_to_img=str(image_path),
            output_dir=str(target_dir),
        )
    )

    return reconstruct_from_mask_and_predictions(
        mask_path=mask_path,
        predictions_path=predictions_path,
        output_dir=output_dir,
        image_stem=image_stem,
        **reconstruct_kwargs,
    )


if __name__ == "__main__":
    reconstruct_from_image_with_roboflow(
        Path("datasets/images/test.png"),
        Path("output"),
        roboflow_api_key=os.getenv("ROBOFLOW_API_KEY") or "",
        roboflow_model_id="floor-plan-ts7cp/31",
    )
