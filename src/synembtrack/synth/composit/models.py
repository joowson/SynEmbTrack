from __future__ import annotations
import os
import random
import math
from glob import glob
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageOps, ImageFont

# ---------- helpers ----------
def resolve_patch_paths(patch_dir: str, limit: Optional[int] = 500):
    bodies = sorted(glob(os.path.join(patch_dir, 'patch_body_*')))
    if limit:
        bodies = bodies[:limit]
    bdrys = [b.replace('body', 'bdry') for b in bodies]
    pairs = [(b, d) for b, d in zip(bodies, bdrys) if os.path.exists(d)]
    return [p[0] for p in pairs], [p[1] for p in pairs]

class Particle:
    def __init__(self, x: float, y: float, v: float, phi: float, shape: Tuple[Image.Image, Image.Image]):
        self.x = x; self.y = y; self.v = v; self.phi = phi
        self.shape = shape  # (body RGBA, bdry RGBA)
        self.area: int = 0
        self.index: Optional[int] = None
        self.color = random.sample(list(ImageColor.colormap), 1)[0]

class ParticleSimulator:
    def __init__(self, dt: float, patch_dir: str, pxum: float, box_size: Tuple[int, int] = (400, 400),
                 patch_limit: Optional[int] = 500, draw_scalebar: bool=False):
        self.dt = dt; self.pxum = pxum
        self.size_w, self.size_h = box_size
        self.margin = 12
        self.draw_index = False
        self.draw_scalebar = draw_scalebar

        bodies, bdrys = resolve_patch_paths(patch_dir, patch_limit)
        self.part_body: List[Image.Image] = []
        self.part_bdry: List[Image.Image] = []
        for b, d in zip(bodies, bdrys):
            try:
                self.part_body.append(Image.open(b))
                self.part_bdry.append(Image.open(d))
            except Exception:
                pass
        self.n_patch = len(self.part_body)
        print(f"# of patches found: {len(bodies)} | usable pairs: {self.n_patch}")

        self.pindex_counter = 0
        self.list_particle: List[Particle] = []

    def create_particle(self, x: float, y: float, v: float, phi: Optional[float] = None) -> None:
        if self.n_patch == 0:
            raise RuntimeError("No usable patch pairs found in patch_dir")
        k = random.randrange(self.n_patch)
        p_body = self.part_body[k]; p_bdry = self.part_bdry[k]
        phi0 = 2 * np.pi * random.random() if phi is None else phi
        p = Particle(x, y, v, phi0, (p_body, p_bdry))
        if self.is_in_box(p):
            p.index = self.pindex_counter; self.pindex_counter += 1
        self.list_particle.append(p)

    def periodic_pos(self, x: float, y: float):
        if x < -self.margin:
            x += self.size_w + 2 * self.margin
        elif x >= self.size_w + self.margin:
            x -= self.size_w + 2 * self.margin
        if y < -self.margin:
            y += self.size_h + 2 * self.margin
        elif y >= self.size_h + self.margin:
            y -= self.size_h + 2 * self.margin
        return x, y

    def is_in_box(self, p: Particle, margin: int = -5) -> bool:
        return not (p.x < -margin or p.x >= self.size_w + margin or p.y < -margin or p.y >= self.size_h + margin)

    def draw_screen(self, tstep: int, bg_path: str):
        try:
            image = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
            image = (image / np.median(image) * 142).astype(np.uint8)
            img = Image.fromarray(image)
        except Exception as e:
            raise RuntimeError(f"Failed to read background from: {bg_path} ({e})")

        org_w, org_h = img.size
        if org_w < self.size_w or org_h < self.size_h:
            raise ValueError("Background image smaller than box size.")
        ox = random.randrange(org_w - self.size_w); oy = random.randrange(org_h - self.size_h)
        img = img.crop((ox, oy, ox + self.size_w, oy + self.size_h)).convert('RGB')

        gt_base = Image.new('RGB', (self.size_w, self.size_h), (0, 0, 0))
        gt_list, id_list = [], []
        for p in self.list_particle:
            body = p.shape[0].rotate(-p.phi / np.pi * 180, Image.NEAREST, expand=True).convert("RGBA")
            bdry = p.shape[1].rotate(-p.phi / np.pi * 180, Image.NEAREST, expand=True).convert("RGBA")

            gt_body = body.copy(); px = gt_body.load(); area = 0
            for i in range(gt_body.size[0]):
                for j in range(gt_body.size[1]):
                    if px[i, j][3] > 0:
                        px[i, j] = (1, 1, 1); area += 1
                    else:
                        px[i, j] = (0, 0, 0)
            p.area = area
            if p.index is None or area < 5:
                continue

            img.paste(body, (int(p.x - body.size[0] / 2), int(p.y - body.size[1] / 2)), body)
            img.paste(bdry, (int(p.x - bdry.size[0] / 2), int(p.y - bdry.size[1] / 2)), bdry)

            instance_gt = gt_base.copy()
            instance_gt.paste(gt_body, (int(p.x - gt_body.size[0] / 2), int(p.y - gt_body.size[1] / 2)), gt_body)
            gray = ImageOps.grayscale(instance_gt)

            g = np.array(gray)
            se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, se1)
            g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, se1)

            if np.sum(g) != 0:
                from PIL import Image as _Image
                gt_list.append(_Image.fromarray(g))
                id_list.append(p.index)

        if self.draw_scalebar: self._draw_scalebar(img, bar_length_um=20, bar_thick_um=2)
        return img, gt_list, id_list

    def _draw_scalebar(self, img: Image.Image, bar_length_um: float, bar_thick_um: float) -> None:
        bar_l_px = int(bar_length_um * self.pxum)
        bar_thick_px = int(bar_thick_um * self.pxum)
        draw = ImageDraw.Draw(img)
        bar_color = (255, 255, 255)
        text = f"{int(bar_length_um)} μm"
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        try:
            font = ImageFont.truetype(font_path, 12)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        bar_pos = (20, img.size[1] - 20 - bar_thick_px)
        cx = bar_pos[0] + bar_l_px // 2
        text_pos = (cx - text_w // 2, bar_pos[1] - text_h - 10)
        draw.rectangle([bar_pos, (bar_pos[0] + bar_l_px, bar_pos[1] + bar_thick_px)], fill=bar_color)
        draw.text(text_pos, text, font=font, fill=bar_color)

class RotationalDiffusion(ParticleSimulator):
    def __init__(self, dt: float, D_T: float, D_R: float, patch_dir: str, pxum: float,
                 box_size: Tuple[int, int] = (400, 400), omg: float = 0.0, patch_limit: Optional[int] = 500, draw_scalebar: bool = False):
        super().__init__(dt, patch_dir, pxum, box_size, patch_limit, draw_scalebar=draw_scalebar)
        self.D_T = D_T; self.D_R = D_R; self.omg = omg

    def step(self) -> None:
        for p in self.list_particle:
            p.x = p.x + p.v * math.cos(p.phi) * self.dt + random.gauss(0, 1) * math.sqrt(2 * self.D_T * self.dt)
            p.y = p.y + p.v * math.sin(p.phi) * self.dt + random.gauss(0, 1) * math.sqrt(2 * self.D_T * self.dt)
            p.phi = p.phi + self.omg * self.dt + random.gauss(0, 1) * math.sqrt(2 * self.D_R * self.dt)
            p.x, p.y = self.periodic_pos(p.x, p.y)
            if self.is_in_box(p):
                if p.index is None:
                    p.index = self.pindex_counter; self.pindex_counter += 1
            elif p.index is not None:
                p.index = None
