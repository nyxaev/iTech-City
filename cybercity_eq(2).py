"""
CyberCity EQ  //  NOSTALGIA.EXE
Edição Windows XP + dreamcore + fita VHS

Comandos:
  M / TAB   abre/fecha o menu da biblioteca  (ou clique no botão "iniciar")
  space     pausa
  N / P     proxima / anterior
  C         troca a paleta na mao (bliss <-> dream)
  F         fullscreen
  ESC       fecha o menu (ou sai, se o menu estiver fechado)

No menu:
  A / O     adicionar musicas      DEL  remove da lista
  ENTER     toca a selecionada     setas / scroll navegam
  Voce tambem pode arrastar arquivos ou pastas para a janela.

A biblioteca e salva automaticamente em cybercity_library.json (ao lado do
script) e recarregada na proxima vez que o player abrir.

ATENCAO: nos drops a tela pisca, inverte as cores e mostra uma "tela azul".
Se isso incomodar, ponha FLASHES = False (ou BLACKOUTS = False) abaixo.

Uso: python cybercity_eq_xp.py [arquivo1 arquivo2 ...]
"""
import json
import math
import os
import random
import sys
import time

import numpy as np
import pygame
import librosa


W, H = 1280, 720
FPS = 60
BANDS = 48
SR = 22050
N_FFT = 2048
HOP = 512
HORIZON = int(H * 0.64)
TASKBAR_H = 30

AUDIO_EXTS = (".mp3", ".wav", ".ogg", ".flac")
LIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cybercity_library.json")

# --- ajustes -----------------------------------------------------------------
SYNC_OFFSET = 0.0      # segundos. Se o visual ficar atrasado/adiantado em relacao ao som, mexa aqui (ex: -0.05)
BLACKOUTS = True       # False = sem "tela azul" nos drops
FLASHES = True         # False = a tela azul continua, mas sem flash/inversao de cor
DREAM_CENTROID = 2200  # Hz: faixas mais "agudas/brilhantes" que isso ganham a paleta dream (rosa/lilas); as graves ficam no bliss (azul/verde)
VHS_YEAR = 2004        # ano que aparece no carimbo de data da fita
VHS_AMOUNT = 1.0       # 0.0 = sem efeito de VHS, 1.0 = normal, 2.0 = fita bem gasta

# --- paletas -------------------------------------------------------------------
PALETTES = {
    # bliss: ceu azul + gramado verde + sol amarelo (aquele papel de parede)
    "bliss": {"main": (44, 112, 232), "hot": (92, 184, 52), "alt": (255, 214, 64), "glow": (255, 255, 255)},
    # dream: tarde rosa e lilas, meio sonho
    "dream": {"main": (255, 150, 212), "hot": (180, 150, 255), "alt": (140, 226, 255), "glow": (255, 248, 255)},
}
ORDER = ["bliss", "dream"]
KEYS = ("main", "hot", "alt", "glow")

# cores do "Luna" (tema do XP)
XP_BEIGE = (236, 233, 216)
XP_SEL = (49, 106, 197)
LUNA_ACTIVE = [(0.0, (0, 88, 238)), (0.07, (62, 150, 255)), (0.18, (8, 100, 242)),
               (0.55, (0, 82, 226)), (0.9, (0, 66, 200)), (1.0, (0, 52, 170))]

MONTHS = ("JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL", "AGO", "SET", "OUT", "NOV", "DEZ")

# padroes de "tela azul": (duracao em segundos, modo). modos: black (=tela azul) | flash | invert | normal
DROP_PATTERNS = (
    ((0.18, "black"), (0.05, "flash"), (0.08, "black"), (0.05, "invert"), (0.06, "black"), (0.04, "flash")),
    ((0.42, "black"), (0.06, "flash")),
    ((0.07, "black"), (0.04, "normal"), (0.07, "black"), (0.04, "normal"), (0.15, "black"), (0.05, "invert")),
)
MICRO_PATTERNS = (
    ((0.035, "black"),),
    ((0.03, "black"), (0.03, "normal"), (0.04, "black")),
)


def lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def scale(c, k):
    return tuple(max(0, min(255, int(v * k))) for v in c)


def fit(font, text, maxw):
    """Corta o texto com reticencias para caber em maxw pixels."""
    if font.size(text)[0] <= maxw:
        return text
    while text and font.size(text + "...")[0] > maxw:
        text = text[:-1]
    return text + "..."


def sample(arr, pos):
    """Interpola entre os quadros da analise (e o que deixa o movimento fluido)."""
    n = arr.shape[-1]
    if n < 2:
        return arr[..., 0]
    pos = max(0.0, min(n - 1.001, pos))
    i = int(pos)
    f = pos - i
    return arr[..., i] * (1.0 - f) + arr[..., i + 1] * f


# ----------------------------------------------------------------------------
# Pecinhas de interface do Windows XP (tema Luna)
# ----------------------------------------------------------------------------
_FONTS = {}
_CACHE = {}


def xpfont(size, bold=False, italic=False, kind="ui"):
    key = (size, bold, italic, kind)
    if key not in _FONTS:
        names = {
            "ui": "tahoma,verdana,dejavusans,arial",
            "title": "trebuchetms,tahoma,verdana,dejavusans,arial",
            "mono": "lucidaconsole,couriernew,consolas,dejavusansmono,monospace",
        }[kind]
        _FONTS[key] = pygame.font.SysFont(names, size, bold=bold, italic=italic)
    return _FONTS[key]


def vgrad(w, h, stops):
    """Superficie com gradiente vertical. stops = [(posicao 0..1, cor), ...]"""
    surf = pygame.Surface((w, h))
    for y in range(h):
        t = y / max(1, h - 1)
        col = stops[-1][1]
        for i in range(len(stops) - 1):
            p0, c0 = stops[i]
            p1, c1 = stops[i + 1]
            if t <= p1:
                col = lerp(c0, c1, (t - p0) / max(1e-6, p1 - p0))
                break
        pygame.draw.line(surf, col, (0, y), (w, y))
    return surf


def title_bar(w, h=28):
    """Barra de titulo azul do XP, com os cantos de cima arredondados."""
    key = ("tb", w, h)
    if key not in _CACHE:
        out = vgrad(w, h, LUNA_ACTIVE).convert_alpha()
        mask = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, w, h),
                         border_top_left_radius=8, border_top_right_radius=8)
        out.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        _CACHE[key] = out
    return _CACHE[key]


def caption_buttons(surf, right, y):
    """Botoes de minimizar / maximizar / fechar."""
    sz = 21
    for i, kind in enumerate(("close", "max", "min")):
        r = pygame.Rect(right - (i + 1) * sz - i * 2, y, sz, sz)
        pygame.draw.rect(surf, (214, 70, 38) if kind == "close" else (38, 102, 238), r, border_radius=4)
        pygame.draw.rect(surf, (255, 255, 255), r, 1, border_radius=4)
        cx, cy = r.center
        if kind == "close":
            pygame.draw.line(surf, (255, 255, 255), (cx - 4, cy - 4), (cx + 4, cy + 4), 2)
            pygame.draw.line(surf, (255, 255, 255), (cx - 4, cy + 4), (cx + 4, cy - 4), 2)
        elif kind == "max":
            pygame.draw.rect(surf, (255, 255, 255), (cx - 5, cy - 5, 10, 10), 2)
        else:
            pygame.draw.rect(surf, (255, 255, 255), (cx - 5, cy + 2, 7, 3))


def draw_window(surf, rect, title, tb_h=28):
    """Janela do XP: moldura azul, barra de titulo, corpo bege."""
    x, y, w, h = rect
    frame = pygame.Rect(x, y, w, h)
    pygame.draw.rect(surf, (0, 60, 215), frame, border_top_left_radius=9, border_top_right_radius=9)
    surf.blit(title_bar(w, tb_h), (x, y))
    pygame.draw.rect(surf, XP_BEIGE, (x + 4, y + tb_h, w - 8, h - tb_h - 4))
    pygame.draw.rect(surf, (0, 0, 120), frame, 1, border_top_left_radius=9, border_top_right_radius=9)
    f = xpfont(14, bold=True, kind="title")
    txt = fit(f, title, w - 110)
    surf.blit(f.render(txt, True, (10, 20, 100)), (x + 10, y + 7))
    surf.blit(f.render(txt, True, (255, 255, 255)), (x + 9, y + 6))
    caption_buttons(surf, x + w - 6, y + 4)


def draw_button(surf, rect, label, hover=False, default=False):
    r = pygame.Rect(rect)
    key = ("btn", r.w - 2, r.h - 2)
    if key not in _CACHE:
        _CACHE[key] = vgrad(r.w - 2, r.h - 2, [(0, (255, 255, 255)), (0.8, (236, 235, 230)), (1, (214, 208, 197))])
    pygame.draw.rect(surf, (0, 60, 116), r, border_radius=3)
    inner = r.inflate(-2, -2)
    surf.blit(_CACHE[key], inner)
    if hover:
        pygame.draw.rect(surf, (255, 200, 70), inner.inflate(-2, -2), 2, border_radius=2)
    elif default:
        pygame.draw.rect(surf, (150, 185, 250), inner.inflate(-2, -2), 2, border_radius=2)
    txt = xpfont(13).render(label, True, (0, 0, 0))
    surf.blit(txt, txt.get_rect(center=r.center))


def draw_field(surf, rect):
    """Campo afundado da barra de status."""
    r = pygame.Rect(rect)
    pygame.draw.rect(surf, XP_BEIGE, r)
    pygame.draw.line(surf, (172, 168, 153), r.topleft, (r.right - 1, r.top))
    pygame.draw.line(surf, (172, 168, 153), r.topleft, (r.left, r.bottom - 1))
    pygame.draw.line(surf, (255, 255, 255), (r.left, r.bottom - 1), (r.right - 1, r.bottom - 1))
    pygame.draw.line(surf, (255, 255, 255), (r.right - 1, r.top), (r.right - 1, r.bottom - 1))


def draw_progress(surf, rect, frac):
    """Barra de progresso do XP: aqueles quadradinhos verdes."""
    r = pygame.Rect(rect)
    pygame.draw.rect(surf, (255, 255, 255), r, border_radius=3)
    pygame.draw.rect(surf, (130, 135, 144), r, 1, border_radius=3)
    bw, gap = 9, 2
    ih = max(2, r.h - 6)
    key = ("blk", bw, ih)
    if key not in _CACHE:
        _CACHE[key] = vgrad(bw, ih, [(0, (214, 246, 200)), (0.35, (84, 204, 64)), (0.75, (40, 164, 36)), (1, (110, 206, 96))])
    n = int((r.w - 6) // (bw + gap) * max(0.0, min(1.0, frac)))
    for i in range(n):
        surf.blit(_CACHE[key], (r.x + 3 + i * (bw + gap), r.y + 3))


def osd_text(surf, text, x, y, color=(255, 255, 255), px=2, align="left"):
    """Texto de 'display da TV/VCR': fonte pequena ampliada sem suavizar (fica quadradao)."""
    f = xpfont(14, bold=True, kind="mono")
    base = f.render(text, False, color)
    sh = f.render(text, False, (10, 10, 70))
    w, h = base.get_width() * px, base.get_height() * px
    base = pygame.transform.scale(base, (w, h))
    sh = pygame.transform.scale(sh, (w, h))
    if align == "right":
        x -= w
    surf.blit(sh, (x + px + 1, y + px + 1))
    surf.blit(base, (x, y))
    return w, h


def make_cloud(w, h, seed):
    """Nuvem fofa e borrada, feita de varias bolhas gaussianas."""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = np.zeros((h, w), np.float32)
    n = 7
    for i in range(n):
        cx = w * (0.18 + 0.64 * i / (n - 1)) + rng.uniform(-8, 8)
        cy = h * (0.62 - 0.28 * math.sin(math.pi * i / (n - 1))) + rng.uniform(-6, 6)
        r = h * rng.uniform(0.22, 0.38)
        d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (r * r)
        a = np.maximum(a, np.exp(-d2 * 1.6))
    a *= np.clip((h * 0.9 - yy) / (h * 0.25), 0, 1)
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = (np.clip(a, 0, 1) * 235).astype(np.uint8)
    return pygame.image.frombuffer(rgba.tobytes(), (w, h), "RGBA").convert_alpha()


def make_halo(size):
    """Brilho redondo e macio (usado atras do sol, somado na tela)."""
    c = np.linspace(-1, 1, size, dtype=np.float32)
    d = np.sqrt(c[None, :] ** 2 + c[:, None] ** 2)
    v = (np.exp(-d ** 2 * 4.5) * 120).astype(np.uint8)
    arr = np.repeat(v[..., None], 3, axis=2)
    return pygame.image.frombuffer(arr.tobytes(), (size, size), "RGB").convert()


def make_window_sprite(kind):
    """Janelinhas que ficam flutuando no ceu. Retorna (superficie, retangulo_da_barra_ou_None)."""
    w, h = 230, 128
    spr = pygame.Surface((w, h), pygame.SRCALPHA)
    titles = {"bliss": "Bliss.bmp", "copy": "Copiando arquivos...", "ask": "Lembrança"}
    draw_window(spr, (0, 0, w, h), titles[kind])
    body = pygame.Rect(4, 28, w - 8, h - 32)
    f = xpfont(12)
    bar = None
    if kind == "bliss":
        img = pygame.Rect(body.x + 8, body.y + 8, body.w - 16, body.h - 16)
        spr.blit(vgrad(img.w, img.h, [(0, (40, 110, 225)), (1, (196, 222, 252))]), img)
        spr.set_clip(img)
        pygame.draw.ellipse(spr, (96, 176, 56), (img.x - 40, img.y + int(img.h * 0.55), img.w + 80, int(img.h * 0.9)))
        pygame.draw.ellipse(spr, (120, 196, 70), (img.x + 20, img.y + int(img.h * 0.62), img.w, int(img.h * 0.8)))
        spr.set_clip(None)
        pygame.draw.rect(spr, (127, 157, 185), img, 1)
    elif kind == "copy":
        spr.blit(f.render("Copiando 'saudade_2004.mp3'", True, (0, 0, 0)), (body.x + 10, body.y + 6))
        spr.blit(f.render("Tempo restante: para sempre", True, (0, 0, 0)), (body.x + 10, body.y + 22))
        bar = pygame.Rect(body.x + 10, body.y + 44, body.w - 20, 18)
        draw_progress(spr, bar, 0.0)
        draw_button(spr, (body.right - 88, body.bottom - 30, 78, 22), "Cancelar")
    else:
        cx, cy = body.x + 28, body.y + 32
        pygame.draw.circle(spr, (40, 110, 235), (cx, cy), 16)
        pygame.draw.circle(spr, (255, 255, 255), (cx, cy), 16, 2)
        q = xpfont(22, bold=True).render("?", True, (255, 255, 255))
        spr.blit(q, q.get_rect(center=(cx, cy)))
        spr.blit(f.render("Você ainda lembra", True, (0, 0, 0)), (body.x + 56, body.y + 18))
        spr.blit(f.render("de mim?", True, (0, 0, 0)), (body.x + 56, body.y + 34))
        draw_button(spr, (body.centerx - 78, body.bottom - 32, 72, 22), "Sim", default=True)
        draw_button(spr, (body.centerx + 6, body.bottom - 32, 72, 22), "Não")
    return spr.convert_alpha(), bar


def make_popup_sprite():
    """Janela de erro classica (usada em cascata nos drops)."""
    w, h = 262, 122
    spr = pygame.Surface((w, h), pygame.SRCALPHA)
    draw_window(spr, (0, 0, w, h), "Erro")
    f = xpfont(13)
    cx, cy = 34, 60
    pygame.draw.circle(spr, (220, 40, 30), (cx, cy), 17)
    pygame.draw.circle(spr, (255, 255, 255), (cx, cy), 17, 2)
    pygame.draw.line(spr, (255, 255, 255), (cx - 6, cy - 6), (cx + 6, cy + 6), 4)
    pygame.draw.line(spr, (255, 255, 255), (cx - 6, cy + 6), (cx + 6, cy - 6), 4)
    spr.blit(f.render("Este lugar não existe mais.", True, (0, 0, 0)), (64, 44))
    spr.blit(f.render("Deseja continuar mesmo assim?", True, (0, 0, 0)), (64, 62))
    draw_button(spr, (w // 2 - 38, h - 34, 76, 24), "OK", default=True)
    return spr.convert_alpha()


def make_backdrop():
    """Fundo estatico 'colina' usado na tela de carregamento."""
    s = vgrad(W, H, [(0, (40, 110, 225)), (0.65, (196, 222, 252)), (1, (196, 222, 252))])
    pygame.draw.ellipse(s, (96, 176, 56), (int(-W * 0.35), int(H * 0.58), int(W * 1.7), int(H * 1.3)))
    pygame.draw.ellipse(s, (120, 196, 70), (int(-W * 0.1), int(H * 0.68), int(W * 1.2), int(H * 1.1)))
    return s.convert()


class Palette:
    """Cores da faixa atual. Troca de paleta com crossfade; gira o matiz com a energia."""

    def __init__(self, name="bliss"):
        self.name = name
        self.cur = {k: np.array(PALETTES[name][k], dtype=np.float64) for k in KEYS}
        self.pulse = 0.0
        self.phase = 0.0

    def set(self, name):
        self.name = name

    def flip(self, secs):
        """Por alguns instantes, mostra a paleta oposta (usado nos drops)."""
        self.pulse = secs

    def update(self, dt, energy):
        self.pulse = max(0.0, self.pulse - dt)
        name = self.name
        if self.pulse > 0:
            name = ORDER[1 - ORDER.index(name)]
        k = 1 - math.exp(-dt * (16 if self.pulse > 0 else 3))
        for key in KEYS:
            target = np.array(PALETTES[name][key], dtype=np.float64)
            self.cur[key] += (target - self.cur[key]) * k
        self.phase += dt * (0.12 + energy * 1.0)

    def col(self, key):
        return tuple(int(v) for v in self.cur[key])

    def cycle(self, p):
        """Cor que gira entre main -> alt -> hot -> main."""
        p = p % 3.0
        seq = (self.col("main"), self.col("alt"), self.col("hot"), self.col("main"))
        i = int(p)
        return lerp(seq[i], seq[i + 1], p - i)


# ----------------------------------------------------------------------------
# Biblioteca
# ----------------------------------------------------------------------------
class Track:
    def __init__(self, path):
        self.path = path
        self.title = os.path.splitext(os.path.basename(path))[0].upper()
        self.data = None  # analise (cache): so e feita na primeira vez que toca


def add_paths(tracks, paths):
    """Adiciona arquivos (ou pastas inteiras) a biblioteca. Retorna os novos Tracks."""
    added = []
    for p in paths:
        p = os.path.abspath(p)
        if os.path.isdir(p):
            inner = [os.path.join(p, f) for f in sorted(os.listdir(p))]
            added += add_paths(tracks, inner)
        elif os.path.isfile(p) and p.lower().endswith(AUDIO_EXTS):
            if any(t.path == p for t in tracks):
                continue
            tr = Track(p)
            tracks.append(tr)
            added.append(tr)
    return added


def save_library(tracks):
    """Grava os caminhos da biblioteca em JSON. Retorna True se deu certo."""
    try:
        tmp = LIB_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"tracks": [t.path for t in tracks]}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, LIB_FILE)
        return True
    except OSError as ex:
        print(f"Nao consegui salvar a biblioteca: {ex}")
        return False


def load_library():
    """Le a biblioteca salva. Retorna (tracks, qtd_arquivos_que_sumiram)."""
    tracks, missing = [], 0
    try:
        with open(LIB_FILE, encoding="utf-8") as f:
            paths = json.load(f).get("tracks", [])
    except (OSError, ValueError, AttributeError):
        return tracks, missing
    for p in paths:
        if isinstance(p, str) and os.path.isfile(p) and not any(t.path == p for t in tracks):
            tracks.append(Track(p))
        else:
            missing += 1
    return tracks, missing


def pick_files():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    paths = filedialog.askopenfilenames(
        title="Escolha uma ou mais musicas",
        filetypes=[("Audio", "*.mp3 *.wav *.ogg *.flac"), ("Todos", "*.*")],
    )
    root.destroy()
    return list(paths)


# ----------------------------------------------------------------------------
# Analise do audio
# ----------------------------------------------------------------------------
class Analysis:
    def __init__(self, bands, onset, bass, treble, calm, drops, palette, sr):
        self.bands = bands      # (BANDS, quadros)
        self.onset = onset      # batidas / ataques
        self.bass = bass
        self.treble = treble
        self.calm = calm        # 1 = trecho bem quieto (breakdown), 0 = pesado
        self.drops = drops      # lista de tempos (s) onde o drop entra
        self.palette = palette  # "bliss" ou "dream", escolhida pelo timbre da faixa
        self.sr = sr


def analyze(path):
    print("Analisando audio, aguarde...")
    y, sr = librosa.load(path, sr=SR, mono=True)
    spec = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    edges = np.geomspace(40, 11000, BANDS + 1)
    bands = np.zeros((BANDS, spec.shape[1]), dtype=np.float32)
    for i in range(BANDS):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if not mask.any():
            mask[np.argmin(np.abs(freqs - edges[i]))] = True
        bands[i] = spec[mask].mean(axis=0)
    db = librosa.amplitude_to_db(bands, ref=np.max, top_db=60)
    bands = (db + 60.0) / 60.0
    tilt = np.linspace(0.9, 1.2, BANDS)[:, None]  # agudos costumam ser fracos
    bands = np.clip(bands * tilt, 0, 1) ** 1.7
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    onset = onset / (onset.max() + 1e-9)
    bass = bands[:6].mean(axis=0)
    treble = bands[-16:].mean(axis=0)

    # --- estrutura: acha os drops (silencio/calmaria -> pancada) e os trechos calmos
    fps = sr / HOP
    rms = np.sqrt((spec ** 2).mean(axis=0))
    cs = np.concatenate(([0.0], np.cumsum(rms)))
    idx = np.arange(len(rms))

    def mean_between(lo, hi):
        """Media do rms entre (i - hi) e (i - lo) quadros, para todo i."""
        a = np.clip(idx - hi, 0, len(rms))
        b = np.clip(idx - lo, 0, len(rms))
        return (cs[b] - cs[a]) / np.maximum(b - a, 1)

    before = mean_between(int(0.5 * fps), int(3.5 * fps))   # de 3.5s ate 0.5s atras
    now = mean_between(0, int(0.4 * fps))                   # ultimos 0.4s
    avg = float(rms.mean()) + 1e-9
    rise = now / (before + 1e-9)
    cand = np.where((rise > 1.6) & (before < avg * 0.9) & (idx > 4 * fps))[0]

    drops, last = [], -1e9
    for i in cand:
        t = i / fps
        if t - last >= 6.0:             # no minimo 6s entre um drop e outro
            drops.append(max(0.0, t - 0.2))
            last = t

    win = max(1, int(0.5 * fps))
    lvl = np.convolve(rms, np.ones(win) / win, mode="same") / (np.percentile(rms, 90) + 1e-9)
    calm = np.clip(1.0 - lvl / 0.45, 0, 1).astype(np.float32)

    centroid = float(np.median(librosa.feature.spectral_centroid(S=spec, sr=sr)))
    palette = "dream" if centroid >= DREAM_CENTROID else "bliss"
    print(f"drops: {len(drops)} | centroide: {centroid:.0f} Hz -> {palette}")
    return Analysis(bands, onset, bass, treble, calm, drops, palette, sr)


# ----------------------------------------------------------------------------
# Visual
# ----------------------------------------------------------------------------
class Motes:
    """Poeira / bolhas brilhantes subindo devagar (o 'ar' do sonho)."""

    def __init__(self, n=90):
        self.p = [[random.uniform(0, W), random.uniform(0, H), random.uniform(0.5, 1.6), random.uniform(0, 6.28)]
                  for _ in range(n)]

    def update_draw(self, surf, dt, treble, pal, t):
        up = 22 + treble * 170
        glow, ring = pal.col("glow"), pal.col("alt")
        for p in self.p:
            p[1] -= up * p[2] * dt
            p[0] += math.sin(t * 0.8 + p[3]) * 18 * dt + 10 * dt
            if p[1] < -8:
                p[1] = H + 8
                p[0] = random.uniform(0, W)
            if p[0] > W + 8:
                p[0] = -8
            x, y = int(p[0]), int(p[1])
            r = p[2] * 2.0
            pygame.draw.circle(surf, glow, (x, y), max(1, int(r)))
            if p[2] > 1.2:
                pygame.draw.circle(surf, ring, (x, y), int(r * 3), 1)
            if p[2] > 1.45:  # brilhinho em cruz
                pygame.draw.line(surf, glow, (x - 6, y), (x + 6, y), 1)
                pygame.draw.line(surf, glow, (x, y - 6), (x, y + 6), 1)


def _hill(base, amp, f, ph):
    pts = [(0, HORIZON + 2)]
    for x in range(0, W + 9, 8):
        y = HORIZON - base + amp * math.sin(x * f + ph) + amp * 0.45 * math.sin(x * f * 2.3 + ph * 1.7)
        pts.append((x, y))
    pts.append((W, HORIZON + 2))
    return pts


class Visualizer:
    def __init__(self):
        self.pal = Palette("bliss")
        self.h = np.zeros(BANDS)
        self.peak = np.zeros(BANDS)
        self.scroll = 0.0
        self.scroll_v = 0.3
        self.spin = 0.0
        self.tunnel = 0.0
        self.kick = 0.0
        self.glitch = 0.0
        self.cooldown = 0.0
        self.micro_cd = 0.0
        self.mirror = 0.0
        self.calm = 0.0
        self.sides = 6
        self.seq = []          # sequencia de "tela azul" em andamento: [[tempo_restante, modo], ...]
        self.motes = Motes()
        self.popups = []       # cascata de janelas de erro: [idade, x, y]
        self.sky_mid = (150, 190, 240)
        self.sky_bot = (200, 225, 255)

        order = sorted(range(BANDS), key=lambda p: abs(p - (BANDS - 1) / 2))
        self.band_at = [0] * BANDS
        for k, pos in enumerate(order):
            self.band_at[pos] = k
        self.base = [random.uniform(0.72, 1.0) for _ in range(BANDS)]
        self.cell_seed = [random.random() for _ in range(BANDS)]

        self.fb = pygame.Surface((W // 2, H // 2)).convert()   # rastro (feedback) em meia resolucao
        self.fb.fill((190, 215, 250))
        self.white = pygame.Surface((W, H)).convert()
        self.white.fill((255, 255, 255))
        self.fog = pygame.Surface((W, H)).convert()            # nevoa lilas dos trechos calmos
        self.fog.fill((228, 214, 255))

        # cenario
        self.clouds = []
        sprites = [make_cloud(320, 120, 1), make_cloud(260, 100, 2), make_cloud(400, 140, 3)]
        for i in range(9):
            spr = random.choice(sprites)
            self.clouds.append([random.uniform(-200, W), random.uniform(10, HORIZON - 190), spr,
                                random.uniform(6, 22), random.uniform(0, 6.28)])
        self.hills = [(_hill(78, 34, 0.0042, 0.6), 0.55), (_hill(40, 26, 0.0060, 2.4), 0.25)]
        self.halo = make_halo(640)
        self.wins = []
        for kind, ax, ay, ph in (("bliss", 60, 70, 0.5), ("copy", 950, 110, 2.1), ("ask", 1000, 265, 4.0)):
            spr, bar = make_window_sprite(kind)
            self.wins.append({"spr": spr, "bar": bar, "ax": ax, "ay": ay, "ph": ph, "kind": kind})
        self.popup_spr = make_popup_sprite()

        # tela azul + VHS
        self.bsod = self._make_bsod()
        self.overlay = self._make_overlay()
        self.noise = self._make_noise()
        self.noise_hi = self._make_noise_hi()
        self.roll = pygame.Surface((W, 90), pygame.SRCALPHA)
        for yy in range(90):
            a = int(math.sin(math.pi * yy / 90) ** 2 * 30)
            pygame.draw.line(self.roll, (255, 255, 255, a), (0, yy), (W, yy))
        self.roll_y = random.uniform(0, H)
        self.track_cd = 2.0
        self.track_len = 0.0
        self.track_y = 0.0
        self.track_h = 40
        self.track_amp = 12.0

    # --- superficies prontas ---------------------------------------------------
    @staticmethod
    def _make_bsod():
        s = pygame.Surface((W, H))
        s.fill((0, 0, 170))
        f = xpfont(17, kind="mono")
        lines = (
            "Foi detectado um problema e o Windows foi encerrado para evitar danos",
            "ao computador.",
            "",
            "SAUDADE_NOT_LESS_OR_EQUAL",
            "",
            "Se esta for a primeira vez que você vê esta tela de erro,",
            "reinicie o computador. Se esta tela aparecer novamente, siga",
            "estes passos:",
            "",
            "Verifique se você ainda lembra de como era antes.",
            "",
            "Informações técnicas:",
            "",
            "*** STOP: 0x0000000A (0x00000004,0x00000002,0x00000000,0xF73E4A12)",
            "",
            "Coletando dados para o despejo de memória...",
        )
        y = 60
        for ln in lines:
            s.blit(f.render(ln, True, (255, 255, 255)), (48, y))
            y += 22
        return s.convert()

    @staticmethod
    def _make_overlay():
        """Vinheta azul-escura + linhas de varredura numa unica camada (um blit so)."""
        yy, xx = np.mgrid[0:H, 0:W]
        d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        av = np.clip((d - 0.55) / 0.9, 0, 1) ** 1.5 * 150
        asc = np.where(yy % 2 == 1, 38.0, 0.0)
        a = 255.0 * (1.0 - (1.0 - av / 255.0) * (1.0 - asc / 255.0))
        arr = np.zeros((H, W, 4), dtype=np.uint8)
        arr[..., 0], arr[..., 1], arr[..., 2] = 6, 3, 24
        arr[..., 3] = a.astype(np.uint8)
        return pygame.image.frombuffer(arr.tobytes(), (W, H), "RGBA").convert_alpha()

    @staticmethod
    def _make_noise(n=6):
        """Chuvisco de fita (listrado na horizontal) ja com um 'lift' que levanta os pretos."""
        out = []
        nw, nh = W + 16, H + 8
        lift = np.array((9, 5, 15), dtype=np.float32)
        tint = np.array((1.0, 0.9, 1.1), dtype=np.float32)
        for _ in range(n):
            g = (np.random.rand(nh // 2, nw // 9 + 1) ** 3 * 62).astype(np.float32)
            arr = np.clip(g[..., None] * tint + lift, 0, 255).astype(np.uint8)
            small = pygame.image.frombuffer(arr.tobytes(), (arr.shape[1], arr.shape[0]), "RGB")
            out.append(pygame.transform.scale(small, (nw, nh)).convert())
        return out

    @staticmethod
    def _make_noise_hi(n=3):
        """Chuvisco forte, para as faixas de falha de rastreamento."""
        out = []
        for _ in range(n):
            g = (np.random.rand(40, W // 7 + 1) ** 1.5 * 190).astype(np.uint8)
            arr = np.repeat(g[..., None], 3, axis=2)
            small = pygame.image.frombuffer(arr.tobytes(), (arr.shape[1], arr.shape[0]), "RGB")
            out.append(pygame.transform.scale(small, (W, 80)).convert())
        return out

    # --- efeitos ------------------------------------------------------------
    def start(self, pattern):
        if not BLACKOUTS:
            return
        seq = []
        for dur, mode in pattern:
            if not FLASHES and mode in ("flash", "invert"):
                mode = "normal"
            seq.append([dur, mode])
        self.seq = seq

    def mode(self):
        return self.seq[0][1] if self.seq else "normal"

    def drop(self):
        self.start(random.choice(DROP_PATTERNS))
        self.glitch = 1.6
        self.mirror = random.uniform(0.6, 1.4)
        self.sides = random.choice((3, 4, 5, 6, 8))
        self.track_cd = 0.0     # fita falha junto com o drop
        # cascata de janelas de erro
        n = random.randint(5, 8)
        x0, y0 = random.randint(220, 520), random.randint(90, 190)
        self.popups = [[-i * 0.07, x0 + i * 26, y0 + i * 26] for i in range(n)]
        if FLASHES:
            self.pal.flip(0.9)

    def reset_fx(self):
        self.seq = []
        self.mirror = 0.0
        self.glitch = 0.0
        self.popups = []

    def update(self, dt, bands_now, onset_now, bass_now, calm_now):
        k_up = 1 - math.exp(-dt * 40)
        k_dn = 1 - math.exp(-dt * 7)
        self.h += (bands_now - self.h) * np.where(bands_now > self.h, k_up, k_dn)
        self.peak = np.maximum(self.peak - dt * 0.35, self.h)

        self.kick = max(self.kick * math.exp(-dt * 9), onset_now)
        self.cooldown -= dt
        self.micro_cd -= dt
        if onset_now > 0.72 and self.cooldown <= 0:
            self.glitch = max(self.glitch, 1.0)
            self.cooldown = 0.25
            if onset_now > 0.85 and self.micro_cd <= 0 and not self.seq and random.random() < 0.5:
                self.start(random.choice(MICRO_PATTERNS))
                self.micro_cd = 0.4
        self.glitch *= math.exp(-dt * 9)
        self.mirror = max(0.0, self.mirror - dt)
        self.calm += (calm_now - self.calm) * (1 - math.exp(-dt * 3))

        energy = float(self.h.mean())
        self.scroll_v += ((0.25 + energy * 4.0 + bass_now * 2.0) - self.scroll_v) * (1 - math.exp(-dt * 4))
        self.scroll += dt * self.scroll_v
        self.spin += dt * (0.15 + energy * 1.2 + bass_now * 0.8)
        self.tunnel += dt * (0.25 + energy * 2.2 + bass_now * 1.2)
        self.pal.update(dt, energy)

        for p in self.popups:
            p[0] += dt
        self.popups = [p for p in self.popups if p[0] < 1.6]

        if self.seq:
            self.seq[0][0] -= dt
            if self.seq[0][0] <= 0:
                self.seq.pop(0)

    # --- desenho ------------------------------------------------------------
    def draw_bg(self, s, bass):
        """Ceu: azul fundo em cima, quase branco no horizonte."""
        pal = self.pal
        main, glow = pal.col("main"), pal.col("glow")
        top, bot = scale(main, 0.78), lerp(main, glow, 0.78)
        boost = self.kick * 0.12 + bass * 0.06
        n = 40
        for i in range(n):
            c = lerp(lerp(top, bot, (i / (n - 1)) ** 0.9), glow, boost)
            y0, y1 = HORIZON * i // n, HORIZON * (i + 1) // n
            pygame.draw.rect(s, c, (0, y0, W, y1 - y0 + 1))
        self.sky_mid = lerp(top, bot, 0.5)
        self.sky_bot = bot

    def draw_tunnel(self, s):
        """Aneis poligonais girando e vindo pra cima da gente (agora feitos de luz)."""
        pal = self.pal
        cx, cy = W // 2, HORIZON - 70
        rings = 9
        for j in range(rings):
            k = ((j + self.tunnel) % rings) / rings
            r = 40 + k ** 1.8 * 900
            ang = self.spin * (1 if j % 2 else -1) + j * 0.2
            pts = [(cx + math.cos(ang + 2 * math.pi * a / self.sides) * r,
                    cy + math.sin(ang + 2 * math.pi * a / self.sides) * r) for a in range(self.sides)]
            inten = (1 - k) ** 1.5 * (0.30 + self.kick * 0.55)
            ring = lerp(pal.cycle(pal.phase + j * 0.35), (255, 255, 255), 0.55)
            pygame.draw.polygon(s, lerp(self.sky_mid, ring, inten * 1.6), pts, 2 + int(k * 3))

    def draw_sun(self, s, bass):
        """Sol palido com um olho no meio, boiando no horizonte."""
        pal = self.pal
        cx, cy = W // 2, HORIZON - 70
        r = int(120 + bass * 50 + self.kick * 18)
        s.blit(self.halo, (cx - 320, cy - 320), special_flags=pygame.BLEND_RGB_ADD)
        glow = pal.col("glow")
        top_c, bot_c = glow, lerp(pal.col("alt"), glow, 0.35)
        for y in range(cy - r, cy + r):
            dy = y - cy
            half = math.sqrt(max(0, r * r - dy * dy))
            if half <= 0:
                continue
            pygame.draw.line(s, lerp(top_c, bot_c, (y - (cy - r)) / (2 * r)), (cx - half, y), (cx + half, y))
        pygame.draw.circle(s, glow, (cx, cy), r, 3)
        pygame.draw.circle(s, pal.cycle(pal.phase * 2), (cx, cy), int(r * 0.8), 2)
        eye_w = 5 + int(self.kick * 16 + bass * 10)
        eye_h = int(r * 0.55)
        pygame.draw.ellipse(s, scale(pal.col("main"), 0.6), (cx - eye_w, cy - eye_h, eye_w * 2, eye_h * 2))
        pw = max(2, eye_w // 3)
        pygame.draw.ellipse(s, (8, 16, 50), (cx - pw, cy - eye_h * 0.8, pw * 2, eye_h * 1.6))
        pygame.draw.circle(s, (255, 255, 255), (cx - eye_w // 2, cy - eye_h // 2), 3)

    def draw_clouds(self, s, dt):
        for c in self.clouds:
            c[0] += c[3] * dt * (1.0 + self.scroll_v * 0.15)
            if c[0] > W:
                c[0] = -c[2].get_width()
            s.blit(c[2], (int(c[0]), int(c[1] + math.sin(c[4] + c[0] * 0.01) * 4)))

    def draw_wave(self, s, t):
        """Osciloscopio cortando o ceu, mais forte no centro (graves)."""
        n = 90
        base_y = HORIZON * 0.30
        pts = []
        for i in range(n + 1):
            u = i / n
            k = min(BANDS - 1, int(abs(u * 2 - 1) * (BANDS - 1)))
            lvl = float(self.h[k])
            y = base_y + math.sin(u * 14 + t * 5) * lvl * 60 + math.sin(u * 37 - t * 9) * lvl * 22
            pts.append((u * W, y))
        shadow = [(x, y + 4) for x, y in pts]
        pygame.draw.lines(s, scale(self.pal.col("main"), 0.55), False, shadow, 2)
        pygame.draw.lines(s, lerp(self.pal.cycle(self.pal.phase * 1.3 + 1.0), (255, 255, 255), 0.6), False, pts, 3)

    def draw_windows(self, s, t):
        """Janelinhas do XP flutuando no ceu. A de 'copiando' mostra o grave na barrinha."""
        for w in self.wins:
            x = w["ax"] + math.sin(t * 0.5 + w["ph"]) * 22
            y = w["ay"] + math.cos(t * 0.7 + w["ph"]) * 14 - self.kick * 8
            if self.glitch > 0.3:
                x += random.randint(-6, 6) * self.glitch
            x, y = int(x), int(y)
            s.blit(w["spr"], (x, y))
            if w["bar"] is not None:
                b = w["bar"]
                draw_progress(s, (x + b.x, y + b.y, b.w, b.h), float(self.h[:8].mean()) * 1.2)

    def draw_hills(self, s):
        green = self.pal.col("hot")
        for pts, mix in self.hills:
            pygame.draw.polygon(s, lerp(green, self.sky_bot, mix), pts)

    def draw_ground(self, s, bass):
        """Gramado com faixas de corte vindo na nossa direcao."""
        pal = self.pal
        green = pal.col("hot")
        g_top, g_bot = lerp(green, self.sky_bot, 0.25), scale(green, 0.70)
        gh = H - HORIZON
        n = 26
        for i in range(n):
            y0, y1 = HORIZON + gh * i // n, HORIZON + gh * (i + 1) // n
            pygame.draw.rect(s, lerp(g_top, g_bot, i / (n - 1)), (0, y0, W, y1 - y0 + 1))

        nn = 14
        base = math.floor(self.scroll)
        frac = self.scroll - base
        light = 0.07 + bass * 0.10
        for i in range(-1, nn):
            if (i - base) % 2:
                continue
            k0, k1 = (i + frac) / nn, (i + 1 + frac) / nn
            y0 = HORIZON + gh * max(0.0, min(1.0, k0)) ** 2.2
            y1 = HORIZON + gh * max(0.0, min(1.0, k1)) ** 2.2
            if y1 - y0 < 1:
                continue
            ym = ((y0 + y1) / 2 - HORIZON) / gh
            col = lerp(lerp(g_top, g_bot, ym), (255, 255, 255), light)
            pygame.draw.rect(s, col, (0, int(y0), W, int(y1) - int(y0) + 1))

        line = lerp(g_top, (255, 255, 255), 0.28 + bass * 0.2)
        for i in range(-22, 23):
            pygame.draw.line(s, line, (W / 2 + i * 6, HORIZON), (W / 2 + i * 120, H), 1)

    def draw_city(self, s, t):
        """O equalizador: colunas de blocos, igual a barra de progresso do XP."""
        pal = self.pal
        glow_c = pal.col("glow")
        trough = lerp(self.sky_mid, (255, 255, 255), 0.7)
        shade = lerp(pal.col("hot"), (0, 0, 0), 0.35)
        bw = W // BANDS
        x0 = (W - bw * BANDS) // 2
        max_h = HORIZON * 0.82
        seg = 11
        for pos in range(BANDS):
            k = self.band_at[pos]
            level = float(self.h[k])
            seed = self.cell_seed[pos]
            height = 30 + level * max_h * self.base[pos]
            x = x0 + pos * bw
            top = HORIZON - height
            col = pal.cycle(pal.phase + k * 0.07)

            pygame.draw.rect(s, shade, (x + 2, HORIZON + 3, bw - 3, int(height * 0.2)))   # sombra no gramado
            pygame.draw.rect(s, trough, (x + 1, top, bw - 2, height))
            pygame.draw.rect(s, (130, 135, 144), (x + 1, top, bw - 2, height), 1)

            rows = int((height - 4) / seg)
            for r in range(rows):
                frac = r / max(1, rows)
                c = lerp(col, glow_c, 0.18 * frac + max(0.0, level - 0.7) * 0.8 * frac)
                pygame.draw.rect(s, c, (x + 4, HORIZON - 3 - (r + 1) * seg, bw - 8, seg - 3))
            pygame.draw.line(s, (255, 255, 255), (x + 4, HORIZON - 3 - rows * seg), (x + 4, HORIZON - 4))  # brilho lateral

            py = HORIZON - (30 + float(self.peak[k]) * max_h * self.base[pos])
            pygame.draw.rect(s, (255, 255, 255), (x + 2, int(py) - 2, bw - 4, 3))
            pygame.draw.line(s, scale(pal.col("main"), 0.5), (x + 2, int(py) + 1), (x + bw - 3, int(py) + 1))

            if level > 0.78 and seed > 0.3:   # brilhinho no topo quando bate forte
                sx, sy, ln = x + bw // 2, int(top) - 10, 4 + int(level * 8)
                pygame.draw.line(s, glow_c, (sx - ln, sy), (sx + ln, sy), 2)
                pygame.draw.line(s, glow_c, (sx, sy - ln), (sx, sy + ln), 2)

    def draw_popups(self, s):
        for age, x, y in self.popups:
            if age >= 0:
                s.blit(self.popup_spr, (x, y))

    def render_scene(self, scene, t, bass, treble, dt):
        self.draw_bg(scene, bass)
        self.draw_tunnel(scene)
        self.draw_sun(scene, bass)
        self.draw_clouds(scene, dt)
        self.draw_wave(scene, t)
        self.draw_windows(scene, t)
        self.draw_hills(scene)
        self.draw_ground(scene, bass)
        self.draw_city(scene, t)
        self.motes.update_draw(scene, dt, treble, self.pal, t)
        self.draw_popups(scene)

    def post(self, scene, screen, t, bass):
        pal = self.pal

        # 1) rastro sonhado: o quadro anterior volta ampliado e meio transparente
        zoom = 1.008 + self.kick * 0.03
        zw, zh = int(W * zoom), int(H * zoom)
        trail = pygame.transform.scale(self.fb, (zw, zh))
        trail.set_alpha(min(200, int(55 + self.glitch * 50 + self.kick * 45)))
        scene.blit(trail, ((W - zw) // 2, (H - zh) // 2))
        pygame.transform.scale(scene, self.fb.get_size(), self.fb)

        # 2) bloom (luz estourando de leve)
        small = pygame.transform.smoothscale(self.fb, (W // 8, H // 8))
        glow = pygame.transform.smoothscale(small, (W, H))
        v = int(16 + 56 * min(1.0, bass * 1.3))
        glow.fill((v, v, v), special_flags=pygame.BLEND_RGB_MULT)
        scene.blit(glow, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

        mode = self.mode()
        if mode == "black":          # "apagao" = tela azul da morte
            screen.blit(self.bsod, (0, 0))
            return

        # 3) aberracao cromatica: sempre um pouquinho (VHS), bem forte nos chutes
        ab = int((2 + self.kick * 6 + self.glitch * 10) * (0.5 + 0.5 * VHS_AMOUNT))
        if ab > 4:
            screen.fill((0, 0, 0))
            for color, off in (((255, 0, 0), -ab), ((0, 255, 0), 0), ((0, 0, 255), ab)):
                layer = scene.copy()
                layer.fill(color, special_flags=pygame.BLEND_RGB_MULT)
                screen.blit(layer, (off, 0), special_flags=pygame.BLEND_RGB_ADD)
        else:
            screen.blit(scene, (0, 0))
            if VHS_AMOUNT > 0:
                red = scene.copy()
                red.fill((255, 0, 0), special_flags=pygame.BLEND_RGB_MULT)
                screen.fill((0, 255, 255), special_flags=pygame.BLEND_RGB_MULT)
                screen.blit(red, (-ab, 0), special_flags=pygame.BLEND_RGB_ADD)

        # 4) ondulacao + fatias deslocadas
        warp = self.kick * 0.25 + self.glitch * 0.5
        if warp > 0.08 or self.glitch > 0.12:
            src = screen.copy()
            if warp > 0.08:
                amp = warp * 28
                for y in range(0, H, 6):
                    dx = int(math.sin(y * 0.035 + t * 14) * amp)
                    if dx:
                        screen.blit(src, (dx, y), (0, y, W, 6))
            if self.glitch > 0.12:
                for _ in range(int(self.glitch * 9)):
                    y = random.randint(0, H - 60)
                    hh = random.randint(6, 46)
                    dx = random.randint(-90, 90)
                    screen.blit(src, (dx, y), (0, y, W, hh))

        # 5) espelho (kaleidoscopio) logo depois do drop
        if self.mirror > 0:
            half = screen.subsurface((0, 0, W // 2, H)).copy()
            screen.blit(pygame.transform.flip(half, True, False), (W // 2, 0))

        # 6) trechos calmos ficam enevoados (a tensao antes do drop)
        if self.calm > 0.02:
            self.fog.set_alpha(int(self.calm * 120))
            screen.blit(self.fog, (0, 0))

        # 7) flash / inversao
        if mode == "flash":
            screen.fill(lerp(pal.col("main"), (255, 255, 255), 0.5))
        elif mode == "invert":
            inv = self.white.copy()
            inv.blit(screen, (0, 0), special_flags=pygame.BLEND_SUB)
            screen.blit(inv, (0, 0))

    def vhs(self, screen, t, dt):
        """Camada final de fita VHS: rastreamento, chuvisco, barra rolando, linhas e vinheta."""
        if VHS_AMOUNT <= 0:
            return
        amt = VHS_AMOUNT

        # flagging: o topo do quadro balanca
        seg = screen.subsurface((0, 0, W, 40)).copy()
        for yy in range(0, 40, 4):
            dx = int((math.sin(t * 2.3 + yy * 0.15) * 3 + random.uniform(-1, 1)) * amt * (1 + self.glitch * 2))
            screen.blit(seg, (dx, yy), (0, yy, W, 4))

        # ruido de troca de cabecote la embaixo
        hs = 16
        seg = screen.subsurface((0, H - hs, W, hs)).copy()
        wob = (5 + 25 * min(1.0, self.glitch + self.kick * 0.5)) * amt
        for yy in range(0, hs, 2):
            screen.blit(seg, (int(random.gauss(0, 1) * wob), H - hs + yy), (0, yy, W, 2))
        screen.blit(random.choice(self.noise_hi), (0, H - hs), (0, 0, W, hs), special_flags=pygame.BLEND_RGB_ADD)

        # faixa de falha de rastreamento passando pela tela
        self.track_cd -= dt
        if self.track_len <= 0 and self.track_cd <= 0:
            self.track_len = random.uniform(0.35, 0.9)
            self.track_y = random.uniform(H * 0.1, H * 0.8)
            self.track_h = random.randint(26, 70)
            self.track_amp = random.uniform(8, 26)
            self.track_cd = random.uniform(3.0, 8.0) / max(0.3, amt)
        if self.track_len > 0:
            self.track_len -= dt
            self.track_y += 140 * dt
            y0 = int(max(0, min(H - self.track_h, self.track_y)))
            th = self.track_h
            seg = screen.subsurface((0, y0, W, th)).copy()
            for yy in range(0, th, 3):
                dx = int((math.sin(yy * 0.25 + t * 40) * self.track_amp + random.randint(-3, 3)) * amt)
                screen.blit(seg, (dx, y0 + yy), (0, yy, W, 3))
            screen.blit(random.choice(self.noise_hi), (0, y0), (0, 0, W, th), special_flags=pygame.BLEND_RGB_ADD)

        # chuvisco geral (ja levanta os pretos)
        screen.blit(random.choice(self.noise), (random.randint(-16, 0), random.randint(-8, 0)),
                    special_flags=pygame.BLEND_RGB_ADD)

        # barra clarinha rolando devagar
        self.roll_y += 55 * dt
        if self.roll_y > H + 90:
            self.roll_y = -90
        screen.blit(self.roll, (0, int(self.roll_y)))

        # linhas de varredura + vinheta
        screen.blit(self.overlay, (0, 0))


# ----------------------------------------------------------------------------
# Menu da biblioteca (uma janela do XP)
# ----------------------------------------------------------------------------
class Menu:
    ROW_H = 26
    ROWS = 12

    def __init__(self):
        self.open = True
        self.sel = 0
        self.scroll = 0
        self.panel = pygame.Rect(W // 2 - 380, 96, 760, 486)
        p = self.panel
        self.list_rect = pygame.Rect(p.x + 16, p.y + 66, p.w - 32, self.ROW_H * self.ROWS)
        self.visible = self.ROWS
        by = self.list_rect.bottom + 12
        self.btn_add = pygame.Rect(p.x + 16, by, 190, 26)
        self.btn_rem = pygame.Rect(p.x + 214, by, 150, 26)
        self.btn_close = pygame.Rect(p.right - 16 - 110, by, 110, 26)
        self.btn_x = pygame.Rect(p.right - 6 - 21, p.y + 4, 21, 21)      # "X" da barra de titulo
        self.hint_y = by + 26 + 12
        self.status_rect = pygame.Rect(p.x + 4, p.bottom - 26, p.w - 8, 22)

        self.back = pygame.Surface((W, H), pygame.SRCALPHA)
        self.back.fill((10, 20, 70, 110))
        self.shadow = pygame.Surface(p.size, pygame.SRCALPHA)
        pygame.draw.rect(self.shadow, (0, 0, 0, 70), self.shadow.get_rect(), border_radius=9)

    def clamp(self, n):
        self.sel = max(0, min(n - 1, self.sel)) if n else 0
        if self.sel < self.scroll:
            self.scroll = self.sel
        if self.sel >= self.scroll + self.visible:
            self.scroll = self.sel - self.visible + 1
        self.scroll = max(0, min(self.scroll, max(0, n - self.visible)))

    def handle(self, e, n):
        """Retorna uma acao: ('play', i) | ('add',) | ('remove', i) | ('close',) | None"""
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_UP:
                self.sel -= 1
            elif e.key == pygame.K_DOWN:
                self.sel += 1
            elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                if n:
                    return ("play", self.sel)
            elif e.key in (pygame.K_a, pygame.K_o):
                return ("add",)
            elif e.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
                if n:
                    return ("remove", self.sel)
            elif e.key in (pygame.K_m, pygame.K_TAB, pygame.K_ESCAPE):
                return ("close",)
            self.clamp(n)
        elif e.type == pygame.MOUSEWHEEL:
            self.scroll = max(0, min(self.scroll - e.y, max(0, n - self.visible)))
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if self.btn_x.collidepoint(e.pos) or self.btn_close.collidepoint(e.pos):
                return ("close",)
            if self.btn_add.collidepoint(e.pos):
                return ("add",)
            if self.btn_rem.collidepoint(e.pos):
                if n:
                    return ("remove", self.sel)
                return None
            if self.list_rect.collidepoint(e.pos):
                i = self.scroll + (e.pos[1] - self.list_rect.y) // self.ROW_H
                if i < n:
                    self.sel = i
                    return ("play", i)
        return None

    def draw(self, screen, tracks, current, paused, status):
        n = len(tracks)
        p = self.panel
        mp = pygame.mouse.get_pos()
        f, fb = xpfont(13), xpfont(13, bold=True)

        screen.blit(self.back, (0, 0))
        screen.blit(self.shadow, (p.x + 7, p.y + 7))
        draw_window(screen, p, "Biblioteca de Músicas - Nostalgia.exe")
        screen.blit(f.render("Músicas na biblioteca:", True, (0, 0, 0)), (p.x + 18, p.y + 40))

        lr = self.list_rect
        pygame.draw.rect(screen, (255, 255, 255), lr)
        pygame.draw.rect(screen, (127, 157, 185), lr.inflate(2, 2), 1)

        if not tracks:
            for k, line in enumerate(("Nenhuma música ainda.",
                                      "Clique em Adicionar ou arraste arquivos/pastas para a janela.")):
                txt = f.render(line, True, (110, 110, 110))
                screen.blit(txt, txt.get_rect(center=(lr.centerx, lr.centery - 12 + k * 24)))

        row_w = lr.w - (18 if n > self.visible else 0)
        for row in range(self.visible):
            i = self.scroll + row
            if i >= n:
                break
            r = pygame.Rect(lr.x, lr.y + row * self.ROW_H, row_w, self.ROW_H)
            sel = i == self.sel
            if sel:
                pygame.draw.rect(screen, XP_SEL, r)
            playing = i == current
            mark = ("|| " if paused else ">> ") if playing else "   "
            col = (255, 255, 255) if sel else ((0, 60, 190) if playing else (0, 0, 0))
            font = fb if playing else f
            text = fit(font, f"{mark}{i + 1:02d}   {tracks[i].title}", r.w - 20)
            screen.blit(font.render(text, True, col), (r.x + 8, r.y + 5))

        if n > self.visible:  # barra de rolagem do XP
            track = pygame.Rect(lr.right - 16, lr.y, 16, lr.h)
            pygame.draw.rect(screen, (244, 243, 238), track)
            hh = max(30, track.h * self.visible // n)
            yy = track.y + (track.h - hh) * self.scroll // max(1, n - self.visible)
            thumb = pygame.Rect(track.x + 1, yy, track.w - 2, hh)
            pygame.draw.rect(screen, (196, 213, 252), thumb, border_radius=3)
            pygame.draw.rect(screen, (112, 141, 232), thumb, 1, border_radius=3)
            for g in (-3, 0, 3):
                pygame.draw.line(screen, (112, 141, 232), (thumb.x + 4, thumb.centery + g), (thumb.right - 5, thumb.centery + g))

        draw_button(screen, self.btn_add, "+ Adicionar [A]", self.btn_add.collidepoint(mp), default=True)
        draw_button(screen, self.btn_rem, "Remover [DEL]", self.btn_rem.collidepoint(mp))
        draw_button(screen, self.btn_close, "Fechar [M]", self.btn_close.collidepoint(mp))

        hint = "ENTER/clique toca  -  DEL remove  -  setas navegam  -  arraste arquivos aqui"
        screen.blit(xpfont(12).render(hint, True, (90, 90, 90)), (p.x + 18, self.hint_y))

        sr = self.status_rect
        left = pygame.Rect(sr.x, sr.y, sr.w - 172, sr.h)
        right = pygame.Rect(left.right + 2, sr.y, 170, sr.h)
        draw_field(screen, left)
        draw_field(screen, right)
        info = status if status else f"{n} música(s) na biblioteca"
        screen.blit(xpfont(12).render(fit(xpfont(12), info, left.w - 12), True, (0, 0, 0)), (left.x + 6, left.y + 4))
        state = "Parado" if current < 0 else ("Pausado" if paused else "Tocando")
        screen.blit(xpfont(12).render(state, True, (0, 0, 0)), (right.x + 6, right.y + 4))


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
class App:
    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 2, 1024)
        pygame.init()
        flags = pygame.SCALED | pygame.RESIZABLE
        try:
            self.screen = pygame.display.set_mode((W, H), flags, vsync=1)
        except (TypeError, pygame.error):
            self.screen = pygame.display.set_mode((W, H), flags)
        pygame.display.set_caption("Nostalgia.exe")
        self.scene = pygame.Surface((W, H)).convert()
        self.clock = pygame.time.Clock()
        self.font = xpfont(13)

        self.vis = Visualizer()
        self.menu = Menu()
        self.backdrop = make_backdrop()
        self._build_taskbar()
        self.tracks, missing = load_library()   # biblioteca salva
        self.cur = -1      # indice da faixa tocando
        self.paused = False
        self.running = True
        self.status = ""
        self.status_t = 0.0
        self.t_est = 0.0   # relogio suavizado da musica (o get_pos do mixer e "pulado")
        self.drop_i = 0    # proximo drop da faixa atual
        if missing:
            self.say(f"{missing} arquivo(s) da biblioteca não foram encontrados e saíram da lista.", 6.0)
            save_library(self.tracks)

    # --- barra de tarefas -------------------------------------------------------
    def _build_taskbar(self):
        self.start_rect = pygame.Rect(0, H - TASKBAR_H, 108, TASKBAR_H)
        self.tb_bg = vgrad(W, TASKBAR_H, [(0, (63, 140, 250)), (0.1, (36, 104, 236)), (0.55, (30, 92, 224)), (1, (24, 76, 196))])
        self.tray_bg = vgrad(122, TASKBAR_H, [(0, (30, 170, 250)), (0.1, (18, 150, 240)), (1, (12, 120, 224))])

        def make_start(dim):
            w, h = self.start_rect.w, TASKBAR_H
            k = 0.82 if dim else 1.0
            g = vgrad(w, h, [(0, scale((120, 208, 110), k)), (0.15, scale((72, 172, 62), k)),
                             (0.8, scale((46, 142, 40), k)), (1, scale((34, 112, 30), k))])
            out = g.convert_alpha()
            mask = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, w, h),
                             border_top_right_radius=15, border_bottom_right_radius=15)
            out.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            # bandeirinha do Windows
            ox, oy = 12, 6
            for pts, c in (
                ([(0, 2), (7, 0), (7, 7), (0, 8)], (243, 83, 37)),
                ([(9, 0), (16, 2), (16, 9), (9, 7)], (129, 188, 0)),
                ([(0, 10), (7, 9), (7, 16), (0, 18)], (5, 166, 240)),
                ([(9, 9), (16, 11), (16, 18), (9, 16)], (255, 186, 8)),
            ):
                pygame.draw.polygon(out, c, [(ox + a, oy + b) for a, b in pts])
            f = xpfont(20, bold=True, italic=True, kind="title")
            tx, ty = 36, TASKBAR_H // 2 - f.get_height() // 2
            out.blit(f.render("iniciar", True, (30, 70, 30)), (tx + 1, ty + 1))
            out.blit(f.render("iniciar", True, (255, 255, 255)), (tx, ty))
            return out

        self.start_up = make_start(False)
        self.start_dn = make_start(True)

    def draw_taskbar(self, screen, title):
        y = H - TASKBAR_H
        screen.blit(self.tb_bg, (0, y))
        screen.blit(self.start_dn if self.menu.open else self.start_up, (0, y))

        # botao da "janela" (a musica atual)
        tb = pygame.Rect(116, y + 3, 300, TASKBAR_H - 6)
        pygame.draw.rect(screen, (22, 74, 180), tb, border_radius=3)
        pygame.draw.rect(screen, (10, 44, 130), tb, 1, border_radius=3)
        f = xpfont(12, bold=True)
        label = ("|| " if self.paused else "> ") + title
        screen.blit(f.render(fit(f, label, tb.w - 16), True, (255, 255, 255)), (tb.x + 8, tb.y + 4))

        hint = "[M] menu  [ESPAÇO] pausa  [N/P] faixa  [C] cor  [F] tela cheia  [ESC] sair"
        hf = xpfont(11)
        screen.blit(hf.render(fit(hf, hint, W - 122 - 430), True, (205, 228, 255)), (430, y + 8))

        # bandeja do sistema (relogio)
        tx = W - 122
        screen.blit(self.tray_bg, (tx, y))
        pygame.draw.line(screen, (9, 69, 164), (tx, y), (tx, y + TASKBAR_H))
        pygame.draw.polygon(screen, (255, 255, 255), [(tx + 12, y + 12), (tx + 17, y + 12), (tx + 22, y + 8),
                                                       (tx + 22, y + 22), (tx + 17, y + 18), (tx + 12, y + 18)])
        clock = xpfont(12).render(time.strftime("%H:%M"), True, (255, 255, 255))
        screen.blit(clock, (W - clock.get_width() - 12, y + 8))

        # balao de aviso (as mensagens de status)
        if self.status and not self.menu.open:
            bf = xpfont(12)
            txt = bf.render(self.status, True, (0, 0, 0))
            bw, bh = txt.get_width() + 22, 28
            br = pygame.Rect(W - bw - 12, y - bh - 12, bw, bh)
            pygame.draw.rect(screen, (255, 255, 225), br, border_radius=5)
            pygame.draw.rect(screen, (0, 0, 0), br, 1, border_radius=5)
            pygame.draw.polygon(screen, (255, 255, 225), [(br.right - 30, br.bottom), (br.right - 18, br.bottom),
                                                          (br.right - 24, br.bottom + 9)])
            pygame.draw.lines(screen, (0, 0, 0), False, [(br.right - 30, br.bottom), (br.right - 24, br.bottom + 9),
                                                         (br.right - 18, br.bottom)])
            screen.blit(txt, (br.x + 11, br.y + 7))

    def draw_osd(self, t_disp):
        """Texto de tela de VCR: PLAY / PAUSE, contador e carimbo de data."""
        if VHS_AMOUNT <= 0:
            return
        white = (255, 255, 255)
        x, y = 40, 28
        if self.cur < 0:
            pygame.draw.rect(self.screen, white, (x, y + 4, 18, 18))
            osd_text(self.screen, "STOP", x + 32, y)
        elif self.paused:
            if int(time.time() * 2) % 2 == 0:
                pygame.draw.rect(self.screen, white, (x, y + 2, 7, 22))
                pygame.draw.rect(self.screen, white, (x + 12, y + 2, 7, 22))
                osd_text(self.screen, "PAUSE", x + 32, y)
        else:
            pygame.draw.polygon(self.screen, white, [(x, y + 2), (x, y + 24), (x + 20, y + 13)])
            osd_text(self.screen, "PLAY", x + 32, y)

        secs = int(max(0, t_disp))
        counter = f"SP  {secs // 3600}:{secs // 60 % 60:02d}:{secs % 60:02d}"
        osd_text(self.screen, counter, W - 40, y, align="right")

        now = time.localtime()
        date = f"{MONTHS[now.tm_mon - 1]}. {now.tm_mday:02d} {VHS_YEAR}"
        osd_text(self.screen, date, W - 40, H - TASKBAR_H - 82, align="right")
        osd_text(self.screen, time.strftime("%I:%M %p"), W - 40, H - TASKBAR_H - 46, align="right")

    # --- helpers -----------------------------------------------------------
    def say(self, msg, secs=4.0):
        self.status, self.status_t = msg, secs

    def loading(self, title):
        self.screen.blit(self.backdrop, (0, 0))
        r = pygame.Rect(W // 2 - 220, H // 2 - 70, 440, 140)
        draw_window(self.screen, r, "Analisando áudio")
        f = xpfont(13)
        self.screen.blit(f.render("Analisando o áudio, aguarde...", True, (0, 0, 0)), (r.x + 20, r.y + 44))
        self.screen.blit(f.render(fit(f, title, r.w - 40), True, (60, 60, 60)), (r.x + 20, r.y + 64))
        draw_progress(self.screen, (r.x + 20, r.y + 92, r.w - 40, 20), 0.28)
        pygame.display.flip()
        pygame.event.pump()

    # --- biblioteca --------------------------------------------------------
    def add(self, paths):
        if isinstance(paths, str):
            paths = [paths]
        added = add_paths(self.tracks, paths)
        if not added:
            self.say("Nada novo adicionado (formato inválido ou já estava na lista).")
            return
        self.menu.sel = self.tracks.index(added[0])
        self.menu.clamp(len(self.tracks))
        save_library(self.tracks)
        self.say(f"{len(added)} música(s) adicionada(s).")
        if self.cur == -1:
            self.play(self.tracks.index(added[0]))

    def remove(self, i):
        if not (0 <= i < len(self.tracks)):
            return
        if i == self.cur:
            pygame.mixer.music.stop()
            self.cur = -1
            self.paused = False
        elif i < self.cur:
            self.cur -= 1
        del self.tracks[i]
        save_library(self.tracks)
        self.menu.clamp(len(self.tracks))

    def play(self, i):
        tr = self.tracks[i]
        try:
            if tr.data is None:
                self.loading(tr.title)
                tr.data = analyze(tr.path)
            pygame.mixer.music.load(tr.path)
            pygame.mixer.music.play()
        except Exception as ex:
            print(f"Erro ao abrir {tr.path}: {ex}")
            self.say(f"ERRO: não foi possível abrir {tr.title}")
            self.remove(i)
            return False
        self.cur = i
        self.paused = False
        self.t_est = 0.0
        self.drop_i = 0
        self.vis.reset_fx()
        self.vis.pal.set(tr.data.palette)   # cada faixa escolhe sua paleta pelo timbre
        return True

    def step(self, delta):
        if self.tracks:
            self.play((self.cur + delta) % len(self.tracks))

    def add_dialog(self):
        paths = pick_files()
        if paths:
            self.add(paths)

    def toggle_menu(self):
        if self.menu.open:
            self.menu.open = False
        else:
            self.menu.open = True
            if self.cur >= 0:
                self.menu.sel = self.cur
            self.menu.clamp(len(self.tracks))

    # --- eventos -----------------------------------------------------------
    def menu_event(self, e):
        act = self.menu.handle(e, len(self.tracks))
        if not act:
            return
        if act[0] == "play":
            if self.play(act[1]):
                self.menu.open = False
        elif act[0] == "add":
            self.add_dialog()
        elif act[0] == "remove":
            self.remove(act[1])
        elif act[0] == "close":
            self.menu.open = False

    def key_event(self, e):
        if e.key == pygame.K_ESCAPE:
            self.running = False
        elif e.key in (pygame.K_m, pygame.K_TAB):
            self.toggle_menu()
        elif e.key == pygame.K_f:
            pygame.display.toggle_fullscreen()
        elif e.key == pygame.K_c:
            self.vis.pal.set(ORDER[1 - ORDER.index(self.vis.pal.name)])
        elif e.key == pygame.K_n:
            self.step(1)
        elif e.key == pygame.K_p:
            self.step(-1)
        elif e.key == pygame.K_SPACE and self.cur >= 0:
            self.paused = not self.paused
            if self.paused:
                pygame.mixer.music.pause()
            else:
                pygame.mixer.music.unpause()

    # --- loop --------------------------------------------------------------
    def run(self, initial):
        if initial:
            self.add(initial)
            if self.cur >= 0:
                self.menu.open = False

        idle = np.zeros(BANDS, dtype=np.float32)

        while self.running:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.1)
            self.status_t = max(0.0, self.status_t - dt)
            if self.status_t == 0.0:
                self.status = ""

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.running = False
                elif e.type == pygame.DROPFILE:
                    self.add(e.file)
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.start_rect.collidepoint(e.pos):
                    self.toggle_menu()          # botao "iniciar"
                elif self.menu.open:
                    self.menu_event(e)
                elif e.type == pygame.KEYDOWN:
                    self.key_event(e)

            # fim da faixa -> proxima da lista (da a volta no final)
            if self.cur >= 0 and not self.paused and not pygame.mixer.music.get_busy():
                self.step(1)

            calm_now = 0.0
            if self.cur >= 0:
                a = self.tracks[self.cur].data
                if not self.paused:
                    # relogio proprio, corrigido aos poucos pelo mixer -> sem "degraus" no movimento
                    pos = max(0, pygame.mixer.music.get_pos()) / 1000.0 + SYNC_OFFSET
                    self.t_est += dt
                    if abs(pos - self.t_est) > 0.25:
                        self.t_est = pos
                    else:
                        self.t_est += (pos - self.t_est) * min(1.0, dt * 8)
                t = self.t_est
                fi = t * a.sr / HOP
                b_now = sample(a.bands, fi)
                o_now = float(sample(a.onset, fi))
                bass = float(sample(a.bass, fi))
                treble = float(sample(a.treble, fi))
                calm_now = float(sample(a.calm, fi))
                if not self.paused:
                    while self.drop_i < len(a.drops) and t >= a.drops[self.drop_i]:
                        self.vis.drop()
                        self.drop_i += 1
            else:
                t = pygame.time.get_ticks() / 1000.0
                b_now, o_now, bass, treble = idle, 0.0, 0.0, 0.0

            if not self.paused:
                self.vis.update(dt, b_now, o_now, bass, calm_now)

            self.vis.render_scene(self.scene, t, bass, treble, 0 if self.paused else dt)
            self.vis.post(self.scene, self.screen, t, bass)

            title = self.tracks[self.cur].title if self.cur >= 0 else "NENHUMA MÚSICA"
            if self.vis.mode() != "black":
                self.draw_taskbar(self.screen, title)
            self.draw_osd(self.t_est if self.cur >= 0 else 0)

            if self.menu.open:
                self.menu.draw(self.screen, self.tracks, self.cur, self.paused, self.status)

            self.vis.vhs(self.screen, t, dt)
            pygame.display.flip()

        pygame.quit()


def main():
    App().run(sys.argv[1:])


if __name__ == "__main__":
    main()