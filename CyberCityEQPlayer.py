"""
CyberCity EQ

Comandos:
  M / TAB   abre/fecha o menu da biblioteca
  space     pausa
  N / P     proxima / anterior
  F         fullscreen
  ESC       fecha o menu (ou sai, se o menu estiver fechado)

No menu:
  A / O     adicionar musicas      DEL  remove da lista
  ENTER     toca a selecionada     setas / scroll navegam
  Voce tambem pode arrastar arquivos ou pastas para a janela.

A biblioteca e salva automaticamente em cybercity_library.json (ao lado do
script) e recarregada na proxima vez que o player abrir.

Uso: python cybercity_eq.py [arquivo1 arquivo2 ...]
"""
import json
import math
import os
import random
import sys

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

AUDIO_EXTS = (".mp3", ".wav", ".ogg", ".flac")
LIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cybercity_library.json")

CYAN = (0, 235, 255)
MAGENTA = (255, 40, 190)
PURPLE = (130, 60, 255)
SUN_TOP = (255, 225, 90)
SUN_BOTTOM = (255, 30, 140)


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
    return bands, onset, bass, treble, sr


# ----------------------------------------------------------------------------
# Visual
# ----------------------------------------------------------------------------
class Rain:
    def __init__(self, n=150):
        self.drops = [
            [random.uniform(0, W), random.uniform(0, H), random.uniform(0.6, 1.2), random.randint(8, 22)]
            for _ in range(n)
        ]

    def update_draw(self, surf, dt, treble):
        speed = 700 + treble * 1800
        for d in self.drops:
            d[1] += speed * d[2] * dt
            d[0] -= 60 * dt
            if d[1] > H:
                d[0] = random.uniform(0, W + 80)
                d[1] = -d[3]
            col = lerp((20, 60, 110), (110, 220, 255), d[2] - 0.6)
            pygame.draw.line(surf, col, (d[0], d[1]), (d[0] + 4, d[1] + d[3]), 1)


def draw_sun(s, bass):
    cx, cy = W // 2, HORIZON - 60
    r = int(150 + bass * 55)
    for y in range(cy - r, HORIZON):
        dy = y - cy
        half = math.sqrt(max(0, r * r - dy * dy))
        if half <= 0:
            continue
        if dy > 0 and (dy % 16) < (dy / r) * 11:  # listras do sol
            continue
        col = lerp(SUN_TOP, SUN_BOTTOM, (y - (cy - r)) / (2 * r))
        pygame.draw.line(s, col, (cx - half, y), (cx + half, y))


class Visualizer:
    def __init__(self):
        self.h = np.zeros(BANDS)
        self.peak = np.zeros(BANDS)
        self.scroll = 0.0
        self.kick = 0.0
        self.glitch = 0.0
        self.cooldown = 0.0
        self.rain = Rain()

        order = sorted(range(BANDS), key=lambda p: abs(p - (BANDS - 1) / 2))
        self.band_at = [0] * BANDS
        for k, pos in enumerate(order):
            self.band_at[pos] = k
        self.base = [random.uniform(0.72, 1.0) for _ in range(BANDS)]
        self.cell_seed = [random.random() for _ in range(BANDS)]

        self.bg = pygame.Surface((W, H)).convert()
        for y in range(H):
            t = y / HORIZON
            self.bg.fill(lerp((6, 0, 20), (70, 10, 90), t ** 1.5), (0, y, W, 1))
        self.scan = pygame.Surface((W, H), pygame.SRCALPHA)
        for y in range(0, H, 3):
            pygame.draw.line(self.scan, (0, 0, 0, 70), (0, y), (W, y))

    def update(self, dt, bands_now, onset_now, bass_now):
        target = bands_now
        up = target > self.h
        self.h = np.where(up, self.h + (target - self.h) * 0.6, self.h + (target - self.h) * 0.12)
        self.peak = np.maximum(self.peak - dt * 0.35, self.h)
        self.kick = max(self.kick * 0.86, onset_now)
        self.cooldown -= dt
        if onset_now > 0.72 and self.cooldown <= 0:
            self.glitch = 1.0
            self.cooldown = 0.25
        self.glitch *= 0.84
        energy = float(self.h.mean())
        self.scroll += dt * (0.25 + energy * 4.0 + bass_now * 2.0)

    def draw_floor(self, s, bass):
        pygame.draw.rect(s, (12, 0, 28), (0, HORIZON, W, H - HORIZON))
        glow = 0.55 + bass * 0.9

        for i in range(-22, 23):
            xb = W / 2 + i * 120
            pygame.draw.line(s, scale(MAGENTA, 0.45 * glow), (W / 2 + i * 6, HORIZON), (xb, H), 1)

        n = 14
        for j in range(n):
            k = ((j + self.scroll) % n) / n
            y = HORIZON + (H - HORIZON) * k ** 2.2
            pygame.draw.line(s, scale(MAGENTA, (0.25 + k) * glow), (0, y), (W, y), 1 + int(k * 2))
        pygame.draw.line(s, scale(CYAN, 1.0), (0, HORIZON), (W, HORIZON), 2)

    def draw_city(self, s, t):
        bw = W // BANDS
        x0 = (W - bw * BANDS) // 2
        max_h = HORIZON * 0.82
        for pos in range(BANDS):
            k = self.band_at[pos]
            level = float(self.h[k])
            height = 36 + level * max_h * self.base[pos]
            x = x0 + pos * bw
            top = HORIZON - height
            col = lerp(MAGENTA, CYAN, k / (BANDS - 1))

            pygame.draw.rect(s, (8, 4, 22), (x + 1, top, bw - 2, height))
            pygame.draw.rect(s, col, (x + 1, top, bw - 2, height), 2)

            # janelas acendem de baixo pra cima conforme a energia
            rows = int(height / 12)
            for r in range(rows):
                if random.random() < 0.04:  # flicker
                    continue
                y = HORIZON - 14 - r * 12
                bright = 0.35 + 0.65 * (r / max(1, rows))
                wc = lerp(scale(col, bright), (255, 255, 255), level * 0.35 * (r / max(1, rows)))
                for cx in (x + 5, x + bw // 2 + 1):
                    pygame.draw.rect(s, wc, (cx, y, 6, 6))

            py = HORIZON - (36 + float(self.peak[k]) * max_h * self.base[pos])
            pygame.draw.line(s, (255, 255, 255), (x + 1, py), (x + bw - 2, py), 2)

            if self.cell_seed[pos] > 0.78:
                pygame.draw.line(s, col, (x + bw // 2, top), (x + bw // 2, top - 18), 1)
                if int(t * 2 + self.cell_seed[pos] * 10) % 2 == 0:
                    pygame.draw.circle(s, (255, 40, 60), (x + bw // 2, int(top - 19)), 2)

    def render_scene(self, scene, t, bass, treble, dt):
        scene.blit(self.bg, (0, 0))
        draw_sun(scene, bass)
        self.draw_floor(scene, bass)
        self.draw_city(scene, t)
        self.rain.update_draw(scene, dt, treble)

    def post(self, scene, screen, bass):
        # glow: versao borrada somada por cima
        small = pygame.transform.smoothscale(scene, (W // 6, H // 6))
        glow = pygame.transform.smoothscale(small, (W, H))
        v = int(110 + 145 * min(1.0, bass * 1.3))
        glow.fill((v, v, v), special_flags=pygame.BLEND_MULT)
        scene.blit(glow, (0, 0), special_flags=pygame.BLEND_ADD)

        ab = int(2 + self.kick * 10 + self.glitch * 10)
        screen.fill((0, 0, 0))
        for color, off in (((255, 0, 0), -ab), ((0, 255, 0), 0), ((0, 0, 255), ab)):
            layer = scene.copy()
            layer.fill(color, special_flags=pygame.BLEND_MULT)
            screen.blit(layer, (off, 0), special_flags=pygame.BLEND_ADD)
        if self.glitch > 0.12:
            src = screen.copy()
            for _ in range(int(self.glitch * 9)):
                y = random.randint(0, H - 60)
                hh = random.randint(6, 46)
                dx = random.randint(-70, 70)
                screen.blit(src, (dx, y), (0, y, W, hh))

        screen.blit(self.scan, (0, 0))


# ----------------------------------------------------------------------------
# Menu da biblioteca
# ----------------------------------------------------------------------------
class Menu:
    ROW_H = 34

    def __init__(self):
        self.open = True
        self.sel = 0
        self.scroll = 0
        self.panel = pygame.Rect(W // 2 - 380, 80, 760, 540)
        self.list_rect = pygame.Rect(self.panel.x + 20, self.panel.y + 70, self.panel.w - 40, self.ROW_H * 10)
        self.visible = self.list_rect.h // self.ROW_H
        self.btn_add = pygame.Rect(self.panel.x + 20, self.panel.bottom - 80, 240, 40)
        self.btn_close = pygame.Rect(self.panel.right - 20 - 160, self.panel.bottom - 80, 160, 40)

        self.back = pygame.Surface((W, H), pygame.SRCALPHA)
        self.back.fill((0, 0, 0, 165))
        self.box = pygame.Surface(self.panel.size, pygame.SRCALPHA)
        self.box.fill((8, 4, 22, 238))

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
                    return "play", self.sel
            elif e.key in (pygame.K_a, pygame.K_o):
                return ("add",)
            elif e.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
                if n:
                    return "remove", self.sel
            elif e.key in (pygame.K_m, pygame.K_TAB, pygame.K_ESCAPE):
                return ("close",)
            self.clamp(n)
        elif e.type == pygame.MOUSEWHEEL:
            self.scroll = max(0, min(self.scroll - e.y, max(0, n - self.visible)))
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if self.btn_add.collidepoint(e.pos):
                return ("add",)
            if self.btn_close.collidepoint(e.pos):
                return ("close",)
            if self.list_rect.collidepoint(e.pos):
                i = self.scroll + (e.pos[1] - self.list_rect.y) // self.ROW_H
                if i < n:
                    self.sel = i
                    return ("play", i)
        return None

    @staticmethod
    def _button(screen, font, rect, label, color):
        pygame.draw.rect(screen, (20, 6, 44), rect)
        pygame.draw.rect(screen, color, rect, 2)
        txt = font.render(label, True, color)
        screen.blit(txt, txt.get_rect(center=rect.center))

    def draw(self, screen, font, big, tracks, current, paused, status):
        n = len(tracks)
        screen.blit(self.back, (0, 0))
        screen.blit(self.box, self.panel.topleft)
        pygame.draw.rect(screen, MAGENTA, self.panel, 2)
        pygame.draw.rect(screen, CYAN, self.panel.inflate(-8, -8), 1)

        screen.blit(big.render("> BIBLIOTECA", True, CYAN), (self.panel.x + 22, self.panel.y + 20))

        if not tracks:
            for k, line in enumerate(("Nenhuma musica ainda.",
                                      "Clique em ADICIONAR ou arraste arquivos/pastas para a janela.")):
                txt = font.render(line, True, (170, 170, 210))
                screen.blit(txt, txt.get_rect(center=(self.list_rect.centerx, self.list_rect.centery - 12 + k * 26)))

        for row in range(self.visible):
            i = self.scroll + row
            if i >= n:
                break
            r = pygame.Rect(self.list_rect.x, self.list_rect.y + row * self.ROW_H, self.list_rect.w - 14, self.ROW_H - 4)
            if i == self.sel:
                pygame.draw.rect(screen, (40, 10, 70), r)
                pygame.draw.rect(screen, MAGENTA, r, 1)
            playing = i == current
            mark = ("|| " if paused else ">> ") if playing else "   "
            col = CYAN if playing else (200, 200, 230)
            text = fit(font, f"{mark}{i + 1:02d}  {tracks[i].title}", r.w - 20)
            screen.blit(font.render(text, True, col), (r.x + 10, r.y + 7))

        if n > self.visible:  # barra de rolagem
            bar = pygame.Rect(self.list_rect.right - 8, self.list_rect.y, 6, self.list_rect.h)
            pygame.draw.rect(screen, (40, 20, 70), bar)
            hh = max(24, bar.h * self.visible // n)
            yy = bar.y + (bar.h - hh) * self.scroll // max(1, n - self.visible)
            pygame.draw.rect(screen, MAGENTA, (bar.x, yy, bar.w, hh))

        info = status if status else f"{n} musica(s) na biblioteca"
        screen.blit(font.render(info, True, MAGENTA if status else (140, 140, 180)),
                    (self.panel.x + 22, self.list_rect.bottom + 14))

        self._button(screen, font, self.btn_add, "+ ADICIONAR [A]", CYAN)
        self._button(screen, font, self.btn_close, "FECHAR [M]", MAGENTA)
        hint = "ENTER/clique toca   DEL remove   setas navegam   arraste arquivos aqui"
        screen.blit(font.render(hint, True, (120, 120, 160)), (self.panel.x + 22, self.panel.bottom - 30))


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
class App:
    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 2, 1024)
        pygame.init()
        self.screen = pygame.display.set_mode((W, H), pygame.SCALED | pygame.RESIZABLE)
        pygame.display.set_caption("NEON SKYLINE")
        self.scene = pygame.Surface((W, H)).convert()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas,couriernew,monospace", 16)
        self.big = pygame.font.SysFont("consolas,couriernew,monospace", 26, bold=True)

        self.vis = Visualizer()
        self.menu = Menu()
        self.tracks, missing = load_library()   # biblioteca salva
        self.cur = -1      # indice da faixa tocando
        self.paused = False
        self.running = True
        self.status = ""
        self.status_t = 0.0
        if missing:
            self.say(f"{missing} arquivo(s) da biblioteca nao foram encontrados e sairam da lista.", 6.0)
            save_library(self.tracks)

    # --- helpers -----------------------------------------------------------
    def say(self, msg, secs=4.0):
        self.status, self.status_t = msg, secs

    def loading(self, title):
        self.screen.fill((6, 0, 20))
        a = self.big.render("ANALISANDO AUDIO...", True, CYAN)
        b = self.font.render(fit(self.font, title, W - 80), True, MAGENTA)
        self.screen.blit(a, a.get_rect(center=(W // 2, H // 2 - 20)))
        self.screen.blit(b, b.get_rect(center=(W // 2, H // 2 + 20)))
        pygame.display.flip()
        pygame.event.pump()

    # --- biblioteca --------------------------------------------------------
    def add(self, paths):
        if isinstance(paths, str):
            paths = [paths]
        added = add_paths(self.tracks, paths)
        if not added:
            self.say("Nada novo adicionado (formato invalido ou ja estava na lista).")
            return
        self.menu.sel = self.tracks.index(added[0])
        self.menu.clamp(len(self.tracks))
        save_library(self.tracks)
        self.say(f"{len(added)} musica(s) adicionada(s).")
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
            self.say(f"ERRO: nao foi possivel abrir {tr.title}")
            self.remove(i)
            return False
        self.cur = i
        self.paused = False
        return True

    def step(self, delta):
        if self.tracks:
            self.play((self.cur + delta) % len(self.tracks))

    def add_dialog(self):
        paths = pick_files()
        if paths:
            self.add(paths)

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
            self.menu.open = True
            if self.cur >= 0:
                self.menu.sel = self.cur
            self.menu.clamp(len(self.tracks))
        elif e.key == pygame.K_f:
            pygame.display.toggle_fullscreen()
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
                elif self.menu.open:
                    self.menu_event(e)
                elif e.type == pygame.KEYDOWN:
                    self.key_event(e)

            # fim da faixa -> proxima da lista (da a volta no final)
            if self.cur >= 0 and not self.paused and not pygame.mixer.music.get_busy():
                self.step(1)

            if self.cur >= 0:
                bands, onset, bass_arr, treble_arr, sr = self.tracks[self.cur].data
                t = max(0, pygame.mixer.music.get_pos()) / 1000.0
                idx = min(bands.shape[1] - 1, int(t * sr / HOP))
                b_now, o_now = bands[:, idx], float(onset[idx])
                bass, treble = float(bass_arr[idx]), float(treble_arr[idx])
            else:
                t = pygame.time.get_ticks() / 1000.0
                b_now, o_now, bass, treble = idle, 0.0, 0.0, 0.0

            if not self.paused:
                self.vis.update(dt, b_now, o_now, bass)

            self.vis.render_scene(self.scene, t, bass, treble, 0 if self.paused else dt)
            self.vis.post(self.scene, self.screen, bass)

            title = self.tracks[self.cur].title if self.cur >= 0 else "NENHUMA MUSICA"
            hud = self.font.render(f"> {fit(self.font, title, W - 40)}", True, CYAN)
            self.screen.blit(hud, (16, 12))
            keys = "[M] menu  [ESPACO] pausa  [N/P] proxima/anterior  [F] tela cheia  [ESC] sair"
            self.screen.blit(self.font.render(keys, True, (110, 150, 170)), (16, 34))
            if self.status and not self.menu.open:
                self.screen.blit(self.font.render(self.status, True, MAGENTA), (16, 56))

            if self.menu.open:
                self.menu.draw(self.screen, self.font, self.big, self.tracks, self.cur, self.paused, self.status)

            pygame.display.flip()

        pygame.quit()


def main():
    App().run(sys.argv[1:])


if __name__ == "__main__":
    main()