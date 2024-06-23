import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
import math
from PIL import Image
import itertools
from functools import lru_cache


THOMAE_DEPTH = 40
MAX_KNOBS = 9
# ENABLE_FPS = True

SQRT3 = math.sqrt(3)
SQRT3_INV = 1/math.sqrt(3)
LOG2_5TH = math.log2(1.5)+1
LOG2_MAJ_3RD = math.log2(1.25)+1
LOG2_MIN_3RD = math.log2(1.2)+1
SQRT_9_8 = math.sqrt(9/8)
SQRT_3_2 = math.sqrt(3/2)
SQRT2 = math.sqrt(2)
SQRT_8_3 = math.sqrt(8/3)
SQRT_32_9 = math.sqrt(32/9)
F4_3 = 4/3
F16_9 = 16/9
# LOG2_PHI = 0.694241913631
PI_INV = 1/np.pi
TAU = 2*np.pi
MIDDLE_C = 220*(2**0.25)
MIDDLE_240 = 30*(2**3)

pygame.init()

ICON = pygame.image.load('assets/icon.png')
HE_DIAGRAM = pygame.image.load('assets/harmonic_entropy.png')
LEGEND_SHOW = pygame.image.load('assets/controls1.png')
LEGEND_HIDE = pygame.image.load('assets/controls2.png')
HUE_WHEEL = Image.open('assets/hue_wheel.png')

pygame.display.set_caption('chord space v0.5.4')
pygame.display.set_icon(ICON)
HEIGHT = 601
WIDTH = int((1+0.5*SQRT3)*HEIGHT)
HEIGHT_HALF = HEIGHT*0.5
KNOB_POS_1 = pygame.math.Vector2(HEIGHT*0.5 + 1, HEIGHT*0.5 - 1)
KNOB_POS_2 = pygame.math.Vector2(WIDTH*0.5 + 1, HEIGHT*0.5 - 1)
knob_center = KNOB_POS_1
window = pygame.display.set_mode((WIDTH, HEIGHT))  # flags=pygame.NOFRAME

font = pygame.font.Font('assets/JetBrainsMono-Regular.otf', 12)
# if ENABLE_FPS:
clock = pygame.time.Clock()

BUFFER_SIZE = 2048
SAMPLE_RATE = 192000
INV_SRATE = 1/44100
MAX_SAMPLE = 32767
t = np.arange(BUFFER_SIZE, dtype=np.float32)[:, np.newaxis]
# VOLUME = np.ones(MAX_KNOBS, dtype=np.float32) * MAX_SAMPLE*0.1
VOLUME = MAX_SAMPLE*0.1
previous_closest_knob = None
pygame.mixer.pre_init(SAMPLE_RATE, -16, 1, buffer=BUFFER_SIZE)
pygame.mixer.init()
buffer = np.zeros((BUFFER_SIZE, 2), dtype=np.int16)
sound = pygame.sndarray.make_sound(buffer)

# view_controls_text = 'SPACE :  view controls / info'
# legend_text = [
#     '       1-9 :  number of tones',
#     ' E/← · R/→ :  rotate',
#     '     D · F :  rotate by 3/2',
#     '     C · V :  rotate active by 3/2',
#     '         W :  spread equally',
#     '         Q :  randomize',
#     '         A :  randomize angle',
#     '         X :  alt display',
#     '         Z :  paint',
#     '         S :  bilateral guide',
#     '         H :  change hue',
#     '       Esc :  quit'
# ]


hue_width = HUE_WHEEL.size[0]
@lru_cache(maxsize=1530)
def get_hue_colors(num_colors, offset=0):
    colors = []
    # offset = width*np.random.rand()
    for i in range(num_colors):
        pos = round(offset + (hue_width-1)*i/num_colors) % (hue_width-1)
        color = np.array(HUE_WHEEL.getpixel((pos, 0)))
        colors.append(color.astype(np.uint8))
    return colors

def thomaes_function():
    numer = [1]
    denom = [1]
    for d in range(2, THOMAE_DEPTH + 1):
        for n in range(d+1, d*2):
            gcd = math.gcd(n, d)

            if gcd == 1:
                numer.append(n)
                denom.append(d)
    numer = np.array(numer[::-1])
    denom = np.array(denom[::-1])
    return (np.log2(numer/denom), 1.0/denom)

@lru_cache(maxsize=131072)
def octave_reduce(x):
    return 2 ** (math.log2(x) % 1)

@lru_cache(maxsize=131072)
def pitch_class_distance(a, b):
    value = abs(a-b)
    return value
    # if value <= 0.5:
    #     return value
    # else:
    #     return 1-value

@lru_cache(maxsize=131072)
def interval_sign(x):
    x = x%1 + 1
    e = 0.0003
    if 1+e<=x<=SQRT_9_8-e:
        return (-1, 'm1')
    elif 1.125+e<=x<=SQRT_3_2-e:
        return (-1, 'm3')
    elif F4_3+e<=x<=SQRT2-e:
        return (-1, 'm5')
    elif 1.5+e<=x<=SQRT_8_3-e:
        return (-1, 'm7')
    elif F16_9+e<=x<=SQRT_32_9-e:
        return (-1, 'm9')
    elif SQRT_9_8+e<=x<=1.125-e:
        return (1, 'M1')
    elif SQRT_3_2+e<=x<=F4_3-e:
        return (1, 'M3')
    elif SQRT2+e<=x<=1.5-e:
        return (1, 'M5')
    elif SQRT_8_3+e<=x<=F16_9-e:
        return (1, 'M7')
    elif SQRT_32_9+e<=x<=2-e:
        return (1, 'M9')
    elif SQRT_9_8-e<x<SQRT_9_8+e:
        return (2, 'n1')
    elif SQRT_3_2-e<x<SQRT_3_2+e:
        return (2, 'n3')
    elif SQRT2-e<x<SQRT2+e:
        return (2, 'n5')
    elif SQRT_8_3-e<x<SQRT_8_3+e:
        return (2, 'n7')
    elif SQRT_32_9-e<x<SQRT_32_9+e:
        return (2, 'n9')
    elif 1<x<1+e or 2-e<x<2:
        return (0, '.0')
    elif 1.125-e<x<1.125+e:
        return (0, 'p2')
    elif F4_3-e<x<F4_3+e:
        return (0, 'p4')
    elif 1.5-e<x<1.5+e:
        return (0, 'p6')
    elif F16_9-e<x<F16_9+e:
        return (0, 'p8')
    return (0, '  ')

# def find_closest_knob(mouse_pos, knob_center, knob_values):
#     log_knob_vals = np.log2(knob_values%1 + 1)
#     angles = log_knob_vals * TAU
#     knob_positions = np.array([
#         knob_center.x + HEIGHT*0.25 * np.sin(angles),
#         knob_center.y - HEIGHT*0.25 * np.cos(angles)
#     ]).T
#     distances = np.sum((knob_positions - mouse_pos)**2, axis=1)
#     return np.argmin(distances)

THOMAE = thomaes_function()
THOMAE_0_SIZE = THOMAE[0].size
WHITE = np.array((255,255,255))
hue_colors = {i: get_hue_colors(i) for i in range(1, MAX_KNOBS + 1)}
circle_surface = pygame.Surface((5, 5), pygame.SRCALPHA)
pygame.draw.circle(circle_surface, (255, 255, 255), (2, 2), 2)
middle_c = True
init_pos = np.array((0, 0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875, 0.0625))
num_knobs = 3
knob_values = np.array(init_pos[:num_knobs])
phase_vectors = np.ones(num_knobs, dtype=np.complex128) * 1j
active_line_index = 0
hue_offset = 0
painting, bilateral_guide, ghost_lines, mouse_held, show_controls = False,False,False,False,False


running = True
while running:
    keys = pygame.key.get_pressed()

    BASE_FREQ = MIDDLE_C if middle_c else MIDDLE_240
    hue_offset_old = hue_offset

    rot_speed = 1
    hue_speed = 1
    rot_interval = LOG2_5TH
    if keys[pygame.K_LSHIFT] or keys[pygame.K_LSHIFT]:
        rot_speed = 2.75
        hue_speed = -1
        rot_interval = LOG2_MAJ_3RD
    elif keys[pygame.K_LCTRL] or keys[pygame.K_LCTRL]:
        rot_speed = 0.025
        hue_speed = 8
        rot_interval = LOG2_MIN_3RD
    
    if keys[pygame.K_h]:
        hue_offset += hue_speed*10

    if keys[pygame.K_q]:
        knob_values = np.exp2((np.log2(knob_values%1 + 1) + np.random.rand(num_knobs)) % 1) - 1
    if keys[pygame.K_a]:
        knob_values = np.exp2((np.log2(knob_values%1 + 1) + np.random.rand()) % 1) - 1

    if keys[pygame.K_e] or keys[pygame.K_LEFT]:
        knob_values = np.exp2((np.log2(knob_values%1 + 1) + rot_speed*-0.003) % 1) - 1
    if keys[pygame.K_r] or keys[pygame.K_RIGHT]:
        knob_values = np.exp2((np.log2(knob_values%1 + 1) + rot_speed*0.003) % 1) - 1

    if keys[pygame.K_KP0]:
        active_line_index = np.random.randint(num_knobs+1)
        active_line_index = min(active_line_index, num_knobs-1)

    knob_center = KNOB_POS_1 if num_knobs>=3 else KNOB_POS_2

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
            
        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_d:
                knob_values = np.exp2((np.log2(knob_values%1 + 1) - rot_interval) % 1) - 1   
            elif event.key == pygame.K_f:
                knob_values = np.exp2((np.log2(knob_values%1 + 1) + rot_interval) % 1) - 1
            elif event.key == pygame.K_c or event.key == pygame.K_DOWN:
                knob_values[active_line_index] = np.exp2((np.log2(knob_values[active_line_index]%1 + 1) - rot_interval) % 1) - 1
            elif event.key == pygame.K_v or event.key == pygame.K_UP:
                knob_values[active_line_index] = np.exp2((np.log2(knob_values[active_line_index]%1 + 1) + rot_interval) % 1) - 1
            elif event.key == pygame.K_SPACE:
                show_controls = not show_controls
            elif event.key == pygame.K_b:
                middle_c = not middle_c
            elif event.key == pygame.K_z:
                window.fill((0,0,0))
                if 3 <= num_knobs:
                    window.blit(HE_DIAGRAM, (HEIGHT+1, 0))
                painting = True
            elif event.key == pygame.K_x:
                ghost_lines = not ghost_lines
            elif event.key == pygame.K_s:
                bilateral_guide = not bilateral_guide
            elif event.key == pygame.K_w:
                knob_values = np.exp2((np.linspace(1, 2, num_knobs, endpoint=False) + np.random.rand())%1) - 1

            # if event.key == pygame.K_g:
            #     knob_values = np.nan_to_num(2 / knob_values)
                

            match event.key:
                case pygame.K_KP1 | pygame.K_KP2 | pygame.K_KP3 | pygame.K_KP4 | pygame.K_KP5 | pygame.K_KP6 | pygame.K_KP7 | pygame.K_KP8 | pygame.K_KP9:
                    active_line_index = event.key - pygame.K_KP1
                    active_line_index = min(active_line_index, num_knobs-1)
                case pygame.K_KP0:
                    active_line_index = np.random.randint(num_knobs+1)
                    active_line_index = min(active_line_index, num_knobs-1)

                case pygame.K_1 | pygame.K_2 | pygame.K_3 | pygame.K_4 | pygame.K_5 | pygame.K_6 | pygame.K_7 | pygame.K_8 | pygame.K_9:
                    num_knobs_old = num_knobs
                    num_knobs = event.key - pygame.K_0

                    knob_values = np.exp2((np.log2(init_pos[:num_knobs]%1 + 1))%1) - 1 # + np.random.rand()

                    phase_vectors_old = phase_vectors
                    if num_knobs_old < num_knobs:
                        phase_vectors = np.ones(num_knobs, dtype=complex)
                        phase_vectors[:num_knobs_old] = phase_vectors_old
                    else:
                        phase_vectors = phase_vectors_old[:num_knobs]
                    
                    active_line_index = min(active_line_index, num_knobs-1)
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_z:
                painting = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                # left click
                mouse_held = True
                mouse_pos = event.pos
                closest_line_index = None
                min_distance = float('inf')
                for i, value in enumerate(knob_values):
                    log_value = np.log2(value%1+1) * TAU
                    line_ctr = knob_center + pygame.math.Vector2(HEIGHT*0.25 * np.sin(log_value),
                                                                -HEIGHT*0.25 * np.cos(log_value))
                    dist = line_ctr.distance_to(mouse_pos)
                    if dist < min_distance:
                        min_distance = dist
                        closest_line_index = i
                active_line_index = closest_line_index
            elif event.button == 3:
                # right click
                if pygame.key.get_mods() & pygame.KMOD_CTRL:
                    # ctrl + right click
                    mouse_pos = event.pos
                    closest_line_index = None
                    min_distance = float('inf')
                    for i, value in enumerate(knob_values):
                        log_value = np.log2(value%1+1) * TAU
                        line_ctr = knob_center + pygame.math.Vector2(HEIGHT*0.25 * np.sin(log_value),
                                                                    -HEIGHT*0.25 * np.cos(log_value))
                        dist = line_ctr.distance_to(mouse_pos)
                        if dist < min_distance:
                            min_distance = dist
                            closest_line_index = i
                    if closest_line_index is not None and num_knobs > 1:
                        knob_values = np.delete(knob_values, closest_line_index)
                        phase_vectors = np.delete(phase_vectors, closest_line_index)
                        num_knobs -= 1
                        active_line_index = min(active_line_index, num_knobs-1)
                elif num_knobs+1 <= MAX_KNOBS:
                    mouse_pos = event.pos
                    dx, dy = pygame.math.Vector2(mouse_pos) - knob_center
                    angle = math.atan2(dy, dx)*0.5*PI_INV
                    new_value = np.exp2((angle+0.25)%1)-1
                    knob_values = np.append(knob_values, new_value)
                    phase_vectors = np.append(phase_vectors, 1j)
                    num_knobs += 1
        elif event.type == pygame.MOUSEMOTION:
            if event.buttons[0] == 1:
                mouse_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_held = False
    
    if mouse_held:
        dx, dy = pygame.math.Vector2(mouse_pos) - knob_center
        angle = math.atan2(dy, dx)*0.5*PI_INV
        knob_values[active_line_index] = np.exp2((angle+0.25)%1)-1

    if not painting:
        window.fill((0,0,0)) # rect=[(0,0),(height, height)]

    if hue_offset != hue_offset_old:
        hue_colors[num_knobs] = get_hue_colors(num_knobs, hue_offset)
    colors = hue_colors[num_knobs]


    if 3 <= num_knobs:
        if not painting:
            window.blit(HE_DIAGRAM, (HEIGHT+1, 0))
            
        if bilateral_guide and not painting:
            line_surface = pygame.Surface((HEIGHT, HEIGHT))
            pygame.draw.aaline(line_surface, (60,16,55), (0,HEIGHT-2), (SQRT3_INV*(HEIGHT-2), 1), 1)
            window.blit(line_surface, (HEIGHT+2, 1), special_flags=pygame.BLEND_RGB_ADD)

        subsets = list(itertools.combinations(set(knob_values), 3))

        tinted_circle = circle_surface.copy()
        tinted_circle.fill(colors[0], special_flags=pygame.BLEND_RGBA_MULT)

        for i in range(len(subsets)):
            a, b, c =  subsets[i][0], subsets[i][1], subsets[i][2]
            M = [octave_reduce(a+1), octave_reduce(b+1), octave_reduce(c+1)]
            sorted_M = sorted(M)
            u, v = np.log2(sorted_M[2]/sorted_M[0]) + 1, np.log2(sorted_M[1]/sorted_M[0]) + 1

            if (v >= 2*u-2) and (u+v <= 3):
                point_x = SQRT3 * (v-1) * (HEIGHT-1) + HEIGHT+1
                point_y = HEIGHT - (2*u-v-1) * (HEIGHT-1)
            elif (u+v > 3) and (2*v-u >= 1):
                point_x = SQRT3 * (u-v) * (HEIGHT-1) + HEIGHT+1
                point_y = HEIGHT - (4-u-v) * (HEIGHT-1)
            elif (v < 2*u-2) and (2*v-u < 1):
                point_x = SQRT3 * (2-u) * (HEIGHT-1) + HEIGHT+1
                point_y = HEIGHT - (2*v-u) * (HEIGHT-1)
            
            window.blit(tinted_circle, (int(point_x), int(point_y-2)))

    if show_controls and not painting:
        window.blit(LEGEND_SHOW, (WIDTH-232, HEIGHT-166))

    log_knob_vals = np.log2(knob_values%1 + 1)
    start_index = round(THOMAE_0_SIZE - THOMAE_0_SIZE/num_knobs)

    if num_knobs == 1:
        pos = THOMAE[0]
        val = THOMAE[1]

        angles = (pos+log_knob_vals) * TAU

        coords_x = knob_center.x + HEIGHT_HALF * np.sin(angles)
        coords_y = knob_center.y - HEIGHT_HALF * np.cos(angles)

        if ghost_lines:
            for i in range(THOMAE_0_SIZE-1):
                color = val[i]*255
                pygame.draw.line(window, (color,color,color), knob_center, (coords_x[i], coords_y[i]))
            
            pygame.draw.aaline(window, tuple(np.ceil(val[-1]*WHITE)), knob_center, (coords_x[-1], coords_y[-1]))
        else:
            coords_x1 = knob_center.x + HEIGHT*0.35 * np.sin(angles)
            coords_y1 = knob_center.y - HEIGHT*0.35 * np.cos(angles)
            
            for i in range(THOMAE_0_SIZE-1):
                color = val[i]*255
                pygame.draw.line(window, (color,color,color), (coords_x[i], coords_y[i]), (coords_x1[i], coords_y1[i]))

            pygame.draw.aaline(window, tuple(np.ceil(val[-1]*WHITE)), knob_center, (coords_x1[-1], coords_y1[-1]))
    else:
        indices = np.arange(start_index, THOMAE_0_SIZE)
        pos = THOMAE[0][indices]
        val = THOMAE[1][indices]

        angles = (pos[:,np.newaxis]+log_knob_vals) * TAU

        coords_x = knob_center.x + HEIGHT_HALF * np.sin(angles)
        coords_y = knob_center.y - HEIGHT_HALF * np.cos(angles)

        line_widths = np.power(255/val, 0.1).astype(int)

        if ghost_lines:
            for i in range(start_index, THOMAE_0_SIZE-1):
                for j in range(num_knobs):
                    color = tuple(np.ceil(val[i-start_index] * colors[j]))
                    pygame.draw.line(window, color, knob_center, (coords_x[i-start_index, j], coords_y[i-start_index, j]), line_widths[i-start_index])
            for j in range(num_knobs):
                color = tuple(np.ceil(val[-1] * colors[j]))
                pygame.draw.aaline(window, color, knob_center, (coords_x[-1, j], coords_y[-1, j]))
        else:
            coords_x1 = knob_center.x + HEIGHT*0.35 * np.sin(angles)
            coords_y1 = knob_center.y - HEIGHT*0.35 * np.cos(angles)

            for i in range(start_index, THOMAE_0_SIZE-1):
                for j in range(num_knobs):
                    color = tuple(np.ceil(val[i-start_index] * colors[j]))
                    pygame.draw.line(window, color, (coords_x1[i-start_index, j], coords_y1[i-start_index, j]), (coords_x[i-start_index, j], coords_y[i-start_index, j]), line_widths[i-start_index])
            for j in range(num_knobs):
                color = tuple(np.ceil(val[-1] * colors[j]))
                pygame.draw.aaline(window, color, knob_center, (coords_x1[-1, j], coords_y1[-1, j]))


    sorted_indices = knob_values.argsort()
    
    freqs = BASE_FREQ * (knob_values[sorted_indices]%1 + 1)
    sorted_colors = np.array(colors)[sorted_indices]

    if not painting:
        if show_controls:
            if num_knobs == 1:
                text_freq = font.render(f'{freqs[0]:.5f} Hz', True, (255, 255, 255))
                window.blit(text_freq, (79, HEIGHT-16))
            else:
                for i in range(num_knobs):
                    text_freq = font.render(f'{freqs[i]:.5f} Hz', True, sorted_colors[i])
                    window.blit(text_freq, (79, HEIGHT + (i-num_knobs)*15 - 1))

                intervals = list(itertools.combinations(set(np.log2(knob_values%1 + 1)), 2))
                distances = sorted([pitch_class_distance(i[0], i[1]) for i in intervals])
                for i, d in enumerate(distances):
                    rot_speed, degree = interval_sign(np.exp2(d))
                        
                    if rot_speed == 1:
                        text_intervals = font.render(f'{degree} {d:.4f}', True, (255, 101, 187))
                    elif rot_speed == -1:
                        text_intervals = font.render(f'{degree} {d:.4f}', True, (0, 220, 176))
                    elif rot_speed == 2:
                        text_intervals = font.render(f'{degree:} {d:.4f}', True, (155, 158, 255))
                    else:
                        text_intervals = font.render(f'{degree:} {d:.4f}', True, (206, 206, 206))
                    window.blit(text_intervals, (2, HEIGHT + (i-len(distances))*15 - 1))

        # for i, text in enumerate(legend_text):
        #     legend_surface = font.render(text, True, (158, 155, 157))
        #     window.blit(legend_surface, (width-241, height-len(legend_text)*14 + i*14 - 3))

        else:
            window.blit(LEGEND_HIDE, (WIDTH-205, HEIGHT-13))
            # legend_surface = font.render(view_controls_text, True, (158, 155, 157))
            # window.blit(legend_surface, )

        # if ENABLE_FPS and show_controls:
        if show_controls:
            fps = clock.get_fps()
            fps_surface = font.render(f'{fps:.2f} fps', True, (158, 155, 157))
            window.blit(fps_surface, (2,-4))

    pygame.display.flip()

    # alt_held = keys[pygame.K_LALT] or keys[pygame.K_RALT]
    # if alt_held:
    #     mouse_pos = pygame.mouse.get_pos()
    #     closest_knob = find_closest_knob(mouse_pos, knob_center, knob_values)
    #     VOLUME[:] = 0
    #     VOLUME[closest_knob] = MAX_SAMPLE * 0.1
    # else:
    #     VOLUME[:] = MAX_SAMPLE * 0.1

    if pygame.mixer.Channel(0).get_queue() is None:
        freqs = BASE_FREQ * (knob_values%1 + 1)
        angles = (freqs*TAU * INV_SRATE).astype(np.float32) * 1j
        rotation_vectors = np.exp(angles)

        all_phase_vectors = phase_vectors * np.exp(np.outer(t, angles))
        np.multiply(phase_vectors, rotation_vectors ** BUFFER_SIZE, out=phase_vectors)
        # wave = np.sum(all_phase_vectors.imag * VOLUME[:num_knobs, np.newaxis].T, axis=1)
        wave = np.sum(all_phase_vectors.imag * VOLUME, axis=1)
        # if alt_held and previous_closest_knob is not None and closest_knob != previous_closest_knob:
        #     wave[-32:] *= np.linspace(1, 0, 32)
        #     wave[:32] *= np.linspace(0, 1, 32)
        buffer_mono = np.rint(wave).astype(np.int16)
        buffer[:] = np.column_stack((buffer_mono, buffer_mono))

    sound = pygame.sndarray.make_sound(buffer)
    pygame.mixer.Channel(0).queue(sound)

    # previous_closest_knob = closest_knob if alt_held else None

    # if ENABLE_FPS:
    clock.tick(60)

pygame.quit()