import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
import math
from PIL import Image
import itertools

BASE_FREQ = 220*(2**(1/4))  # 30*(2**3)
DEPTH = 35
MAX_KNOBS = 7
ENABLE_FPS = True
ENABLE_DETAIL = False

SQRT3 = math.sqrt(3)
SQRT3_INV = 1/math.sqrt(3)
LOG2_5TH = math.log2(1.5)+1
LOG2_MAJ_3RD = math.log2(1.25)+1
LOG2_MIN_3RD = math.log2(1.2)+1
# LOG2_PHI = 0.694241913631
PI_INV = 1/np.pi
TAU = 2*np.pi

pygame.init()
pygame.display.set_caption('v0.5.2')
if ENABLE_FPS:
    clock = pygame.time.Clock()
font = pygame.font.Font('assets/JetBrainsMono-Regular.ttf', 12)
height = 600
width = int((1+0.5*SQRT3)*height)-1
window = pygame.display.set_mode((width, height))

buffersize = 2048
sample_rate = 192000
inv_smplr = 1/44100
max_sample = 32767
pygame.mixer.pre_init(sample_rate, -16, 1, buffer=buffersize)
pygame.mixer.init()
buf = np.zeros((buffersize, 2), dtype=np.int16)
sound = pygame.sndarray.make_sound(buf)

view_controls_text = 'SPACE :  view controls / info'
legend_text = [
    '       1-9 :  number of tones',
    ' E/← · R/→ :  rotate',
    '     D · F :  rotate by 3/2',
    '     C · V :  rotate active by 3/2',
    '         W :  spread equally',
    '         Q :  randomize',
    '         A :  randomize angle',
    '         X :  alt display',
    '         Z :  paint',
    '         S :  bilateral guide',
    '         H :  change hue',
    '       Esc :  quit'
]

huewheel = Image.open('assets/hue_wheel.png')
harm = pygame.image.load('assets/harmonic_entropy.png')
legend = pygame.image.load('assets/controls.png')
legend2 = pygame.image.load('assets/controls2.png')

hue_width, _ = huewheel.size

def get_hue_colors(num_colors, offset=0):
    colors = []
    # offset = width*np.random.rand()
    for i in range(num_colors):
        pos = round(offset + (hue_width-1)*i/num_colors) % (hue_width-1)
        color = np.array(huewheel.getpixel((pos, 0)))
        colors.append(color.astype(np.uint8))
    return colors

def thomaes_function():
    numer = [1]
    denom = [1]
    for d in range(2, DEPTH + 1):
        for n in range(d+1, d*2):
            gcd = math.gcd(n, d)

            if gcd == 1:
                numer.append(n)
                denom.append(d)
    numer = np.array(numer[::-1])
    denom = np.array(denom[::-1])
    return (np.log2(numer/denom), 1.0/denom)

def octave_reduce(x):
    return 2 ** (math.log2(x) % 1)

def pitch_class_distance(a, b):
    value = abs(a-b)
    return value
    # if value <= 0.5:
    #     return value
    # else:
    #     return 1-value
    
def interval_sign(x):
    x = x%1 + 1
    e = 0.0003
    if 1+e<=x<=math.sqrt(9/8)-e:
        return (-1, 'm1')
    elif 9/8+e<=x<=math.sqrt(3/2)-e:
        return (-1, 'm3')
    elif 4/3+e<=x<=math.sqrt(2)-e:
        return (-1, 'm5')
    elif 3/2+e<=x<=math.sqrt(8/3)-e:
        return (-1, 'm7')
    elif 16/9+e<=x<=math.sqrt(32/9)-e:
        return (-1, 'm9')
    elif math.sqrt(9/8)+e<=x<=9/8-e:
        return (1, 'M1')
    elif math.sqrt(3/2)+e<=x<=4/3-e:
        return (1, 'M3')
    elif math.sqrt(2)+e<=x<=3/2-e:
        return (1, 'M5')
    elif math.sqrt(8/3)+e<=x<=16/9-e:
        return (1, 'M7')
    elif math.sqrt(32/9)+e<=x<=2-e:
        return (1, 'M9')
    elif math.sqrt(9/8)-e<x<math.sqrt(9/8)+e:
        return (2, 'n1')
    elif math.sqrt(3/2)-e<x<math.sqrt(3/2)+e:
        return (2, 'n3')
    elif math.sqrt(2)-e<x<math.sqrt(2)+e:
        return (2, 'n5')
    elif math.sqrt(8/3)-e<x<math.sqrt(8/3)+e:
        return (2, 'n7')
    elif math.sqrt(32/9)-e<x<math.sqrt(32/9)+e:
        return (2, 'n9')
    elif 1<x<1+e or 2-e<x<2:
        return (0, '.0')
    elif 9/8-e<x<9/8+e:
        return (0, 'p2')
    elif 4/3-e<x<4/3+e:
        return (0, 'p4')
    elif 3/2-e<x<3/2+e:
        return (0, 'p6')
    elif 16/9-e<x<16/9+e:
        return (0, 'p8')
    return (0, '  ')

    # binary tree version

    # if x < math.sqrt(2):
    #     pass
    # else:
    #     pass


hue_colors = {i: get_hue_colors(i) for i in range(1, MAX_KNOBS + 1)}
white = np.array((255,255,255))
thomae = thomaes_function()
init_pos = np.array((0, 0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875, 0.0625))
knob_center = pygame.math.Vector2(height*0.5, height*0.5)
num_knobs = 3
knob_values, phase_vectors = np.array(init_pos[:num_knobs]), np.ones(num_knobs, dtype=complex) * 1j
active_line_index = 0
global_time = 0
hue_offset = 0
painting = False
divider = False
ghost_lines = False
mouse_held = False

running = True
while running:
    keys = pygame.key.get_pressed()

    sign = 1
    signh = 1
    rot_val = LOG2_5TH
    if keys[pygame.K_LSHIFT]:
        sign = 2.75
        signh = -1
        rot_val = LOG2_MAJ_3RD
    elif keys[pygame.K_LCTRL]:
        sign = 0.025
        signh = 8
        rot_val = LOG2_MIN_3RD
    
    if keys[pygame.K_h]:
        hue_offset += signh*10

    if keys[pygame.K_q]:
        knob_values = np.power(2, (np.log2(knob_values%1 + 1) + np.random.rand(num_knobs)) % 1) - 1
    if keys[pygame.K_a]:
        knob_values = np.power(2, (np.log2(knob_values%1 + 1) + np.random.rand()) % 1) - 1

    if keys[pygame.K_e] or keys[pygame.K_LEFT]:
        knob_values = np.power(2, (np.log2(knob_values%1 + 1) + sign*-0.003) % 1) - 1
    if keys[pygame.K_r] or keys[pygame.K_RIGHT]:
        knob_values = np.power(2, (np.log2(knob_values%1 + 1) + sign*0.003) % 1) - 1

    if keys[pygame.K_KP0]:
        active_line_index = np.random.randint(num_knobs+1)
        active_line_index = min(active_line_index, num_knobs-1)

    if 3 <= num_knobs:
        knob_center = pygame.math.Vector2(height*0.5, height*0.5)
    else:
        knob_center = pygame.math.Vector2(width*0.5, height*0.5)

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
            
        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_d:
                knob_values = np.power(2, (np.log2(knob_values%1 + 1) - rot_val) % 1) - 1   
            elif event.key == pygame.K_f:
                knob_values = np.power(2, (np.log2(knob_values%1 + 1) + rot_val) % 1) - 1
            elif event.key == pygame.K_c or event.key == pygame.K_DOWN:
                knob_values[active_line_index] = np.power(2, (np.log2(knob_values[active_line_index]%1 + 1) - rot_val) % 1) - 1
            elif event.key == pygame.K_v or event.key == pygame.K_UP:
                knob_values[active_line_index] = np.power(2, (np.log2(knob_values[active_line_index]%1 + 1) + rot_val) % 1) - 1
            elif event.key == pygame.K_SPACE:
                ENABLE_DETAIL = not ENABLE_DETAIL
            elif event.key == pygame.K_z:
                window.fill((0,0,0))
                if 3 <= num_knobs:
                    window.blit(harm, (height, 0))
                painting = True
            elif event.key == pygame.K_x:
                ghost_lines = not ghost_lines
            elif event.key == pygame.K_s:
                divider = not divider
            elif event.key == pygame.K_w:
                knob_values = np.power(2, (np.linspace(1, 2, num_knobs, endpoint=False) + np.random.rand())%1) - 1

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

                    knob_values = np.power(2, (np.log2(np.array(init_pos[:num_knobs])%1 + 1))%1) - 1 # + np.random.rand()

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
                    log_value = np.log2(value%1+1)
                    line_ctr = knob_center + pygame.math.Vector2(height*0.25 * np.sin(log_value * TAU),
                                                                -height*0.25 * np.cos(log_value * TAU))
                    dist = line_ctr.distance_to(mouse_pos)
                    if dist < min_distance:
                        min_distance = dist
                        closest_line_index = i
                active_line_index = closest_line_index
            elif event.button == 3:  # right click
                if pygame.key.get_mods() & pygame.KMOD_CTRL:  # ctrl + left click
                    mouse_pos = event.pos
                    closest_line_index = None
                    min_distance = float('inf')
                    for i, value in enumerate(knob_values):
                        log_value = np.log2(value%1+1)
                        line_ctr = knob_center + pygame.math.Vector2(height*0.25 * np.sin(log_value * TAU),
                                                                    -height*0.25 * np.cos(log_value * TAU))
                        dist = line_ctr.distance_to(mouse_pos)
                        if dist < min_distance:
                            min_distance = dist
                            closest_line_index = i
                    if closest_line_index is not None and num_knobs > 1:
                        knob_values = np.delete(knob_values, closest_line_index)
                        phase_vectors = np.delete(phase_vectors, closest_line_index)
                        num_knobs -= 1
                        active_line_index = min(active_line_index, num_knobs-1)
                elif num_knobs+1 <= 9:
                    mouse_pos = event.pos
                    dx, dy = pygame.math.Vector2(mouse_pos) - knob_center
                    angle = math.atan2(dy, dx)*0.5*PI_INV
                    new_value = np.power(2,(angle+0.25)%1)-1
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
        knob_values[active_line_index] = np.power(2,(angle+0.25)%1)-1

    if not painting:
        window.fill((0,0,0)) # rect=[(0,0),(height, height)]

    hue_colors[num_knobs] = get_hue_colors(num_knobs, hue_offset)
    colors = hue_colors[num_knobs]


    if 3 <= num_knobs:
        if not painting:
            window.blit(harm, (height, 0))
            
        if divider and not painting:
            line_surface = pygame.Surface((height, height))
            pygame.draw.aaline(line_surface, (60,16,55), (0,height-1), (SQRT3_INV*(height-1), 0), 1)
            window.blit(line_surface, (height+1, 0), special_flags=pygame.BLEND_RGB_ADD)

        subsets = list(itertools.combinations(set(knob_values), 3))

        for i in range(len(subsets)):
            a, b, c =  subsets[i][0], subsets[i][1], subsets[i][2]
            M = [octave_reduce(a+1), octave_reduce(b+1), octave_reduce(c+1)]
            sorted_M = sorted(M)
            u, v = np.log2(np.array((sorted_M[2]/sorted_M[0], sorted_M[1]/sorted_M[0]))) + 1

            if (v >= 2*u - 2) and (u + v <= 3):
                point_x = SQRT3 * (v - 1) * (height-1) + height+1
                point_y = height - (2*u - v - 1) * (height-1)
            elif (u + v > 3) and (2*v - u >= 1):
                point_x = SQRT3 * (u - v) * (height-1) + height+1
                point_y = height - (4 - u - v) * (height-1)
            elif (v < 2*u - 2) and (2*v - u < 1):
                point_x = SQRT3 * (2 - u) * (height-1) + height+1
                point_y = height - (2*v - u) * (height-1)
            
            pygame.draw.circle(window, colors[0], (point_x, point_y), 2)

    if ENABLE_DETAIL and not painting:
        window.blit(legend, (width-239, height-166))

    log_knob_vals = np.log2(knob_values%1 + 1)
    start_index = round(thomae[0].size - thomae[0].size/num_knobs)

    if num_knobs == 1:
        pos = thomae[0]
        val = thomae[1]

        angles = (pos+log_knob_vals) * TAU

        coords_x = knob_center.x + height*0.5 * np.sin(angles)
        coords_y = knob_center.y - height*0.5 * np.cos(angles)

        if ghost_lines:
            for i in range(thomae[0].size-1):
                color = val[i]*255
                pygame.draw.line(window, (color,color,color), knob_center, (coords_x[i], coords_y[i]))
            
            pygame.draw.aaline(window, tuple(np.round(val[-1]*white)), knob_center, (coords_x[-1], coords_y[-1]))
        else:
            coords_x1 = knob_center.x + height*0.35 * np.sin(angles)
            coords_y1 = knob_center.y - height*0.35 * np.cos(angles)
            
            for i in range(thomae[0].size-1):
                color = val[i]*255
                pygame.draw.line(window, (color,color,color), (coords_x[i], coords_y[i]), (coords_x1[i], coords_y1[i]))

            pygame.draw.aaline(window, tuple(np.round(val[-1]*white)), knob_center, (coords_x1[-1], coords_y1[-1]))
    else:
        indices = np.arange(start_index, thomae[0].size)
        pos = thomae[0][indices]
        val = thomae[1][indices]

        angles = (pos[:,np.newaxis]+log_knob_vals) * TAU

        coords_x = knob_center.x + height*0.5 * np.sin(angles)
        coords_y = knob_center.y - height*0.5 * np.cos(angles)

        if ghost_lines:
            for i in range(start_index, thomae[0].size-1):
                line_width = round(np.power(255/(val[i-start_index]), 0.06))
                for j in range(num_knobs):
                    color = tuple(np.round(val[i-start_index] * colors[j]))
                    pygame.draw.line(window, color, knob_center, (coords_x[i-start_index, j], coords_y[i-start_index, j]), line_width)
            for j in range(num_knobs):
                color = tuple(np.round(val[-1] * colors[j]))
                pygame.draw.aaline(window, color, knob_center, (coords_x[-1, j], coords_y[-1, j]))
        else:
            coords_x1 = knob_center.x + height*0.35 * np.sin(angles)
            coords_y1 = knob_center.y - height*0.35 * np.cos(angles)

            for i in range(start_index, thomae[0].size-1):
                line_width = round(np.power(255/(val[i-start_index]), 0.06))
                for j in range(num_knobs):
                    color = tuple(np.round(val[i-start_index] * colors[j]))
                    pygame.draw.line(window, color, (coords_x1[i-start_index, j], coords_y1[i-start_index, j]), (coords_x[i-start_index, j], coords_y[i-start_index, j]), line_width)
            for j in range(num_knobs):
                color = tuple(np.round(val[-1] * colors[j]))
                pygame.draw.aaline(window, color, knob_center, (coords_x1[-1, j], coords_y1[-1, j]))


    sorted_indices = knob_values.argsort()
    
    freqs = BASE_FREQ * (np.sort(knob_values[sorted_indices])%1 + 1)
    sorted_colors = np.array(colors)[sorted_indices]

    if not painting:
        if ENABLE_DETAIL:
            if num_knobs == 1:
                text_freq = font.render(f'{freqs[0]:.5f} Hz', True, (255, 255, 255))
                window.blit(text_freq, (55, height-16))
            else:
                for i in range(num_knobs):
                    text_freq = font.render(f'{freqs[i]:.5f} Hz', True, sorted_colors[i])
                    window.blit(text_freq, (120, height-num_knobs*15 + i*15 - 1))

                intervals = list(itertools.combinations(set(np.log2(knob_values%1 + 1)), 2))
                distances = sorted([pitch_class_distance(i[0], i[1]) for i in intervals])
                for i, d in enumerate(distances):
                    sign, degree = interval_sign(np.power(2,d))
                        
                    if sign == 1:
                        text_intervals = font.render(f'{degree} {d:.4f}', True, (255, 101, 187))
                    elif sign == -1:
                        text_intervals = font.render(f'{degree} {d:.4f}', True, (0, 220, 176))
                    elif sign == 2:
                        text_intervals = font.render(f'{degree:} {d:.4f}', True, (155, 158, 255))
                    else:
                        text_intervals = font.render(f'{degree:} {d:.4f}', True, (206, 206, 206))
                    window.blit(text_intervals, (3, height-len(distances)*15 + i*15 - 1))

        # for i, text in enumerate(legend_text):
        #     legend_surface = font.render(text, True, (158, 155, 157))
        #     window.blit(legend_surface, (width-241, height-len(legend_text)*14 + i*14 - 3))

        else:
            window.blit(legend2, (width-205, height-13))
            # legend_surface = font.render(view_controls_text, True, (158, 155, 157))
            # window.blit(legend_surface, )

        if ENABLE_FPS and ENABLE_DETAIL:
            fps = clock.get_fps()
            fps_surface = font.render(f'{fps:.2f} fps', True, (158, 155, 157))
            window.blit(fps_surface, (3,-2))

    pygame.display.flip()

    freqs = BASE_FREQ * (knob_values%1 + 1)

    if pygame.mixer.Channel(0).get_queue() is None:
        angles = (TAU * freqs * inv_smplr).astype(complex)
        rotation_vectors = np.power(np.e, angles* 1j)

        for t in range(buffersize):
            phase_vectors *= rotation_vectors
            # phase_vectors *= (2.0-abs(phase_vectors))

            wave = phase_vectors.imag.sum()
            buf[t] = round(max_sample * 0.1 * wave)

    sound = pygame.sndarray.make_sound(buf)
    pygame.mixer.Channel(0).queue(sound)

    global_time += buffersize
    if ENABLE_FPS:
        clock.tick(60)

pygame.quit()