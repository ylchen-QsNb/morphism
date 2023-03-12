import pygame as pg
from snake import *
from A2C_lstm import ai_play, ACTOR, HIDDEN

'''
initialize pygame and set globals
'''
pg.init()
WIDTH = 800
HEIGHT = 800
ROW = 16
COL = 16
SIZE = HEIGHT//ROW

BG_COLOR = (255, 245, 235)
HEAD_COLOR = (128, 128, 128)
FOOD_COLOR = (255, 64, 0)
BODY_COLOR = (0, 16, 245)

TOG_KEYBOARD = False

'''
CLASS AND FUNCS
'''
class Point():
    def __init__(self, list):
        self.row = list[0]
        self.col = list[1]

def rect(point, color):
    left = point.col * SIZE
    top = point.row * SIZE

    pg.draw.rect(
        window, color, (left, top, SIZE, SIZE)
    )
    pass

'''
INITIALIZE GAME
'''
txt_font = pg.font.SysFont(None, 48)


snake, apple, direction = game_ini()
manipulation = direction

actor = ACTOR().cuda()
actor.load_state_dict(torch.load('D:\DRL\Rein_actor.pt'))
actor.eval()
h = torch.zeros(2, HIDDEN).cuda()
c = torch.zeros(2, HIDDEN).cuda()
'''
RENDERING
'''
roun = 0
size = (WIDTH, HEIGHT)
window = pg.display.set_mode(size)

pg.display.set_caption('贪吃蛇')
direct = ''
quit = False
clock = pg.time.Clock()
death = 0
total_score = 0
high_score = 0
while not quit:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            quit = True
        elif event.type == pg.KEYDOWN:
            if event.key == 1073741906:
                direct = 'up'
            elif event.key == 1073741905:
                direct = 'down'
            elif event.key == 1073741904:
                direct = 'left'
            elif event.key == 1073741903:
                direct = 'right'
    if TOG_KEYBOARD:
        if direct == 'left':
            manipulation = [0, -1]
        elif direct == 'right':
            manipulation = [0, 1]
        elif direct == 'up':
            manipulation = [-1, 0]
        elif direct == 'down':
            manipulation = [1, 0]



    roun += 1

    if roun == 4:
        if not TOG_KEYBOARD:
            manipulation, h, c = ai_play(actor, snake, apple, direction, h, c)

        direction = direction_checker(manipulation, direction, snake)

        snake, apple, direction, score, death = game(snake, apple, direction, death)
        total_score += score
        high_score = max(high_score, total_score)

        roun = 0
    if death:
        total_score = 0
        h = torch.zeros(2, HIDDEN).cuda()
        c = torch.zeros(2, HIDDEN).cuda()
    head = snake[0]


    pg.draw.rect(window, BG_COLOR, (0, 0, WIDTH, HEIGHT))

    headp = Point(head)
    rect(headp, HEAD_COLOR)
    for body in snake[1:]:
        bodyp = Point(body)
        rect(bodyp, BODY_COLOR)
    applep = Point(apple)
    rect(applep, FOOD_COLOR)

    txt_image = txt_font.render(f'Score: {total_score}, Highest Score: {high_score}', True, (16, 16, 16))
    txt_rect = txt_image.get_rect()

    window.blit(txt_image, txt_rect)
    pg.display.flip()



    clock.tick(30)
