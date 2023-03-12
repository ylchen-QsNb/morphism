import numpy as np
import time
import torch


BOARD_HEIGHT = 16
BOARD_WIDTH = 16

def snake_board(snake_status, apple_loc):
    board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH))
    if apple_loc[0] >= 0:
        board[apple_loc[0], apple_loc[1]] = 5
    count = 0
    for pt in snake_status:
        if count == 0:
            board[pt[0], pt[1]] = 2
        else:
            board[pt[0], pt[1]] = 1
        count = 1
    return board

def snake_move(control, snake_status):
    tail = snake_status.pop()
    temp = snake_status[0]
    snake_status.insert(0, [temp[0] + control[0], temp[1] + control[1]])
    return snake_status, tail


def good_bad(snake_status, tail, apple_loc):
    sc = 0
    new_loc = apple_loc
    head = snake_status[0]
    # EAT APPLE
    if head == apple_loc:
        snake_status.append(tail)
        new_loc = [-1, -1]
        sc = 1
    # BANG THE WALL
    if head[0] == -1 or head[0] == BOARD_HEIGHT or head[1] == -1 or head[1] == BOARD_WIDTH:
        return 1, new_loc, sc
    # BANG ITS BODY
    if head in snake_status[1:]:
        return 1, new_loc, sc

    return 0, new_loc, sc



def new_apple(apple_loc, snake_status):
    new_loc = apple_loc[:]
    if apple_loc[0] == -1:
        new_loc = [0, 0]
        new_loc[0] = np.random.randint(0, high=BOARD_HEIGHT)
        new_loc[1] = np.random.randint(0, high=BOARD_WIDTH)
        while new_loc in snake_status:
            new_loc[0] = np.random.randint(0, high=BOARD_HEIGHT)
            new_loc[1] = np.random.randint(0, high=BOARD_WIDTH)

    return new_loc

'''
game body
'''


def game_ini():
    head = [0, 0]
    head[0] = np.random.randint(2, high=BOARD_HEIGHT-2)
    head[1] = np.random.randint(2, high=BOARD_WIDTH-2)
    snake_dir = np.random.randint(2, size=2)
    if snake_dir[0] == 0:
        if snake_dir[1] == 0:
            snake = [head] + [[head[0] + 1, head[1]]] + [[head[0] + 2, head[1]]]
            direction = [-1, 0]
        else:
            snake = [head] + [[head[0] - 1, head[1]]] + [[head[0] - 2, head[1]]]
            direction = [1, 0]
    else:
        if snake_dir[1] == 0:
            snake = [head] + [[head[0], head[1] + 1]] + [[head[0], head[1] + 2]]
            direction = [0, -1]
        else:
            snake = [head] + [[head[0], head[1] - 1]] + [[head[0], head[1] - 2]]
            direction = [0, 1]

    apple = new_apple([-1, -1], snake)
    return snake, apple, direction

def game(snake, apple, direction, death, display=False, delay=False):
    if death == 0:
        snake, tail = snake_move(direction, snake)
        death, apple, score = good_bad(snake, tail, apple)
        apple = new_apple(apple, snake)

    else:
        snake, apple, direction = game_ini()
        score = 0
        death = 0

    if display:
        if death == 0:
            board = snake_board(snake, apple)
            print(board)
        else:
            print('game over!')
    if delay:
        time.sleep(0.5)
    return snake, apple, direction, score, death

def output_interpreter(output):
    # except_rate = np.random.random()
    # cumprob = torch.cumsum(output, dim=1).tolist()[0]
    # for temp in range(4):
    #     if except_rate < cumprob[temp]:
    #         break
    # print(output)

    temp = torch.multinomial(output, 1).item()
    out = torch.zeros(size=(1, 4))
    if temp == 0:
        control = [1, 0]
        out[0, 0] = 1
    elif temp == 1:
        control = [-1, 0]
        out[0, 1] = 1
    elif temp == 2:
        control = [0, 1]
        out[0, 2] = 1
    elif temp == 3:
        control = [0, -1]
        out[0, 3] = 1
    return control, out


def input_interpreter(snake, apple, direction):
    out = torch.zeros(size=(1, 8))
    head = snake[0]
    body = snake[1:]

    test = [head[0] + 1, head[0]]
    if test[0] >= BOARD_HEIGHT-1 or test in body:
        out[0, 0] = 0
    else:
        out[0, 0] = 1

    test = [head[0] - 1, head[0]]
    if test[0] <= 0 or test in body:
        out[0, 1] = 0
    else:
        out[0, 1] = 1

    test = [head[0], head[0] + 1]
    if test[1] >= BOARD_WIDTH-1 or test in body:
        out[0, 2] = 0
    else:
        out[0, 2] = 1

    test = [head[0], head[0] - 1]
    if test[1] <= 0 or test in body:
        out[0, 3] = 0
    else:
        out[0, 3] = 1

    # distance = ((apple[0] - head[0])**2 + (apple[1] - head[1])**2) ** (1/2)
    out[0, 4] = (apple[0] - head[0])
    out[0, 5] = (apple[1] - head[1])

    out[0, 6] = direction[0]
    out[0, 7] = direction[1]
    # out[8] = len(snake)

    return out

def direction_checker(input, control, snake_status):
    forbid = (np.array(snake_status)[1] - np.array(snake_status)[0]).tolist()
    # print('forbid:', forbid)
    if input != forbid:
        return input
    else:
        return control
