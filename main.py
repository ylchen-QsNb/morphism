import torch.nn as nn
import torch
from snake import *

class RDL(nn.Module):
    def __init__(self):
        super(RDL, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.layer(x)
        return out

def reward(model, bias, sampling_num):
    total_reward = torch.zeros(size=(1,))
    all_score = 0
    total_step = 0
    for i in range(sampling_num):
        snake, apple, direction = game_ini()
        death = 0
        total_score = 0
        # fitness = 0
        # outputs = []
        temp_step = 0
        envs = torch.empty(size=(0, 8), dtype=torch.float32)
        outputs = torch.empty(size=(0, 4), dtype=torch.int)
        while death == 0:
            env = input_interpreter(snake, apple, direction)
            envs = torch.cat((envs, env), 0)
            out = model.forward(env)
            manipulation, output = output_interpreter(out)
            # outputs.append(output)
            outputs = torch.cat((outputs, output), 0)
            control = direction_checker(manipulation, direction, snake)
            snake, apple, direction, score, death = game(snake, apple, control, death)
            total_score += score
            temp_step += 1
            if score == 1:
                temp_step = 0
            if temp_step > 64:
                death = 1
            total_step += 1

        # length = len(outputs)
        expect_out = model.forward(envs)
        # temp_sum = torch.zeros(size=(length,))
        # for j in range(length):
        #     temp = outputs[j]
        #     temp_sum[j] += torch.log(expect_out[j, temp])
        # print(temp_sum)
        # print(torch.sum(temp_sum))
        # print(outputs)
        # print(expect_out)
        temp_sum = torch.log(torch.diagonal(torch.matmul(expect_out, torch.transpose(outputs, 0, 1)), 0))
        # print(torch.sum(temp_sum))
        # print(fitness)
        temp_reward = (total_score**1.1 - bias) * torch.sum(temp_sum)
        # print(temp_reward)
        total_reward -= temp_reward
        all_score += total_score
    return total_reward / sampling_num, all_score // sampling_num, total_step // sampling_num


def train_model(model, epochs, learning_rate, bias = 1.5, num=132):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        game_nun = num - epoch//10
        loss, mean_score, step = reward(model, bias, game_nun)
        loss.backward()
        optimizer.step()
        print('epoch:', epoch, 'reward:', loss.item(), 'average score:', mean_score, 'average step:', step)


def model_eval(model):
    model.eval()
    snake, apple, direction = game_ini()
    death = 0
    total_score = 0
    with torch.no_grad():
        while death == 0:
            env = input_interpreter(snake, apple, direction)
            out = model.forward(env)
            manipulation, output = output_interpreter(out)
            control = direction_checker(manipulation, direction, snake)
            snake, apple, direction, score, death = game(snake, apple, control, death, display=True, delay=True)
            total_score += score
            print('score:', total_score, 'death:', death)



if __name__ == '__main__':

    path = 'D:\DRL\model.pt'

    learning_rate = 0.01
    epochs = 1000

    model = RDL()
    model.load_state_dict(torch.load(path))

    # train_model(model, epochs, learning_rate, num=132)
    #
    # torch.save(model.state_dict(), path)

    model_eval(model)
