import torch.nn as nn
from snake import *



class RDL(nn.Module):
    def __init__(self):
        super(RDL, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        logout = self.layer(x)
        expout = torch.exp(logout)
        out = expout / torch.sum(expout)
        return out

class VALUE(nn.Module):
    def __init__(self):
        super(VALUE, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layer(x)

def reward(model, value, discount, sampling_num):
    all_score = 0
    total_step = 0
    for i in range(sampling_num):
        snake, apple, direction = game_ini()
        death = 0
        total_score = 0

        temp_step = 0
        envs = torch.empty(size=(0, 8), dtype=torch.float32)
        outputs = torch.empty(size=(0, 4), dtype=torch.int)
        scores = []
        while death == 0:
            env = input_interpreter(snake, apple, direction)
            envs = torch.cat((envs, env), 0)
            out = model.forward(env)
            manipulation, output = output_interpreter(out)

            outputs = torch.cat((outputs, output), 0)
            control = direction_checker(manipulation, direction, snake)
            snake, apple, direction, score, death = game(snake, apple, control, death)
            total_score += score
            scores.append(score - 2*death)
            temp_step += 1
            if score == 1:
                temp_step = 0
            if temp_step > 64:
                death = 1
            total_step += 1

        length = envs.size(dim=0)
        expect_out = model.forward(envs)
        values = torch.transpose(value.forward(envs), 0, 1)
        temp_sum = torch.log(torch.diagonal(torch.matmul(expect_out, torch.transpose(outputs, 0, 1)), 0)).unsqueeze(-1)

        ret = torch.zeros(size=(1, length))
        temp = 0
        for i in range(length):
            temp = scores[length-i-1] + discount * temp
            ret[0, length-i-1] = temp

        error = (values - ret).clone().detach()

        model_loss = torch.matmul(error, temp_sum).sum()
        value_loss = torch.matmul(torch.transpose(error, 0, 1), values).sum()

        all_score += total_score
    return model_loss / sampling_num, value_loss / sampling_num, all_score // sampling_num, total_step // sampling_num


def train_model(model, value, discount, epochs, learning_rate, num=160, min_num=16):
    m_opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
    v_opt = torch.optim.SGD(value.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        value.train()
        m_opt.zero_grad()
        game_nun = max(num - epoch//8, min_num)
        m_loss, v_loss, mean_score, step = reward(model, value, discount, game_nun)
        m_loss.backward()
        m_opt.step()
        v_opt.zero_grad()
        v_loss.backward()
        v_opt.step()
        print('epoch:', epoch, 'average score:', mean_score, 'average step:', step)

    return model, value


def model_eval(model, show_out=False):
    model.eval()
    snake, apple, direction = game_ini()
    death = 0
    total_score = 0
    with torch.no_grad():
        while death == 0:
            env = input_interpreter(snake, apple, direction)
            out = model.forward(env)
            if show_out:
                print(out)
            manipulation, output = output_interpreter(out)
            control = direction_checker(manipulation, direction, snake)
            snake, apple, direction, score, death = game(snake, apple, control, death, display=True, delay=True)
            total_score += score
            print('score:', total_score, 'death:', death)



if __name__ == '__main__':

    patha = 'D:\DRL\Rein_actor.pt'
    pathv = 'D:\DRL\Rein_value.pt'

    learning_rate = 2**-7
    epochs = 2**10
    discount = 0.9
    model = RDL()
    value = VALUE()
    model.load_state_dict(torch.load(patha))
    value.load_state_dict(torch.load(pathv))


    model, value = train_model(model, value, discount, epochs, learning_rate, num=160)

    torch.save(model.state_dict(), patha)
    torch.save(value.state_dict(), pathv)

    model_eval(model, show_out=True)
