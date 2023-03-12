import torch.nn as nn
from snake import *
import math


HIDDEN = 256

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.lstm = nn.LSTM(
            8, HIDDEN, 2
        )
        self.dense = nn.Linear(HIDDEN, 4)

    def forward(self, x, h, c):
        rawout, (hn, cn) = self.lstm(x, (h, c))
        out = self.dense(rawout)
        meanout = torch.mean(out.clone().detach(), dim=1).unsqueeze(-1)
        out = out - meanout
        return nn.functional.softmax(out, dim=1), hn, cn

class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        self.lstm = nn.LSTM(
            8, HIDDEN, 2
        )
        self.dense = nn.Linear(HIDDEN, 1)

    def forward(self, x, h, c):
        rawout, (hn, cn) = self.lstm(x, (h, c))
        out = self.dense(rawout)
        return out, hn, cn

def reward(model, value, discount):
    snake, apple, direction = game_ini()
    death = 0
    total_score = 0
    temp_step = 0
    step = 0
    envs = torch.empty(size=(0, 8), dtype=torch.float32).to(device)
    outputs = torch.empty(size=(0, 4), dtype=torch.int).to(device)
    scores = torch.empty(size=(0, 1), dtype=torch.float32).to(device)
    discounts = torch.empty(size=(0, 1), dtype=torch.float32).to(device)
    h = torch.zeros(2, HIDDEN).to(device)
    c = torch.zeros(2, HIDDEN).to(device)
    while death == 0:
        env = input_interpreter(snake, apple, direction).to(device)
        envs = torch.cat((envs, env), 0)
        out, h, c = model.forward(env, h, c)
        manipulation, output = output_interpreter(out)
        outputs = torch.cat((outputs, output.to(device)), 0)
        control = direction_checker(manipulation, direction, snake)
        snake, apple, direction, score, death = game(snake, apple, control, death)
        total_score += score
        discounts = discount * discounts
        discounts = torch.cat((discounts, torch.ones(1, 1).to(device)), 0)
        addon = (score - death) * discounts
        scores = torch.cat((scores, torch.zeros(1, 1).to(device)), 0)
        scores = scores + addon
        temp_step += 1
        if score == 1:
            temp_step = 0
        if temp_step > 64:
            death = 1
        step += 1

    h0 = torch.zeros(2, HIDDEN).to(device)
    c0 = torch.zeros(2, HIDDEN).to(device)

    value_seq, _, _ = value.forward(envs, h0, c0)
    td = value_seq - scores
    td_error = td.clone().detach().squeeze().to(device)

    out, _, _ = model.forward(envs, h0, c0)
    model_loss = torch.dot(td_error, torch.log(torch.diagonal(torch.matmul(out, torch.transpose(outputs, 0, 1)))))
    value_loss = torch.dot(td_error, value_seq.squeeze())

    return model_loss, value_loss, total_score, step


def train_model(model, value, m_opt, v_opt, discount, epochs):
    for epoch in range(epochs):
        model.train()
        value.train()
        m_loss, v_loss, mean_score, step = reward(model, value, discount)

        m_opt.zero_grad()
        m_loss.backward()
        m_opt.step()

        v_opt.zero_grad()
        v_loss.backward()
        v_opt.step()

        print('\r', 'Epoch [{}/{}], score: {}, step: {}'.format(epoch + 1, epochs, mean_score, step), end='')

    return model, value


def model_eval(model, show_out=False):
    model.eval()
    snake, apple, direction = game_ini()
    death = 0
    total_score = 0
    h0 = torch.zeros(2, HIDDEN).to(device)
    c0 = torch.zeros(2, HIDDEN).to(device)
    with torch.no_grad():
        while death == 0:
            env = input_interpreter(snake, apple, direction).to(device)
            out, _, _ = model.forward(env, h0, c0)
            if show_out:
                print(out)
            manipulation, output = output_interpreter(out)
            control = direction_checker(manipulation, direction, snake)
            snake, apple, direction, score, death = game(snake, apple, control, death, display=True, delay=True)
            total_score += score
            print('score:', total_score, 'death:', death)



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('if my gpu is available:', torch.cuda.is_available(), 'my cuda version:', torch.cuda_version,
          'torch version:', torch.__version__)

    patha = 'D:\DRL\Rein_actor.pt'
    pathv = 'D:\DRL\Rein_value.pt'


    epochs = 2**10
    era = 2**6
    discount = 0.9
    model = Policy().to(device)
    value = Value().to(device)

    for i in range(era):
        model.load_state_dict(torch.load(patha))
        value.load_state_dict(torch.load(pathv))
        learning_rate = 2 ** (-8-math.log(i+1))
        m_opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
        v_opt = torch.optim.SGD(value.parameters(), lr=learning_rate)
        print('Era: [{}/{}]'.format(i+1, era))
        model, value = train_model(model, value, m_opt, v_opt, discount, epochs)

        torch.save(model.state_dict(), patha)
        torch.save(value.state_dict(), pathv)

        model_eval(model, show_out=True)