import torch.nn as nn
from snake import *
import queue

HIDDEN = 256

class ACTOR(nn.Module):
    def __init__(self):
        super(ACTOR, self).__init__()
        self.lstm = nn.LSTM(
            8, HIDDEN, 2
        )
        self.dense = nn.Linear(HIDDEN, 4)

    def forward(self, x, h, c):
        rawout, (hn, cn) = self.lstm(x, (h, c))
        out = self.dense(rawout)
        maxout, _ = torch.max(out.clone().detach(), dim=1)
        out = out - maxout[:, None]
        return nn.functional.softmax(out, dim=1), hn, cn

class CRITIC(nn.Module):
    def __init__(self):
        super(CRITIC, self).__init__()
        self.lstm = nn.LSTM(
            8, HIDDEN, 2
        )
        self.dense = nn.Linear(HIDDEN, 1)

    def forward(self, x, h, c):
        rawout, (hn, cn) = self.lstm(x, (h, c))
        out = self.dense(rawout)
        return out, hn, cn


def ac_train(actor, critic, actor_opt, critic_opt, run_time, discount, MultStep=32, display=False, delay=False):
    state_q = queue.Queue(maxsize=MultStep)
    ha = torch.zeros(2, HIDDEN).to(device)
    ca = torch.zeros(2, HIDDEN).to(device)
    snake, apple, direction = game_ini()
    death = 0
    total_score = 0

    for i in range(run_time):

        actor.train()
        critic.train()

        if death != 0:
            env = input_interpreter(snake, apple, direction)
            snake, apple, direction = game_ini()
            death = 0
            total_score = 0
            # newenv = input_interpreter(snake, apple, direction)
            reward = 0
            ha = torch.zeros(2, HIDDEN).to(device)
            ca = torch.zeros(2, HIDDEN).to(device)
        else:
            env = input_interpreter(snake, apple, direction).to(device)
            out, ha, ca = actor.forward(env, ha, ca)
            # print(out)
            manipulation, output = output_interpreter(out)
            control = direction_checker(manipulation, direction, snake)
            snake, apple, direction, score, death = game(snake, apple, control, death, display=display, delay=delay)
            # newenv = input_interpreter(snake, apple, direction)
            reward = (score - death)

            total_score += score
            if total_score >= 16:
                print('epoch:', i+1, '/ '+str(run_time), 'score:', total_score)

        state_q.put((env, output, reward))

        if i >= MultStep - 1:
            rewards = torch.zeros(size=(MultStep, 1)).to(device)
            train_seq = torch.zeros(size=(0, 8)).to(device)
            outputs = torch.zeros(size=(0, 4)).to(device)
            for j in range(MultStep):
                element = state_q.get()
                rewards[j, 0] = element[2]
                train_seq = torch.cat((train_seq, element[0].to(device)), 0)
                outputs = torch.cat((outputs, element[1].to(device)), 0)
                if j >= 1:
                    state_q.put(element)

            h0 = torch.zeros(2, HIDDEN).to(device)
            c0 = torch.zeros(2, HIDDEN).to(device)

            value_seq, _, _ = critic.forward(train_seq, h0, c0)
            td = value_seq[:-1, 0] - rewards[:-1, 0] - discount * value_seq[1:, 0]
            td_error = td.clone().detach().to(device)

            out, _, _ = actor.forward(train_seq[:-1], h0, c0)
            actor_loss = torch.dot(td_error, torch.log(torch.diagonal(torch.matmul(out, torch.transpose(outputs[:-1, :], 0, 1)))))
            # print(out)
            # print(actor_loss)
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            value, _, _ = critic.forward(train_seq, h0, c0)
            critic_loss = torch.dot(td_error, value.squeeze()[:-1])
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

    return actor, critic

def model_eval(model):
    h = torch.zeros(2, HIDDEN).to(device)
    c = torch.zeros(2, HIDDEN).to(device)
    model.eval()
    snake, apple, direction = game_ini()
    death = 0
    total_score = 0
    with torch.no_grad():
        while death == 0:
            env = input_interpreter(snake, apple, direction).to(device)
            out, h, c = model.forward(env, h, c)
            manipulation, output = output_interpreter(out)
            control = direction_checker(manipulation, direction, snake)
            snake, apple, direction, score, death = game(snake, apple, control, death, display=True, delay=False)
            total_score += score
            print('score:', total_score, 'death:', death)

def ai_play(model, snake, apple, direction, h, c):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        env = input_interpreter(snake, apple, direction).to(device)
        out, h, c = model.forward(env, h, c)
        manipulation, output = output_interpreter(out)

    return manipulation, h, c


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('if my gpu is available:', torch.cuda.is_available(), 'my cuda version:', torch.cuda_version,
          'torch version:', torch.__version__)

    learning_rate_a = 2**-20
    learning_rate_c = 2**-20
    run_time = 2**14
    discount = 0.9
    train = 1

    patha = 'D:\DRL\A2C_lstm_actor1.pt'
    pathc = 'D:\DRL\A2C_lstm_critic1.pt'
    pathm = 'D:\DRL\model.pt'

    for _ in range(2**5):
        actor = ACTOR().to(device)
        critic = CRITIC().to(device)
        actor.load_state_dict(torch.load(patha))
        critic.load_state_dict(torch.load(pathc))

        actor_opt = torch.optim.SGD(actor.parameters(), lr=learning_rate_a)
        critic_opt = torch.optim.SGD(critic.parameters(), lr=learning_rate_c)
        if train:
            actor, critic = ac_train(actor, critic, actor_opt, critic_opt, run_time, discount, MultStep=2*11, display=False, delay=False)

            torch.save(actor.state_dict(), patha)
            torch.save(critic.state_dict(), pathc)

        model_eval(actor)