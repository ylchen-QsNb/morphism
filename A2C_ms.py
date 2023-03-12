import torch.nn as nn
from snake import *
import queue

class ACTOR(nn.Module):
    def __init__(self):
        super(ACTOR, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        rawout = self.layer(x)
        # print(rawout)
        return rawout / torch.sum(rawout)

class CRITIC(nn.Module):
    def __init__(self):
        super(CRITIC, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.layer(x)


def ac_train(actor, critic, actor_opt, critic_opt, run_time, discount, MultStep=32, display=False, delay=False):
    state_q = queue.Queue(maxsize=MultStep)

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
            newenv = input_interpreter(snake, apple, direction)
            reward = 0
        else:
            env = input_interpreter(snake, apple, direction)
            out = actor.forward(env)
            # print(out)
            manipulation, output = output_interpreter(out)
            control = direction_checker(manipulation, direction, snake)
            snake, apple, direction, score, death = game(snake, apple, control, death, display=display, delay=delay)
            newenv = input_interpreter(snake, apple, direction)
            reward = (score - death)

            total_score += score
            if total_score >= 16:
                print('epoch:', i+1,'/ '+str(run_time), 'score:', total_score)

        state_q.put((env, output, reward, newenv))

        if i >= MultStep - 1:
            dis_ret = 0
            for j in range(MultStep):
                element = state_q.get()
                dis_ret += element[2] * discount**j
                if j == 0:
                    s_t = element[0]
                    output = element[1]
                    # print('output:', output)
                elif j == MultStep - 1:
                    s_tm = element[3]
                    state_q.put(element)
                else:
                    state_q.put(element)

            td_error = critic.forward(s_t) - dis_ret - critic.forward(s_tm) * discount**MultStep
            td_error = td_error.clone().detach()
            # print(td_error)

            out = actor.forward(s_t)
            actor_loss = td_error * torch.log(torch.matmul(out, torch.transpose(output, 0, 1)))
            # print(out)
            # print(actor_loss)
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            value = critic.forward(s_t)
            critic_loss = td_error * value
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

    return actor, critic

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
            snake, apple, direction, score, death = game(snake, apple, control, death, display=True, delay=False)
            total_score += score
            print('score:', total_score, 'death:', death)

def ai_play(model, snake, apple, direction):
    with torch.no_grad():
        env = input_interpreter(snake, apple, direction)
        out = model.forward(env)
        manipulation, output = output_interpreter(out)

    return manipulation


if __name__ == '__main__':
    learning_rate_a = 2**-25
    learning_rate_c = 2**-25
    run_time = 2**15
    discount = 0.9
    train = 1

    patha = 'D:\DRL\A3C_actor1.pt'
    pathc = 'D:\DRL\A3C_critic1.pt'
    pathm = 'D:\DRL\model.pt'

    actor = ACTOR()
    critic = CRITIC()
    actor.load_state_dict(torch.load(patha))
    critic.load_state_dict(torch.load(pathc))

    actor_opt = torch.optim.SGD(actor.parameters(), lr=learning_rate_a)
    critic_opt = torch.optim.SGD(critic.parameters(), lr=learning_rate_c)
    if train:
        actor, critic = ac_train(actor, critic, actor_opt, critic_opt, run_time, discount, MultStep=2*10, display=False, delay=False)

        torch.save(actor.state_dict(), patha)
        torch.save(critic.state_dict(), pathc)

    model_eval(actor)