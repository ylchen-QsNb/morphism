import torch
import torch.nn as nn
from snake import *

class ACTOR(nn.Module):
    def __init__(self):
        super(ACTOR, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layer(x)

class CRITIC(nn.Module):
    def __init__(self):
        super(CRITIC, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layer(x)

def critic_feed(env, output):
    out = torch.cat((env, output), 1)
    return out

def ac_train(actor, critic, actor_opt, critic_opt, run_time, discount, display=False, delay=False):
    death = 1
    total_score = 0
    for i in range(run_time):
        if death != 0:
            snake, apple, direction = game_ini()
            death = 0
            total_score = 0

        else:
            env = input_interpreter(snake, apple, direction)
            out = actor.forward(env)
            # print('1:', out)
            manipulation, output = output_interpreter(out)

            control = direction_checker(manipulation, direction, snake)
            comment = critic.forward(critic_feed(env, output))
            # print(comment)
            snake, apple, direction, score, death = game(snake, apple, control, death, display=display, delay=delay)
            total_score += score
            print('epoch:', i+1, 'score:', total_score)
            env_plus = input_interpreter(snake, apple, direction)
            quasi_out = actor.forward(env_plus)
            # print('2:', quasi_out)
            quasi_mani, quasi_output = output_interpreter(quasi_out)

            quasi_comm = critic.forward(critic_feed(env_plus, quasi_output))
            bll = (comment - (32*score - death + 0.1 + discount * quasi_comm))
            bl = bll.clone().detach()
            # cc = comment.clone().detach()
            td = bl * comment
            # print(bl)
            td.backward()
            actor_loss = -bl * torch.log(torch.matmul(out, torch.transpose(output, 0, 1)))
            # print(actor_loss)
            actor_loss.backward()
            critic_opt.step()
            actor_opt.step()

    return actor, critic

if __name__ == '__main__':
    learning_rate_a = 0.0001
    learning_rate_c = 0.001
    run_time = 100000
    discount = 0.9

    patha = 'D:\DRL\_actor.pt'
    pathc = 'D:\DRL\_critic.pt'
    pathm = 'D:\DRL\model.pt'

    actor = ACTOR()
    critic = CRITIC()
    actor.load_state_dict(torch.load(patha))
    critic.load_state_dict(torch.load(pathc))

    actor_opt = torch.optim.SGD(actor.parameters(), lr=learning_rate_a)
    critic_opt = torch.optim.SGD(critic.parameters(), lr=learning_rate_c)

    actor_trained, critic_trained = ac_train(actor, critic, actor_opt, critic_opt, run_time, discount, display=False, delay=False)

    torch.save(actor_trained.state_dict(), patha)
    torch.save(critic_trained.state_dict(), pathc)