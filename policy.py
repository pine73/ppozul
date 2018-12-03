import numpy as np
import azul
import time
import mcts
from copy import deepcopy
from multiprocessing import Pool
import ppozul

def random_policy(game):
    mask = game.mask()
    valid_commands = np.argwhere(mask == 1)
    random_index = np.random.randint(valid_commands.shape[0])
    command = valid_commands[random_index]
    return command

def slightly_less_random_policy(game):
    epsilon = 0.1
    mask = game.mask()
    valid_commands_without_floor = np.argwhere(mask[:,:,:5] == 1)
    valid_commands = np.argwhere(mask == 1)
    if valid_commands_without_floor.shape[0] == 0:
        valid_commands_without_floor = valid_commands
    if np.random.random() >= epsilon:
        random_index = np.random.randint(valid_commands_without_floor.shape[0])
        command = valid_commands_without_floor[random_index]
        return command
    else:
        random_index = np.random.randint(valid_commands.shape[0])
        command = valid_commands[random_index]
        return command

# inf func
def rollout(gamein):
    game = deepcopy(gamein)
    player = game.active_player_num

    mask = game.mask()
    prior = (mask/np.sum(mask)).flatten()

    while True:
        while True:
            command = slightly_less_random_policy(game)
            if game.take_command(command) == True: break    
        game.turn_end(verbose = False)
        if game.is_terminal:
            break
        game.start_turn()

    game.final_score()
    winner = game.leading_player_num
    if player == winner:
        return 1.,prior
    else:
        return -1.,prior

# policy
def mcts_roolout(game):
    commands = np.argwhere(np.ones((6,5,6))==1)
    search = mcts.MCTSearch(game,rollout,commands)
    action = search.start_search_deterministic(300)
    return action


class PPOHelper(object):
    def __init__(self):
        self._nn = ppozul.Nn_estimator(ppozul.Config())

    def __call__(self, game):
        commands = np.argwhere(np.ones((6,5,6))==1)
        _, prior = self._nn.predict(game.states()[None,:], game.flat_mask()[None,:])
        # act_idx = np.random.choice(180,p=prior.squeeze())
        act_idx = np.argmax(prior)
        act = commands[act_idx]
        return act




        


def poolvs(game,policy1,policy2):
    return game.aivs(policy1,policy2)


if __name__ == '__main__':
    pponn = PPOHelper()
    for _ in range(5):
        a = time.time()
        game = azul.Azul(2)
        results = game.aivs(pponn,slightly_less_random_policy)
        print(results)
        print(time.time()-a)

