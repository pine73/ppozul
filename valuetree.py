import numpy as np
from copy import deepcopy

input_size = 155
output_size = 180

class _TreeNode(object):

    def __init__(self, game, value, prior, is_turn_end):

        self.visit_count = 1
        self.game = deepcopy(game)

        self.total_value = value
        self.prior = np.squeeze(prior)

        self.parent = None
        self.child = {}

        self.is_turn_end = is_turn_end

class ValueSearch(object):
    def __init__(self, game, nn):
        self._commands = np.argwhere(np.ones((6,5,6))==1)
        self._nn = nn
        root_value, root_prior = self._nn(game)
        self._root = _TreeNode(game, root_value, root_prior, False)



    def select(self, node):
        node.visit_count += 1
        #terminal node
        if node.is_turn_end:
            return node,[]

        #normal node
        
        action_index = np.random.choice(180, p=node.prior)

        if not action_index in node.child.keys():
            return node,action_index
        else:
            return self.select(node.child[action_index])

    def expand_and_evaluate(self,node,action_index):
        #terminal node
        if node.is_turn_end:
            game_prime = deepcopy(node.game)
            #evaluate
            game_prime.turn_end(verbose = False)
            if game_prime.is_terminal:
                game_prime.final_score()
                self._nn()
                return 1.,game_prime.leading_player_num
            game_prime.start_turn()
            value, prior = self._nn(game_prime)
            return value, game_prime.active_player_num

        #normal node
        action_command = self._commands[action_index]
        game_prime = deepcopy(node.game)
        is_turn_end = game_prime.take_command(action_command)
        #found a terminal node
        if is_turn_end:
            
            #expand
            child = _TreeNode(game_prime,None,[],is_turn_end)
            child.parent = node
            node.child[action_index] = child
            #evaluate
            game_prime.turn_end(verbose = False)
            if game_prime.is_terminal:
                game_prime.final_score()
                child.total_value = 1.
                self._nn()
                return 1.,game_prime.leading_player_num
            game_prime.start_turn()
            value, prior = self._nn(game_prime)
            child.total_value = value
            
        #found a normal node
        else:
            #evaluate
            value, prior = self._nn(game_prime)
            #expand
            child = _TreeNode(game_prime, value, prior, is_turn_end)
            child.parent = node
            node.child[action_index] = child

        return value, game_prime.active_player_num


    def backup(self,node,value,player_num):

        if node.game.active_player_num != player_num:
            v = -value
        else:
            v = value
        node.total_value += v

        if node.parent is not None:
            self.backup(node.parent,value,player_num)

    def search(self):
        #condition check
        if self._root.is_turn_end:
            raise Exception('root turn end')

        under_node,action_index = self.select(self._root)
        value, player_num = self.expand_and_evaluate(under_node,action_index)
        self.backup(under_node,value,player_num)

    def start_search(self, num_search, choice):
        self._root.visit_count += 1
        value, num = self.expand_and_evaluate(self._root, choice)
        self.backup(self._root, value, num)
        for i in range(num_search-2):
            self.search()

        v_tar = self._root.total_value/self._root.visit_count

        assert self._root.game.active_player_num != self._root.child[choice].game.active_player_num



        adv = -self._root.child[choice].total_value/self._root.child[choice].visit_count - v_tar
        return v_tar.squeeze(), adv.squeeze()


    def change_root(self, choice):
        if self._root.is_turn_end:
            raise Exception('root is turn end')
        else:
            self._root = self._root.child[choice]
            self._root.parent = None



