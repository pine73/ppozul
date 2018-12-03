import random
import numpy as np
import sys
import pickle

class _TileManager(object):
    """docstring for _TileManager"""
    colors = ['R','B','W','Y','I','T']

    def __init__(self):
        super(_TileManager, self).__init__()
        self.tiles = []
        

    @property
    def is_empty(self):
        if len(self.tiles) == 0:
            return True
        else:
            return False    

    def state(self):
        s = np.zeros([5])
        for tile in self.tiles:
            s[tile.number//20] += 1
        return s

    def receive(self, tiles):
        self.tiles.extend(tiles)

    def clean(self):
        self.tiles = []



class _Tile(object):
    """docstring for _Tile"""
    color_dict = {0:'R',1:'B',2:'W',3:'Y',4:'I',5:'T'}

    def __init__(self, number):
        assert isinstance(number, int) and number <= 100 and number >= 0, 'int between 0 100'
        self.number = number
        self.color = _Tile.color_dict[self.number//20]


class _Tray(_TileManager):
    """docstring for _Tray"""
    def __init__(self):
        super(_Tray, self).__init__()
    


class _Pool(_TileManager):
    """docstring for _Pool"""
    def __init__(self):
        super(_Pool, self).__init__()
        self.has_token = True

    def reset(self):
        self.has_token = True

    def state(self):
        s = np.zeros([6])
        for tile in self.tiles:
            s[tile.number//20] += 1
        if self.has_token:
            s[5] += 1
        return s


class _Bag(_TileManager):
    """docstring for _Bag"""
    def __init__(self):
        super(_Bag, self).__init__()
        for i in range(100):
            self.tiles.append(_Tile(i))
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.tiles)

    def fill_tray(self, trays, grave, verbose=False):
        self.shuffle()
        for tray in trays:
            if len(self.tiles) < 4:
                if grave.is_empty:
                    if verbose:
                        print('out of tiles')
                    tray.receive(self.tiles[:4])
                    self.tiles = self.tiles[4:]
                    break
                else:
                    self.receive(grave.tiles)
                    grave.clean()
                    self.shuffle()
                    if len(self.tiles) < 4:
                        if verbose:
                            print('recycled and out of tiles')
                        tray.receive(self.tiles[:4])
                        self.tiles = self.tiles[4:]
                        break
                    else:
                        if verbose:
                            print('recycled')
                        tray.receive(self.tiles[:4])
                        self.tiles = self.tiles[4:]
            else:
                tray.receive(self.tiles[:4])
                self.tiles = self.tiles[4:]
        return trays

class _Grave(_TileManager):
    """docstring for _Grave"""
    def __init__(self):
        super(_Grave, self).__init__()

    

class _Buffer(_TileManager):
    """docstring for _Buffer"""
    def __init__(self):
        super(_Buffer, self).__init__()
        self.rows = []
        for i in range(5):
            self.rows.append(_Row(i+1))

    def state(self):
        states = []
        for row in self.rows:
            states.append(row.state())
        return np.stack(states)

        
class _Row(_TileManager):
    """docstring for _Row"""
    def __init__(self, cap):
        super(_Row, self).__init__()
        assert isinstance(cap, int) and cap <= 5 and cap >= 1, 'row cap: int between 1 5'
        self._cap = cap

    def receive(self, tiles, floor):
        self.tiles.extend(tiles)
        if len(self.tiles) > self._cap:
            floor.receive(self.tiles[self._cap:])
            self.tiles = self.tiles[:self._cap]

    @property
    def is_full(self):
        if len(self.tiles) == self._cap:
            return True
        else:
            return False
        
    
        
class _Board(_TileManager):
    """docstring for _Board"""
    def __init__(self):
        super(_Board, self).__init__()
        self.im = np.zeros((5,5),int)

    def state(self):
        return self.im.astype(np.float32)

    def score(self, indexes):
        assert self.im[indexes] == 0 , 'invalid'
        r,c = indexes
        rs, cs = [0]*2
        column = self.im[:,c]
        row = self.im[r,:]
        # row score
        for i in range(c-1,-1,-1):
            if row[i] == 0:
                break
            else:
                rs += 1
        for i in range(c+1,5):
            if row[i] == 0:
                break
            else:
                rs += 1
        # column score
        for i in range(r-1,-1,-1):
            if column[i] == 0:
                break
            else:
                cs += 1
        for i in range(r+1,5):
            if column[i] == 0:
                break
            else:
                cs += 1
        if rs != 0 and cs != 0:
            rs += 1
        self.im[r,c]  = 1
        return rs+cs+1




class _Floor(_TileManager):
    """docstring for _Floor"""
    def __init__(self):
        super(_Floor, self).__init__()
        self.has_token = False

    def score(self):
        s = 0
        num = len(self.tiles)
        if self.has_token:
            num += 1 
        for i in range(num):
            if i <= 1:
                s -= 1
            elif i <= 4:
                s -= 2
            else :
                s -= 3
        return s
    
    def state(self):
        s = np.zeros([6])
        for tile in self.tiles:
            s[tile.number//20] += 1
        if self.has_token:
            s[5] += 1
        return s


class _Player(object):
    """docstring for _Player"""
    num_player = 1

    def __init__(self):
        super(_Player).__init__()
        self.buffer = _Buffer()
        self.board = _Board()
        self.floor = _Floor()
        self.score = 0
        self.num = (_Player.num_player + 1) % 2 + 1
        _Player.num_player += 1
        

class Azul(object):
    """docstring for Azul"""
    def __init__(self, num_player):
        self._num_player = num_player
        self._players = []
        self._bag = _Bag()
        self._grave = _Grave()
        self._trays = []
        self._pool = _Pool()
        self._active_player = None
        self._turn = None

        if num_player == 1 or num_player == 2:
            for i in range(5):
                tray = _Tray()
                self._trays.append(tray)
        else:
            raise Exception('wrong player number')

        for i in range(num_player):
            self._players.append(_Player())

    @property
    def active_player_num(self):
        return self._active_player.num

    @property
    def leading_player_num(self):
        max_score = 0
        num = None
        for player in self._players:
            if player.score >= max_score:
                num = player.num
                max_score = player.score
        return num

    @property
    def turn(self):
        return self._turn
    
    
    


    def start(self):
        self._bag.fill_tray(self._trays, self._grave)
        self._turn = 1
        self._active_player = self._players[0]

    def display(self):
        print('Turn:{} Active Player:{}'.format(int(self._turn), self._active_player.num))

        bag_s = self._bag.state()
        grave_s = self._grave.state()
        tray_s = []
        pool_s = self._pool.state()
        for tray in self._trays:
            tray_s.append(tray.state())
        for player in self._players:
            buffer_s = player.buffer.state()
            board_s = player.board.state()
            floor_s = player.floor.state()

            print('Player {} score: {}'.format(player.num, player.score))
            for i, row_s in enumerate(buffer_s):
                print('Buffer{}:R{},B{},W{},Y{},I{}'.format(i+1,*row_s.astype(int)))
            print('Board:\n',board_s)
            print('Floor:R{},B{},W{},Y{},I{},T{}'.format(*floor_s.astype(int)))

            

        print('Bag:R{},B{},W{},Y{},I{}'.format(*bag_s.astype(int)))
        print('Recycling:R{},B{},W{},Y{},I{}'.format(*grave_s.astype(int)))
        for i, tray_s in enumerate(tray_s):
            print('Tray{}:R{},B{},W{},Y{},I{}'.format(i+1,*tray_s.astype(int)))
        print('Pool:R{},B{},W{},Y{},I{},T{}'.format(*pool_s.astype(int)))

    def states(self):
        bag_s = self._bag.state()
        grave_s = self._grave.state()
        tray_s = []
        pool_s = self._pool.state()
        for tray in self._trays:
            tray_s.append(tray.state())

        player = self._active_player
        buffer_s = player.buffer.state().flatten()
        board_s = player.board.state().flatten()
        floor_s = player.floor.state()

        player = self._players[self._active_player.num%2]
        buffer2_s = player.buffer.state().flatten()
        board2_s = player.board.state().flatten()
        floor2_s = player.floor.state()
        

        full_state = np.concatenate([bag_s, grave_s, *tray_s, pool_s, buffer_s, board_s, floor_s, buffer2_s, \
            board2_s, floor2_s, [self._turn], [self.active_player_num]])
        return full_state

    def mask(self):
        color_dict = {0:'R',1:'B',2:'W',3:'Y',4:'I',5:'T'}
        dict_color = {'R':2,'B':0,'W':4,'Y':1,'I':3}

        player = self._active_player

        mask = np.ones((6,5,6))

        # -2
        valid = []
        for i,tray in enumerate(self._trays):
            if tray.is_empty:
                mask[i,:,:] = 0
            else:
                valid.append((tray,i))
        if self._pool.is_empty:
            mask[5,:,:] = 0
        else:
            valid.append((self._pool,5))

        # -3
        for source, index in valid:
            for i in range(5):

                taken = []
                left = []
                for tile in source.tiles:
                    if tile.color == color_dict[i]:
                        taken.append(tile)
                    else:
                        left.append(tile)
                if len(taken) == 0:
                    mask[index,i,:] = 0

        # -4 -5
        for color_index in range(5):
            for target_index in range(5):
                target = player.buffer.rows[target_index]
                # -4
                if (not target.is_empty) and (target.tiles[0].color != color_dict[color_index]):
                    mask[:,color_index,target_index] = 0
                # -5
                elif target.is_empty and player.board.im[target_index,(dict_color[color_dict[color_index]]+target_index)%5] != 0:
                    mask[:,color_index,target_index] = 0

        return mask

    def flat_mask(self):
        mask = self.mask()
        return mask.flatten()



    def take(self, command, player):
        color_dict = {'R':2,'B':0,'W':4,'Y':1,'I':3}

        # validation check
        command = command.upper()
        if not (len(command.strip()) == 3 and command[0] in '12345P' and command[1] in 'RBWYI' and command[2] in '12345F'):
            print('eg. 1r3 pyf')
            return -1

        if command[0] == 'P':
            source = self._pool
        else:
            source = self._trays[int(command[0])-1]
        if source.is_empty:
            print('source empty')
            return -2

        taken = []
        left = []
        for tile in source.tiles:
            if tile.color == command[1]:
                taken.append(tile)
            else:
                left.append(tile)
        if len(taken) == 0:
            print('color invalid')
            return -3

        if command[2] == 'F':
            target = player.floor
        else:
            target = player.buffer.rows[int(command[2])-1]
            if (not target.is_empty) and target.tiles[0].color != command[1]:
                print('color does not agree with target buffer')
                return -4
            elif target.is_empty and player.board.im[int(command[2])-1,(color_dict[command[1]]+int(command[2])-1)%5] != 0:
                print('color in conflict with board')
                return -5

        # take action
        if command[0] == 'P':
            if source.has_token:
                source.has_token = False
                player.floor.has_token = True
        if command[2] != 'F':
            target.receive(taken, player.floor)
        else :
            target.receive(taken)
        source.clean()
        self._pool.receive(left)
        return 0

    def take_stdin(self):
        print('take your move')
        while self.take(sys.stdin.readline(), self._active_player) < 0 : pass
        print('-----------------------------------------\n')
        self._active_player = self._players[self._active_player.num%self._num_player]

        all_empty = self._pool.is_empty
        for tray in self._trays:
            all_empty = all_empty and tray.is_empty
        return all_empty


    # index command eg. (1,2,3)
    def take_command(self, command):
        color_dict = {0:'R',1:'B',2:'W',3:'Y',4:'I',5:'T'}
        source_index, color_index, target_index = command
        color = color_dict[color_index]
        source = source_index+1 if source_index !=5 else 'P'
        target = target_index+1 if target_index !=5 else 'F'
        command_string = '{}{}{}'.format(source, color, target)

        self.take(command_string, self._active_player)

        # print('Move:{}\n-----------------------------------------\n'.format(command_string))

        self._active_player = self._players[self._active_player.num%self._num_player]
        all_empty = self._pool.is_empty
        for tray in self._trays:
            all_empty = all_empty and tray.is_empty
        # return is turn end
        return all_empty

    
    def turn_2p(self):
        self.display()
        while self.take_stdin() == False: self.display()
        self.turn_end()

    def turn_2ai(self, policy):
        # self.display()
        while True:
            command = policy(self)
            if self.take_command(command) == True: break
            # self.display()
        self.turn_end(verbose = False)
    
    def aivs(self,policy1,policy2):
        self.start()
        while True:
            while True:
                if self._active_player.num == 1:
                    command = policy1(self)
                else:
                    command = policy2(self)
                if self.take_command(command) == True: break
            self.turn_end(verbose = False)
            if self.is_terminal:break
            self.start_turn()
        self.final_score(True)
        return self.leading_player_num


    ############### debug
    def _save(self):
        with open('azul-debug.pkl','wb') as f:
            pickle.dump(self, f)
    ##############

    def turn_end(self, verbose = True):
        for tray in self._trays:
            assert tray.is_empty

        color_dict = {'R':2,'B':0,'W':4,'Y':1,'I':3}
        for player in self._players:
            for row in player.buffer.rows:
                if row.is_full:
                    indexes = (row._cap-1, (color_dict[row.tiles[0].color]+row._cap-1)%5)
                    player.board.receive([row.tiles[0]])
                    self._grave.receive(row.tiles[1:])
                    row.clean()
                    player.score += player.board.score(indexes)
            player.score += player.floor.score()
            self._grave.receive(player.floor.tiles)
            player.floor.clean()
            if player.score < 0 :
                player.score = 0
            if player.floor.has_token:
                player.floor.has_token = False
                self._pool.reset()
                self._active_player = player
        if verbose:
            print('turn end')

    def start_turn(self):
        self._bag.fill_tray(self._trays, self._grave)
        self._turn += 1

    @property    
    def is_terminal(self):
        for player in self._players:
            terminal = np.sum(player.board.im, axis = 1)
            if 5 in terminal:
                return True
        return False

    def final_score(self,verbose=False):

        for player in self._players:
            row = np.sum(player.board.im, axis = 1)
            column = np.sum(player.board.im, axis = 0)
            color = np.zeros((5),int)
            for tile in player.board.tiles:
                color[tile.number//20] += 1
            for i,j,k in zip(row,column,color):
                if i == 5:
                    player.score += 2
                if j == 5:
                    player.score += 7
                if k == 5:
                    player.score += 10
            if verbose:
                print(player.score)
        if verbose:
            print('game over')
        
if __name__ == '__main__':
            game = Azul(2)
            
            game.start()
            game.display()
            


            while True:
                game.turn_2p()
                if game.is_terminal:
                    break
                game.start_turn()


            # with open('azul-debug.pkl','rb') as f:
            #     game  = pickle.load(f)
            # game._states()
            # game.display()
            
