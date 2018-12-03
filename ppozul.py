import tensorflow as tf
import numpy as np
import azul
from copy import deepcopy
import valuetree
from multiprocessing import Process,Pipe,Queue
import random


class Config(object):
    input_size = 155
    output_size = 180
    num_layer = 4
    num_hidden = 256
    dtype = tf.float32
    epsilon = 0.15
    c1 = 1
    c2 = 0.01
    normalize_advantages = True
    learning_rate = 1e-4
    epoch = 10
    z_factor = 0.5
    batch_size = 256

    num_env = 8
    max_path_length = 100

class Nn_estimator(object):
    def __init__(self, config):
        self._config = config
        input_size = config.input_size
        output_size = config.output_size
        dtype = config.dtype

        self._graph = tf.Graph()
        with self._graph.as_default():
            with tf.name_scope('input_layer'):
                self._input_states = tf.placeholder(dtype, [None,input_size], 'input_states')
                self._action_indices = tf.placeholder(tf.int32, [None], 'action_indices')
                self._old_action_prob = tf.placeholder(dtype, [None], 'action_index')
                self._advantages = tf.placeholder(dtype, [None], 'advantages')
                self._mask = tf.placeholder(dtype, [None,output_size], 'mask')

            with tf.name_scope('labels'):
                self._label_value = tf.placeholder(dtype, [None,1], 'label_value')

            with tf.name_scope('MLP'):
                layer_out = self._input_states
                for i in range(config.num_layer):
                    layer_out = tf.layers.dense(layer_out, config.num_hidden, tf.nn.relu, name='MLP_layer_{}'.format(i))

            with tf.name_scope('value_header'):
                self._prediction_value = tf.layers.dense(layer_out, 1, tf.nn.tanh, name='value_layer')

            with tf.name_scope('distribution_header'):
                self._logits = tf.layers.dense(layer_out, output_size, name='logits')
                masked_logits = self._logits + (self._mask - 1.) * tf.float32.max / 10
                self._prediction_distribution = tf.nn.softmax(masked_logits)

            with tf.name_scope('loss'):
                # clip loss
                action_onehot = tf.one_hot(self._action_indices, config.output_size)
                p_prob = tf.reduce_sum(self._prediction_distribution * action_onehot, axis=1)
                r_t = p_prob / self._old_action_prob
                clipped = tf.clip_by_value(r_t, clip_value_min=1-self._config.epsilon, clip_value_max=1+self._config.epsilon)
                l_clip = tf.minimum(r_t*self._advantages, clipped*self._advantages)
                l_clip = tf.reduce_mean(l_clip)
                self._l_clip = l_clip

                # value loss
                l_vf = tf.losses.mean_squared_error(labels=self._label_value, predictions=self._prediction_value)
                self._l_vf = l_vf

                # entropy
                l_s = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._prediction_distribution, logits=masked_logits)
                l_s = tf.reduce_mean(l_s)
                self._l_s = l_s

                # total loss
                self._loss = - l_clip + self._config.c1 * l_vf - self._config.c2 * l_s

            with tf.name_scope('train'):
                optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
                self._train_op = optimizer.minimize(self._loss)


            init = tf.global_variables_initializer()
            self._saver = tf.train.Saver()

        self._sess = tf.Session(graph = self._graph)

        try:
            self.restore()
        except ValueError as e:
            print(e)
            self._sess.run(init)
            self.save()

        # writer = tf.summary.FileWriter("./tensorboard/log/", self._sess.graph)
        # writer.close()


    def train(self, states, actions, act_prob, advs, values, mask):
        assert len(states) >= self._config.batch_size

        if self._config.normalize_advantages:
            adv_mean = np.mean(advs)
            adv_std = np.std(advs)
            advs = (advs-adv_mean)/(adv_std+1e-8)


        for i in range(self._config.epoch):
            permutated_index = np.random.permutation(len(states))
            permutated_index = permutated_index[:self._config.batch_size]
            feed_dict = {
                self._input_states:states[permutated_index],
                self._action_indices:actions[permutated_index],
                self._old_action_prob:act_prob[permutated_index],
                self._advantages:advs[permutated_index],
                self._label_value:values[permutated_index],
                self._mask:mask[permutated_index]
            }
            l_clip, l_vf, l_s, _ = self._sess.run([self._l_clip, self._l_vf, self._l_s, self._train_op], feed_dict=feed_dict)
            print('clip_loss:', l_clip, 'value_loss', l_vf, 'entropy', l_s)
        self.save()

    def predict(self, states, mask):
        feed_dict = {self._input_states:states, self._mask:mask}
        value_p, distribution_p = self._sess.run([self._prediction_value,self._prediction_distribution],feed_dict=feed_dict)
        return value_p, distribution_p

    def save(self, path="./model/latest.ckpt"):
        self._saver.save(self._sess, path)

    def restore(self, path="./model/latest.ckpt"):
        self._saver.restore(self._sess, path)

    def close(self):
        self._sess.close()

class Agent(object):
    def __init__(self, nn, config=Config()):
        self._nn = nn
        self._sshape = config.input_size
        self._adim = config.output_size
        self._max_path_length = config.max_path_length
        self._normalize_advantages = config.normalize_advantages
        self._num_env = config.num_env
        self._commands = np.argwhere(np.ones((6,5,6))==1)
        self._z_factor = config.z_factor

    def make_envs(self):
        self._env_dics = []
        for _ in range(self._num_env):
            dic = {
            'env':azul.Azul(2),
            'envs':[],
            'obs':[],
            'masks':[],
            'acs':[],
            'probs':[],
            'terminals':[],
            'done':False,
            'winner':0
            }
            self._env_dics.append(dic)

    def sample_trajectory(self):
        # init
        envs_ob, envs_mask = [],[]
        for dic in self._env_dics:
            game = dic['env']
            game.start()
            dic['envs'].append(deepcopy(game))
            ob = game.states()
            mask = game.flat_mask()
            dic['obs'].append(ob)
            dic['masks'].append(mask)
            envs_ob.append(ob)
            envs_mask.append(mask)
        envs_ob = np.stack(envs_ob)
        envs_mask = np.stack(envs_mask)
        steps = 0

        # stepping loop
        while True:
            _, envs_prob = self._nn.predict(envs_ob, envs_mask)
            envs_ob, envs_mask = [],[]
            for i, dic in enumerate(self._env_dics):
                if dic['done']:
                    envs_ob.append(np.zeros(shape=self._sshape))
                    envs_mask.append(np.ones(shape=self._adim))
                    continue
                act_idx = np.random.choice(self._adim,p=envs_prob[i])
                dic['acs'].append(act_idx)
                dic['probs'].append(envs_prob[i,act_idx])

                command = self._commands[act_idx]
                is_turn_end = dic['env'].take_command(command)
                if is_turn_end:
                    dic['env'].turn_end(verbose = False)
                    if dic['env'].is_terminal:
                        dic['env'].final_score(verbose=True)
                        print('end in', dic['env'].turn)
                        dic['done'] = True
                        dic['winner'] = dic['env'].leading_player_num
                        envs_ob.append(np.zeros(shape=self._sshape))
                        envs_mask.append(np.ones(shape=self._adim))
                        dic['terminals'].append(-1)
                        continue
                    else:
                        dic['env'].start_turn()
                        dic['terminals'].append(1)
                else:
                    dic['terminals'].append(0)


                dic['envs'].append(deepcopy(dic['env']))
                ob, mask = dic['env'].states(), dic['env'].flat_mask()
                dic['obs'].append(ob)
                dic['masks'].append(mask)
                envs_ob.append(ob)
                envs_mask.append(mask)
            envs_ob = np.stack(envs_ob)
            envs_mask = np.stack(envs_mask)
            steps += 1

            is_done = True
            for dic in self._env_dics:
                if not dic['done']:
                    is_done = False
            if is_done:
                break

    @property
    def dics(self):
        return self._env_dics
    

class NNHelper(object):
    def __init__(self, w2s_conn):
        self._w2s_conn = w2s_conn
        self._dummy = azul.Azul(2)
        self._dummy.start()

    def __call__(self, game=None):
        if game == None:
            game = self._dummy
        self._w2s_conn.send((game.states(), game.flat_mask(), False))
        value, prior = self._w2s_conn.recv()
        return value, prior

def worker_routine(data, w2s_conn, public_q):
    helper = NNHelper(w2s_conn)
    values, advs = [],[]
    for env, state, mask, action, prob, winner in data:
        search = valuetree.ValueSearch(env, helper)
        value, adv = search.start_search(100, action)

        if env.active_player_num == winner:
            value = 0.2 + 0.8*value

        else:
            value = -0.2 + 0.8*value

        values.append(value)
        advs.append(adv)
    envs, states, masks, actions, probs, winners = list(zip(*data))
    w2s_conn.send((None,None,True))
    public_q.put((states, masks, actions, probs, values, advs))



def server_routine(s2w_conns, nn, num_processes=8):
    done = False
    while True:
        states,masks = [],[]
        for i in range(num_processes):
            state, mask, flag = s2w_conns[i].recv()
            if flag:
                done = True
                break
            states.append(state)
            masks.append(mask)
        if done:
            break
        states = np.stack(states, axis=0)
        masks = np.stack(masks, axis=0)

        values, priors = nn.predict(states, masks)

        for i in range(num_processes):
            s2w_conns[i].send((values[i], priors[i]))



if __name__ == '__main__':
    config = Config()
    nn = Nn_estimator(config)

    for trl in range(100):
        print(trl)
        agent = Agent(nn)
        agent.make_envs()
        agent.sample_trajectory()
        dics = agent.dics

        envs, states, masks, actions, probs, winners = [[] for _ in range(6)]
        for dic in dics:
            for env, state, mask, action, prob in zip(dic['envs'],dic['obs'],dic['masks'],dic['acs'],dic['probs']):
                    envs.append(env)
                    states.append(state)
                    masks.append(mask)
                    actions.append(action)
                    probs.append(prob)
                    winners.append(dic['winner'])
        data = list(zip(envs, states, masks, actions, probs, winners))
        random.shuffle(data)
        chunk_size = len(data)//8
        processes = []
        s2w_conns = []
        public_q = Queue()
        for i in range(8):
            data_chunk = data[i*chunk_size:(i+1)*chunk_size]
            w2s_conn, s2w_conn = Pipe()
            s2w_conns.append(s2w_conn)
            p = Process(target=worker_routine, args=(data_chunk, w2s_conn, public_q))
            processes.append(p)

        for p in processes:
            p.start()

        server_routine(s2w_conns, nn)


        states_f, masks_f, actions_f, probs_f, values_f, advs_f = [[] for _ in range(6)]
        for i in range(8):
            states, masks, actions, probs, values, advs = public_q.get()
            states_f.extend(states)
            masks_f.extend(masks)
            actions_f.extend(actions)
            probs_f.extend(probs)
            values_f.extend(values)
            advs_f.extend(advs)

        for p in processes:
            p.join()


        states_f = np.stack(states_f)
        masks_f = np.stack(masks_f)
        actions_f = np.stack(actions_f)
        probs_f = np.stack(probs_f)
        values_f = np.stack(values_f).reshape((-1,1))
        advs_f = np.stack(advs_f)

        nn.train(states_f, actions_f, probs_f, advs_f, values_f, masks_f)