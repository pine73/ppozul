import tensorflow as tf
import numpy as np
import gym

class Config(object):
    input_size = None
    output_size = None
    num_layer = 2
    num_hidden = 128
    dtype = tf.float32
    epsilon = 0.2
    c1 = 0.5
    c2 = 0.01
    gamma = 0.99
    normalize_advantages = True
    learning_rate = 3e-4
    epoch = 10

    num_env = 16
    max_path_length = 1000


        


class Nn_estimator(object):
    def __init__(self, config):
        self._config = config
        input_size = config.input_size
        output_size = config.output_size
        dtype = config.dtype

        with tf.name_scope('input_layer'):
            self._input_states = tf.placeholder(dtype, [None,input_size], 'input_states')
            self._action_indices = tf.placeholder(tf.int32, [None], 'action_indices')
            self._old_action_prob = tf.placeholder(dtype, [None], 'action_index')
            self._advantages = tf.placeholder(dtype, [None], 'advantages')

        with tf.name_scope('labels'):
            self._label_value = tf.placeholder(dtype, [None,1], 'label_value')
            self._label_distribution = tf.placeholder(tf.int32, [None], 'label_distribution')

        with tf.name_scope('MLP'):
            layer_out = self._input_states
            for i in range(config.num_layer):
                layer_out = tf.layers.dense(layer_out, config.num_hidden, tf.nn.relu, name='MLP_layer_{}'.format(i))

        with tf.name_scope('value_header'):
            self._prediction_value = tf.layers.dense(layer_out, 1, name='value_layer')

        with tf.name_scope('distribution_header'):
            self._logits = tf.layers.dense(layer_out, output_size, name='logits')
            self._prediction_distribution = tf.nn.softmax(self._logits)
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
            l_s = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._prediction_distribution, logits=self._logits)
            l_s = tf.reduce_mean(l_s)

            # total loss
            self._loss = - l_clip + self._config.c1 * l_vf - self._config.c2 * l_s

        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
            self._train_op = optimizer.minimize(self._loss)


        init = tf.global_variables_initializer()
        self._saver = tf.train.Saver()

        self._sess = tf.Session()

        try:
            self.restore()
        except ValueError as e:
            print(e)
            self._sess.run(init)
            self.save()

        # writer = tf.summary.FileWriter("./tensorboard/log/", self._sess.graph)
        # writer.close()


    def train(self, states, actions, act_prob, advs, values):
        feed_dict = {
            self._input_states:states,
            self._action_indices:actions,
            self._old_action_prob:act_prob,
            self._advantages:advs,
            self._label_distribution:actions,
            self._label_value:values
        }
        for i in range(self._config.epoch):
            l_clip, l_vf, _ = self._sess.run([self._l_clip, self._l_vf, self._train_op], feed_dict=feed_dict)
        # print('clip_loss:', l_clip, 'value_loss', l_vf)
        self.save()

    def predict(self, states):
        feed_dict = {self._input_states:states}
        value_p, distribution_p = self._sess.run([self._prediction_value,self._prediction_distribution],feed_dict=feed_dict)
        return value_p, distribution_p

    def save(self, path="./model/latest.ckpt"):
        self._saver.save(self._sess, path)

    def restore(self, path="./model/latest.ckpt"):
        self._saver.restore(self._sess, path)

    def close(self):
        self._sess.close()


class Agent(object):
    def __init__(self, env_name, config=Config()):
        env = gym.make(env_name)
        config.input_size = env.observation_space.shape[0]
        config.output_size = env.action_space.n
        self._nn = Nn_estimator(config)
        self._sshape = config.input_size
        self._adim = config.output_size
        self._max_path_length = config.max_path_length
        self._gamma = config.gamma
        self._normalize_advantages = config.normalize_advantages
        self._env_name = env_name
        self._num_env = config.num_env


    def make_envs(self):
        self._env_dics = []
        for _ in range(self._num_env):
            dic = {
            'env':gym.make(self._env_name),
            'obs':[],
            'acs':[],
            'probs':[],
            'rewards':[],
            'next_obs':[],
            'terminals':[],
            'done':False
            }
            self._env_dics.append(dic)

    def sample_trajectory(self):
        envs_ob = []
        for dic in self._env_dics:
            ob = dic['env'].reset()
            dic['obs'].append(ob)
            envs_ob.append(ob)
        envs_ob = np.stack(envs_ob)
        steps = 0
        while True:
            _, envs_prob = self._nn.predict(envs_ob)
            envs_ob = []
            for i, dic in enumerate(self._env_dics):
                if dic['done']:
                    envs_ob.append(np.zeros(shape=self._sshape))
                    continue
                act_idx = np.random.choice(self._adim,p=envs_prob[i])
                dic['acs'].append(act_idx)
                dic['probs'].append(envs_prob[i,act_idx])
                ob, rew, done, _ = dic['env'].step(act_idx)
                envs_ob.append(ob)
                dic['next_obs'].append(ob)
                dic['rewards'].append(rew)
                if done or steps > self._max_path_length:
                    dic['terminals'].append(1)
                    dic['done'] = True
                else:
                    dic['terminals'].append(0)
                    dic['obs'].append(ob)

            steps += 1

            is_done = True
            for dic in self._env_dics:
                if not dic['done']:
                    is_done = False
            if is_done:
                offset = steps % 5 if steps % 5 != 0 else 5
                obs_t, next_obs_t, rewards_t, terminals_t, acs_t, probs_t = [], [], [], [], [], []
                for dic in self._env_dics:
                    obs_t.extend(dic['obs'][steps-offset:steps])
                    next_obs_t.extend(dic['next_obs'][steps-offset:steps])
                    rewards_t.extend(dic['rewards'][steps-offset:steps])
                    terminals_t.extend(dic['terminals'][steps-offset:steps])
                    acs_t.extend(dic['acs'][steps-offset:steps])
                    probs_t.extend(dic['probs'][steps-offset:steps])
                obs_t = np.stack(obs_t)
                next_obs_t = np.stack(next_obs_t)
                rewards_t = np.stack(rewards_t)
                terminals_t = np.stack(terminals_t)
                acs_t = np.stack(acs_t)
                probs_t = np.stack(probs_t)
                self.update_nn(obs_t, next_obs_t, rewards_t, terminals_t, acs_t, probs_t)

                total_reward = 0
                for dic in self._env_dics:
                    total_reward += np.sum(dic['rewards'])
                print('average reward', total_reward/len(self._env_dics), '\n')
                break

            if steps % 5 == 0:
                offset = 5
                obs_t, next_obs_t, rewards_t, terminals_t, acs_t, probs_t = [], [], [], [], [], []
                for dic in self._env_dics:
                    obs_t.extend(dic['obs'][steps-offset:steps])
                    next_obs_t.extend(dic['next_obs'][steps-offset:steps])
                    rewards_t.extend(dic['rewards'][steps-offset:steps])
                    terminals_t.extend(dic['terminals'][steps-offset:steps])
                    acs_t.extend(dic['acs'][steps-offset:steps])
                    probs_t.extend(dic['probs'][steps-offset:steps])
                obs_t = np.stack(obs_t)
                next_obs_t = np.stack(next_obs_t)
                rewards_t = np.stack(rewards_t)
                terminals_t = np.stack(terminals_t)
                acs_t = np.stack(acs_t)
                probs_t = np.stack(probs_t)
                self.update_nn(obs_t, next_obs_t, rewards_t, terminals_t, acs_t, probs_t)


            envs_ob = np.stack(envs_ob)

    def estimate_advantage(self, obs, next_obs, rewards, terminals):
        v0, _ = self._nn.predict(obs)
        v1, _ = self._nn.predict(next_obs)
        qs = rewards + (1 - terminals) * self._gamma * v1.flatten()
        advs = qs - v0.flatten()

        if self._normalize_advantages:
            adv_mean = np.mean(advs)
            adv_std = np.std(advs)
            advs = (advs-adv_mean)/(adv_std+1e-8)

        return advs, qs.reshape([-1,1])


    def update_nn(self, obs, next_obs, rewards, terminals, acs, probs):
        advs, values = self.estimate_advantage(obs, next_obs, rewards, terminals)
        self._nn.train(obs, acs, probs, advs, values)




if __name__ == '__main__':

    agent =Agent('LunarLander-v2')
    for i in range(100):
        agent.make_envs()
        agent.sample_trajectory()
