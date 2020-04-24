import numpy as np
import torch
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence

from maml_rl.utils.torch_utils import weighted_normalize

class BatchEpisodes(object):
    def __init__(self, batch_size, gamma=0.95, device='cpu'):
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

        #self._observations_list = [[] for _ in range(batch_size)]
        #self._observations_ids_list = [[] for _ in range(batch_size)]
        # self._observatons_embed_list = [[] for _ in range(batch_size)]

        #self._triplets_list = [[] for _ in range(batch_size)]
        #self._input_node_name_list = [[] for _ in range(batch_size)]
        #self._input_relation_name_list = [[] for _ in range(batch_size)]

        self._adj_mats_list = [[] for _ in range(batch_size)]
        #self._candidates_list = [[] for _ in range(batch_size)]
        self._action_ids_list = [[] for _ in range(batch_size)]
        #self._actions_list = [[] for _ in range(batch_size)]
        self._rewards_list = [[] for _ in range(batch_size)]
        self._chosen_indices_list = [[] for _ in range(batch_size)]
        #self._input_node_names_list = [[] for _ in range(batch_size)]
        #self._relation_names_list = [[] for _ in range(batch_size)]

        #self._observation_shape = None
        #self._action_shape = None
        self._adj_mat_shape = None
        self._chosen_indices_shape = None

        #self._observations = None
        #self._observations_embed = None
        #self._observations_ids = None
        self._adj_mats = None
        self._action_ids = None
        #self._triplets = None
        self._chosen_indices = None
        #self._candidates = None
        #self._actions = None
        self._rewards = None
        self._returns = None
        self._mask = None
        self._advantages = None
        self._lengths = None
        self._logs = {}

    '''@property
    def observation_shape(self):
        if self._observation_shape is None:
            self._observation_shape = self.observations.shape[2:]
        return self._observation_shape

    @property
    def action_shape(self):
        if self._action_shape is None:
            self._action_shape = self.actions.shape[2:]
        return self._action_shape'''

    @property
    def adj_mat_shape(self):
        if self._adj_mat_shape is None:
            self._adj_mat_shape = self.adj_mats.shape[2:]
        return self._adj_mat_shape

    @property
    def chosen_indices_shape(self):
        if self._chosen_indices_shape is None:
            self._chosen_indices_shape = self.chosen_indices.shape[2:]
        return self._chosen_indices_shape

    '''@property
    def observations(self):
        if self._observations is None:
            self._observations = self._observations_list
            #print(len(self._observations))
            #print(len(self._observations[0]))
            del self._observations_list
        return self._observations'''

    '''@property
    def observations(self):
        if self._observations is None:
            observations = [pad_sequence(self._observations_list[i], batch_first=True) for i in range(self.batch_size)]
            padded_observations = pad_sequence(observations, batch_first=False)
            self._observations = torch.as_tensor(padded_observations, device=self.device)
            del self._observations_list

        return self._observations

    @property
    def observations_embed(self):
        if self._observations_embed is None:
            observations = [pad_sequence(self._observations_embed_list[i], batch_first=True) for i in range(self.batch_size)]
            padded_observations = pad_sequence(observations, batch_first=False)
            self._observations_embed = torch.as_tensor(padded_observations, device=self.device)
            del self._observations_embed_list

        return self._observations_embed'''

    '''@property
    def actions(self):
        if self._actions is None:
            self._actions = self._actions_list
            del self._actions_list
        return self._actions'''

    @property
    def chosen_indices(self):
        if self._chosen_indices is None:
            chosen_indices_shape = self._chosen_indices_list[0][0].shape
            #print(str(len(self)) + " len " + str(self.batch_size) + " bs ")
            chosen_indices = np.zeros((len(self), self.batch_size) + chosen_indices_shape,
                               dtype=np.float32)
            for i in range(self.batch_size):
                length = self.lengths[i]
                np.stack(self._chosen_indices_list[i], axis=0, out=chosen_indices[:length, i])
            self._chosen_indices = torch.as_tensor(chosen_indices, device=self.device)
            del self._chosen_indices_list
        return self._chosen_indices

    @property
    def action_ids(self):
        if self._action_ids is None:
            ##action_ids = [pad_sequence(ele, batch_first=True) for elem in self._action_ids_list]
            ##padded_action_ids = pad_sequence(action_ids, batch_first=True)
            ##self._action_ids = torch.as_tensor(padded_action_ids).permute(1,0,2,3) # hard coded permute
            #max_num_can = max([max(ele.shape[0] for ele in elem) for elem in self._action_ids_list])
            #max_can_len = max([max(ele.shape[1] for ele in elem) for elem in self._action_ids_list])
            #action_ids_shape = self._action_ids_list[0][0].shape # len X bs X num_can X can_len
            '''action_ids_shape = (max_num_can, max_can_len)
            print(str(len(self)) + " len " + str(self.batch_size) + " bs " + str(type(self._action_ids_list[0][0])) + " type " + str(max_num_can) + " max num can " + str(max_can_len) + " max can len ")
            action_ids = np.zeros((len(self), self.batch_size) + action_ids_shape,
                               dtype=np.float32)
            print("action ids shape " + str(action_ids_shape) + " action ids " + str(action_ids.shape))
            for i in range(self.batch_size):
                length = self.lengths[i]
                for j in range(length):
                    np.stack(self._action_ids_list[i], axis=0, out=action_ids[:length, i, :self._action_ids_list[i].shape[0], :self._action_ids_list[i].shape[1]])
            self._action_ids = torch.as_tensor(action_ids, device=self.device)'''
            del self._action_ids_list
        return self._action_ids

    @property
    def adj_mats(self):
        if self._adj_mats is None:
            adj_mats_shape = self._adj_mats_list[0][0].shape
            adj_mats = np.zeros((len(self), self.batch_size) + adj_mats_shape,
                                    dtype=np.float32)
            for i in range(self.batch_size):
                length = self.lengths[i]
                np.stack(self._adj_mats_list[i],
                         axis=0,
                         out=adj_mats[:length, i])
            self._adj_mats = torch.as_tensor(adj_mats, device=self.device)
            del self._adj_mats_list
        return self._adj_mats
        

    '''@property
    def chosen_indices(self):
        if self._chosen_indices is None:
            self._chosen_indices = self._chosen_indices_list
            del self._chosen_indices_list
        return self._chosen_indices'''
        

    '''@property
    def triplets(self):
        if self._triplets is None:
            self._triplets = self._triplets_list
            del self._triplets_list
        return self._triplets'''

    @property
    def candidates(self):
        if self._candidates is None:
            self._candidates = self._candidates_list
            del self._candidates_list
        return self._candidates

    @property
    def rewards(self):
        if self._rewards is None:
            rewards = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = self.lengths[i]
                np.stack(self._rewards_list[i], axis=0, out=rewards[:length, i])
            self._rewards = torch.as_tensor(rewards, device=self.device)
            del self._rewards_list
        return self._rewards

    @property
    def returns(self):
        if self._returns is None:
            self._returns = torch.zeros_like(self.rewards)
            return_ = torch.zeros((self.batch_size,), dtype=torch.float32)
            for i in range(len(self) - 1, -1, -1):
                return_ = self.gamma * return_ + self.rewards[i] * self.mask[i]
                self._returns[i] = return_
        return self._returns

    @property
    def mask(self):
        if self._mask is None:
            self._mask = torch.zeros((len(self), self.batch_size),
                                     dtype=torch.float32,
                                     device=self.device)
            for i in range(self.batch_size):
                length = self.lengths[i]
                self._mask[:length, i].fill_(1.0)
        return self._mask

    @property
    def advantages(self):
        if self._advantages is None:
            raise ValueError('The advantages have not been computed. Call the '
                             'function `episodes.compute_advantages(baseline)` '
                             'to compute and store the advantages in `episodes`.')
        return self._advantages

    #def append(self, observations, input_ids, triplets, adj_mats, candidates, action_input_ids, actions, chosen_indices, rewards, batch_ids):
    def append(self, adj_mats, candidates, action_input_ids, actions, chosen_indices, rewards, batch_ids):
        #print("adj_mats " + str(adj_mats.shape) + " candidates " + str(len(candidates)) + " action_input_ids " + str(action_input_ids.shape))
        # print(str(type(input_ids)) + " "+str(len(input_ids)) + " input_ids")
        # for observation, input_id, triplet, adj_mat, candidate, action_input_id, action, chosen_index, reward, batch_id in zip(
         #       observations, input_ids, triplets, adj_mats, candidates, action_input_ids, actions, chosen_indices, rewards, batch_ids):
        for adj_mat, action_input_id, chosen_index, reward, batch_id in zip(
               adj_mats, action_input_ids, chosen_indices, rewards, batch_ids):
            if batch_id is None:
                continue
            #self._observations_list[batch_id].append(observation)
            #self._triplets_list[batch_id].append(triplet)
            #self._input_node_names_list[batch_id].append(input_node_name)
            #self._relation_names_list[batch_id].append(input_relation_name)
            #self._candidates_list[batch_id].append(candidate)
            #self._actions_list[batch_id].append(action)
            self._chosen_indices_list[batch_id].append(chosen_index)
            self._rewards_list[batch_id].append(reward.astype(np.float32))
            #self._observations_ids_list[batch_id].append(input_id)
            self._adj_mats_list[batch_id].append(adj_mat)
            self._action_ids_list[batch_id].append(action_input_id)
            #print("in append")
            #print(str(adj_mat.shape) + " " + str(action_input_id.shape) + " " + str(input_relation_name.shape) + " " + str(input_node_name.shape))
            #print(str(action_input_id) + " " + str(input_relation_name) + " " + str(input_node_name))
            #break
            #print(str(action_input_id) + " " + str(len(triplet)))
            #print(str(adj_mat.shape) + " " + str(len(triplet)))

    @property
    def logs(self):
        return self._logs

    def log(self, key, value):
        self._logs[key] = value

    def compute_advantages(self, baseline, agent, gae_lambda=1.0, normalize=True):
        # Compute the values based on the baseline
        values = baseline(self, agent).detach().t() # not sure if this should be reshaped/ t()
        # Add an additional 0 at the end of values for
        # the estimation at the end of the episode
        values = F.pad(values * self.mask, (0, 0, 0, 1))

        # Compute the advantages based on the values
        deltas = self.rewards + self.gamma * values[1:] - values[:-1]
        self._advantages = torch.zeros_like(self.rewards)
        gae = torch.zeros((self.batch_size,), dtype=torch.float32)
        for i in range(len(self) - 1, -1, -1):
            gae = gae * self.gamma * gae_lambda + deltas[i]
            self._advantages[i] = gae

        # Normalize the advantages
        if normalize:
            self._advantages = weighted_normalize(self._advantages,
                                                  lengths=self.lengths)
        # Once the advantages are computed, the returns are not necessary
        # anymore (only to compute the parameters of the baseline)
        del self._returns
        del self._mask

        return self.advantages

    @property
    def lengths(self):
        if self._lengths is None:
            self._lengths = [len(rewards) for rewards in self._rewards_list]
        return self._lengths

    def __len__(self):
        return max(self.lengths)
