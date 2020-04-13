import numpy as np
import torch
import torch.multiprocessing as mp
import multiprocessing as mp1
#mp = _mp.get_context("spawn")
#mp.set_start_method('forkserver', force=True)
#mp.set_start_method('spawn')
#mp.set_sharing_strategy('file_system')
import asyncio
import threading
import time

from datetime import datetime, timezone
from copy import deepcopy

from maml_rl.samplers.sampler import Sampler, make_env
from maml_rl.envs.utils.sync_vector_env import SyncVectorEnv
from maml_rl.episode import BatchEpisodes
from maml_rl.utils.reinforcement_learning import reinforce_loss
import torch._C
_is_in_bad_fork = getattr(torch._C, "_cuda_isInBadFork", lambda: False)

#from generic import to_pt
torch.set_num_threads(1)

def _create_consumer(queue, futures, loop=None):
    if loop is None:
        loop = asyncio.get_event_loop()
    while True:
        data = queue.get()
        #print('Thread name : '+str(threading.currentThread().getName()))
        if data is None:
            #print('breaking from : '+ str(threading.currentThread().getName()))
            break
        import pdb
        #pdb.set_trace()
        #print("In Create Consumer ")
        index, step, episodes = data
        future = futures if (step is None) else futures[step]
        if not future[index].cancelled():
            loop.call_soon_threadsafe(future[index].set_result, episodes)


class MultiTaskSampler(Sampler):
    """Vectorized sampler to sample trajectories from multiple environements.

    Parameters
    ----------
    env_name : str
        Name of the environment. This environment should be an environment
        registered through `gym`. See `maml.envs`.

    ~env_kwargs : dict
        Additional keywork arguments to be added when creating the environment.~
     We don't have kwargs or tw-env

    batch_size : int
        Number of trajectories to sample from each task (ie. `fast_batch_size`).

    ~policy : `maml_rl.policies.Policy` instance
        The policy network for sampling. Note that the policy network is an
        instance of `torch.nn.Module` that takes observations as input and
        returns a distribution (typically `Normal` or `Categorical`).~

    agent : `TWr.agent.Agent`
        The GATA agent with TWr/model.KG_Manipulation for sampling. Can we used for 
        preprocessing as well as getting distributions of actions given
        the observations.

    baseline : `maml_rl.baseline.LinearFeatureBaseline` instance
        The baseline. This baseline is an instance of `nn.Module`, with an
        additional `fit` method to fit the parameters of the model.

    env : `gym.Env` instance (optional)
        An instance of the environment given by `env_name`. This is used to
        sample tasks from. If not provided, an instance is created from `env_name`.

    seed : int (optional)
        Random seed for the different environments. Note that each task and each
        environement inside every process use different random seed derived from
        this value if provided.

    num_workers : int
        Number of processes to launch. Note that the number of processes does
        not have to be equal to the number of tasks in a batch (ie. `meta_batch_size`),
        and can scale with the amount of CPUs available instead.
    """
    def __init__(self,
                 env_name,
                 #env_kwargs,
                 batch_size,
                 agent, #policy,
                 baseline,
                 env=None,
                 seed=None,
                 num_workers=1):
        super(MultiTaskSampler, self).__init__(env_name,
                                               #env_kwargs,
                                               batch_size,
                                               agent, #policy,
                                               seed=seed,
                                               env=env)

        self.num_workers = num_workers
        #ctx = mp.get_context("spawn")
        

        self.task_queue = mp.JoinableQueue()
        self.train_episodes_queue = mp.Queue()
        self.valid_episodes_queue = mp.Queue()
        agent_lock = mp.Lock() # policy_lock = mp.Lock()
        import pdb
        #pdb.set_trace()
        print("entered sampler")

        self.workers = [SamplerWorker(index,
                                      env_name,
                                      #env_kwargs,
                                      batch_size,
                                      self.env.observation_space,
                                      self.env.action_space,
                                      self.agent, # self.policy,
                                      deepcopy(baseline),
                                      self.seed,
                                      self.task_queue,
                                      self.train_episodes_queue,
                                      self.valid_episodes_queue,
                                      agent_lock) #policy_lock)
            for index in range(num_workers)]
        import pdb
       
        print("CUDA initialized *after* init Samplers " + str(torch.cuda.is_initialized()))
        #pdb.set_trace()
        #pdb.set_trace()
        for worker in self.workers:
            worker.daemon = True
            worker.start()
            '''worker.agent.use_cuda = True
            worker.agent.policy_net.cuda()
            worker.agent.pretrained_cmd_gen_net.cuda()

            worker.baseline.agent.use_cuda = True
            worker.baseline.agent.policy_net.cuda()
            worker.baseline.agent.pretrained_cmd_gen_net.cuda()'''
            
            #worker.join()

        self._waiting_sample = False
        self._event_loop = asyncio.get_event_loop()
        self._train_consumer_thread = None
        self._valid_consumer_thread = None

        '''self.agent.policy_net.use_cuda = True
        self.agent.policy_net.cuda()
        self.agent.pretrained_cmd_gen_net.cuda()
        self.agent.policy_net.share_memory()'''
        print("finishing init")

    def sample_tasks(self, num_tasks):
        import pdb
        #pdb.set_trace()
        return self.env.unwrapped.sample_tasks(num_tasks)

    def sample_async(self, tasks, **kwargs):
        if self._waiting_sample:
            raise RuntimeError('Calling `sample_async` while waiting '
                               'for a pending call to `sample_async` '
                               'to complete. Please call `sample_wait` '
                               'before calling `sample_async` again.')

        import pdb
        #pdb.set_trace()
        for index, task in enumerate(tasks):
            self.task_queue.put((index, task, kwargs))
            #pdb.set_trace()
        import pdb
        #pdb.set_trace()
        num_steps = kwargs.get('num_steps', 1)
        futures = self._start_consumer_threads(tasks,
                                               num_steps=num_steps)
        #pdb.set_trace()
        self._waiting_sample = True
        print(torch.multiprocessing.current_process())
        print("Hello from Sample async" + str(torch.cuda.is_initialized()))
        return futures

    def sample_wait(self, episodes_futures):
        if not self._waiting_sample:
            raise RuntimeError('Calling `sample_wait` without any '
                               'prior call to `sample_async`.')

        async def _wait(train_futures, valid_futures):
            # Gather the train and valid episodes
            train_episodes = await asyncio.gather(*[asyncio.gather(*futures)
                                                  for futures in train_futures])
            valid_episodes = await asyncio.gather(*valid_futures)
            return (train_episodes, valid_episodes)

        samples = self._event_loop.run_until_complete(_wait(*episodes_futures))
        self._join_consumer_threads()
        self._waiting_sample = False
        return samples

    def sample(self, tasks, **kwargs):
        futures = self.sample_async(tasks, **kwargs)
        return self.sample_wait(futures)

    @property
    def train_consumer_thread(self):
        if self._train_consumer_thread is None:
            raise ValueError()
        return self._train_consumer_thread

    @property
    def valid_consumer_thread(self):
        if self._valid_consumer_thread is None:
            raise ValueError()
        return self._valid_consumer_thread

    def _start_consumer_threads(self, tasks, num_steps=1):
        # Start train episodes consumer thread
        import pdb
        #pdb.set_trace()
        train_episodes_futures = [[self._event_loop.create_future() for _ in tasks]
                                  for _ in range(num_steps)]
        self._train_consumer_thread = threading.Thread(target=_create_consumer,
            args=(self.train_episodes_queue, train_episodes_futures),
            kwargs={'loop': self._event_loop},
            name='train-consumer')
        self._train_consumer_thread.daemon = True
        self._train_consumer_thread.start()

        # Start valid episodes consumer thread
        valid_episodes_futures = [self._event_loop.create_future() for _ in tasks]
        self._valid_consumer_thread = threading.Thread(target=_create_consumer,
            args=(self.valid_episodes_queue, valid_episodes_futures),
            kwargs={'loop': self._event_loop},
            name='valid-consumer')
        self._valid_consumer_thread.daemon = True
        self._valid_consumer_thread.start()

        return (train_episodes_futures, valid_episodes_futures)

    def _join_consumer_threads(self):
        if self._train_consumer_thread is not None:
            self.train_episodes_queue.put(None)
            self.train_consumer_thread.join()

        if self._valid_consumer_thread is not None:
            self.valid_episodes_queue.put(None)
            self.valid_consumer_thread.join()

        self._train_consumer_thread = None
        self._valid_consumer_thread = None

    def close(self):
        if self.closed:
            return

        for _ in range(self.num_workers):
            self.task_queue.put(None)
        self.task_queue.join()
        self._join_consumer_threads()

        self.closed = True


class SamplerWorker(mp.Process): # need to pass the agent
    def __init__(self,
                 index,
                 env_name,
                 #env_kwargs,
                 batch_size,
                 observation_space,
                 action_space,
                 agent, #policy,
                 baseline,
                 seed,
                 task_queue,
                 train_queue,
                 valid_queue,
                 agent_lock): # policy_lock):
        super(SamplerWorker, self).__init__()

        env_fns = [make_env(env_name) #, env_kwargs=env_kwargs)
                   for _ in range(batch_size)]
        self.envs = SyncVectorEnv(env_fns,
                                  observation_space=observation_space,
                                  action_space=action_space)
        self.envs.seed(None if (seed is None) else seed + index * batch_size)
        self.batch_size = batch_size
        self.agent = agent # self.policy = policy
        #self.agent.use_cuda = True
        #self.agent.policy_net.cuda()
        #self.agent.pretrained_cmd_gen_net.cuda()
        #self.agent.policy_net.share_memory()
        self.baseline = baseline

        #self.baseline.agent.use_cuda = True
        #self.baseline.agent.policy_net.cuda()
        #self.baseline.agent.pretrained_cmd_gen_net.cuda()


        self.task_queue = task_queue
        self.train_queue = train_queue
        self.valid_queue = valid_queue
        self.agent_lock = agent_lock # self.policy_lock = policy_lock

    def sample(self,
               index,
               num_steps=1,
               fast_lr=0.5,
               gamma=0.95,
               gae_lambda=1.0,
               device='cpu'):
        # Sample the training trajectories with the initial policy and adapt the
        # policy to the task, based on the REINFORCE loss computed on the
        # training trajectories. The gradient update in the fast adaptation uses
        # `first_order=True` no matter if the second order version of MAML is
        # applied since this is only used for sampling trajectories, and not
        # for optimization.
        print("Hi " + str(torch.multiprocessing.current_process()))
        print("CUDA initialize before .cuda() in sample " + str(torch.cuda.is_initialized()) + " bad state : " + str(_is_in_bad_fork()))

        self.agent.policy_net.use_cuda = True
        self.agent.use_cuda = True
        self.agent.policy_net.cuda()
        self.agent.pretrained_cmd_gen_net.cuda()
        self.agent.policy_net.share_memory()


        self.baseline.agent.use_cuda = True
        self.baseline.agent.policy_net.cuda()
        self.baseline.agent.policy_net.use_cuda = True
        self.baseline.agent.pretrained_cmd_gen_net.cuda()
        self.baseline.agent.policy_net.share_memory()

        params = None
        for step in range(num_steps):
            train_episodes = self.create_episodes(params=params,
                                                  gamma=gamma,
                                                  gae_lambda=gae_lambda,
                                                  device=device)
            train_episodes.log('_enqueueAt', datetime.now(timezone.utc))
            # QKFIX: Deep copy the episodes before sending them to their
            # respective queues, to avoid a race condition. This issue would
            # cause the policy pi = policy(observations) to be miscomputed for
            # some timesteps, which in turns makes the loss explode.
            self.train_queue.put((index, step, deepcopy(train_episodes)))

            with self.agent_lock: # self.policy_lock:
                print("after lock")
                loss = reinforce_loss(self.agent, train_episodes, params=params) # self.policy, train_episodes, params=params)
                print("after RL")
                #params = self.agent.policy_net.update_params(loss, #self.policy.update_params(loss,
                  #                                 params=params,
                   #                                step_size=fast_lr,
                    #                               first_order=True)

        # Sample the validation trajectories with the adapted policy
        print("Out of lock scope")
        valid_episodes = self.create_episodes(params=params,
                                              gamma=gamma,
                                              gae_lambda=gae_lambda,
                                              device=device, valid=True)
        valid_episodes.log('_enqueueAt', datetime.now(timezone.utc))
        self.valid_queue.put((index, None, deepcopy(valid_episodes)))

    def create_episodes(self,
                        params=None,
                        gamma=0.95,
                        gae_lambda=1.0,
                        device='cpu', valid=False):
        if valid:
            print("HEYHEYHEYHEYHEYHEYHEYHEYEHYEHEYHEYHEYEHEYHEY")
        episodes = BatchEpisodes(batch_size=self.batch_size,
                                 gamma=gamma,
                                 device=device)
        episodes.log('_createdAt', datetime.now(timezone.utc))
        episodes.log('process_name', self.name)

        t0 = time.time()
        #print("create episode")
        #if params is not None:
         #   old_params = self.agent.policy_net.state_dict()
          #  self.agent.policy_net.load_state_dict(params, strict=False)
        for item in self.sample_trajectories(params=params):
            episodes.append(*item)
            #print("obs len")
            #print(len(episodes._observations_list[-1]))
        #print("Hey1Hey1Hey1")
        #print(len(episodes._observations_list))
        #print("Yo1")
        #print([len(elem) for elem in episodes._observations_list])
        #print(episodes._observations_list[-1])
        episodes.log('duration', time.time() - t0)
        #print("HeyHeyHey")
        #print(len(episodes._observations_list))
        #print(len(episodes._observations_list[0]))
        #print(episodes._observations_list[0])
        self.baseline.fit(episodes)
        #print("I just fit!")
        episodes.compute_advantages(self.baseline,
                                    gae_lambda=gae_lambda,
                                    normalize=True)
        #if params is not None:
         #   self.agent.policy_net.load_state_dict(old_params) # hope there is no race condition here! NOTE: strict=True
        #print("I just computed the ads")
        return episodes

    def sample_trajectories(self, params=None): # need to pass Agent() to the class? # need to incorporate params for valid_trajs
        _ = self.envs.reset()
        _, _, dones, infos = self.envs.step(["tw-reset"] * self.batch_size)  # HACK: since reset doesn't return `infos`.
        #print("In st")
        import pdb
        with torch.no_grad():
            ######
            ## Preprocess
            # Initialize
            prev_triplets, chosen_actions, prev_game_facts = [], [], []
            prev_step_dones, prev_scores = [], []
            for _ in range(self.batch_size):
                prev_triplets.append([])
                chosen_actions.append('tw-restart')
                prev_game_facts.append(set())
                prev_step_dones.append(0.0)
                prev_scores.append(0.0)
            ####
            # Don't need Rl^2 here but just in case
            #meta_dones = to_pt(np.zeros(self.batch_size), enable_cuda=self.agent.use_cuda, type='float')
            #meta_torch_step_rewards = to_pt(np.zeros(self.batch_size), enable_cuda=self.agent.use_cuda, type='float')
            #meta_prev_h = to_pt(np.zeros((1, self.batch_size, self.agent.policy_net.block_hidden_dim)), enable_cuda=self.agent.use_cuda, type='float')
            ####
            #print("Before qhile loo")
            while not self.envs.dones.all():
                #print("Just enetered the loop")
                observations = [info["feedback"] for info in infos["infos"]]
                info_for_agent = [info for info in infos["infos"]]
                observation_strings, current_triplets, action_candidate_list, dict_info_for_agent, _, current_game_facts = self.agent.get_game_info_at_certain_step_maml(info_for_agent, prev_actions=chosen_actions, prev_facts=None)
                #print("Hey after get game info")
                observation_strings = [item + " <sep> " + a for item, a in zip(observation_strings, chosen_actions)]
                #print("just before acting")
                print("CUDA initialize before act() st " + str(torch.cuda.is_initialized()) + " agent polict cuda : " + str(next(self.agent.policy_net.parameters()).is_cuda))
                value, chosen_actions, action_log_probs, chosen_indices, _, prev_h, prev_c = self.agent.act(observation_strings, current_triplets, action_candidate_list) ## incorporate params
                #print("after acting")
                chosen_actions = [(action if not done else "restart") for done, action in zip(dones, chosen_actions)]
                chosen_actions_before_parsing = [(item[idx] if not done else "*restart*") for item, idx, done in zip(dict_info_for_agent["admissible_commands"], chosen_indices, dones)]
                #print(chosen_actions_before_parsing)
                #print(chosen_indices)
                #print("after choosing actions")
                ######
                # TODO:
                # observations_tensor = torch.from_numpy(observations) ## Do we realy want numpy? If so, i will need to demarcate inside the agent.act and essentialy write the function explicitly write here--easy
                # pi = self.policy(observations_tensor, params=params)
                # actions_tensor = pi.sample()
                # actions = actions_tensor.cpu().numpy()
                # actions = ["look"] * self.batch_size
                #actions = chosen_actions_before_parsing.cpu().numpy()

                new_observations, rewards, dones, infos = self.envs.step(chosen_actions_before_parsing)
                batch_ids = infos['batch_ids']
                yield (observations, current_triplets, action_candidate_list, chosen_actions_before_parsing, chosen_indices, rewards, batch_ids)
                observations = new_observations
                prev_actions = chosen_actions_before_parsing

    def run(self):
        print("inside run " + str(torch.cuda.is_initialized()))
        print("Run before cuda: process name " + str(mp.current_process().name) + " cuda :" + str(next(self.agent.policy_net.parameters()).is_cuda))
        #self.agent.policy_net.use_cuda = True
        #self.agent.policy_net.cuda()
        #self.agent.pretrained_cmd_gen_net.cuda()

        #self.baseline.agent.use_cuda = True
        #self.baseline.agent.policy_net.cuda()
        #self.baseline.agent.pretrained_cmd_gen_net.cuda()
        print("Run after cuda: process name " + str(mp.current_process().name) + " cuda :" + str(next(self.agent.policy_net.parameters()).is_cuda))
        while True:
            data = self.task_queue.get()
            print("data " + str(data))
            #print("Hey from run sw")
            import pdb
            if data is None:
                print("about to break")
                self.envs.close()
                self.task_queue.task_done()
                print("About to exit : "+str(mp.current_process().name))
                break

            index, task, kwargs = data
            #print(data)
            self.envs.reset_task(task)
            self.sample(index, **kwargs)
            self.task_queue.task_done()
