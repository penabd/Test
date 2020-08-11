import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    '''
    For size 8x8:
        conv1 = nn.Conv2d(1,8,3)
        conv2 = nn.Conv2d(8,32,3)
        conv3 = nn.Conv2d(32,64,3)
        
        fc1 = nn.Linear(256, 512)
        fc2 = nn.Linear(512,5)
    
    For size 8x10:
        conv1 = nn.Conv2d(1,8,3)
        conv2 = nn.Conv2d(8,32,3)
        conv3 = nn.Conv2d(32,64,3)
        
        fc1 = nn.Linear(512,512)
        fc2 = nn.Linear(512,5)
        
    For size 8x12:
        conv1 = nn.Conv2d(1,8,3)
        conv2 = nn.Conv2d(8,32,3)
        conv3 = nn.Conv2d(32,64,3)
        
        fc1 = nn.Linear(768,1024)
        fc2 = nn.Linear(1024,5)
    
    For size 15x25:
        conv1 = nn.Conv2d(1,8,3)
        conv2 = nn.Conv2d(8,32,3)
        conv3 = nn.Conv2d(32,64,3)
        
        fc1 = nn.Linear(10944,1024)
        fc2 = nn.Linear(1024,5)
    
    For size 150x300:
        conv1 = nn.Conv2d(1,8,3)
        conv2 = nn.Conv2d(8,32,3)
        conv3 = nn.Conv2d(32,64,3)
        
        fc1 = nn.Linear(2709504,1024)
        fc2 = nn.Linear(1024,5)
        
    '''
    def __init__(self, ALPHA):
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        
    
        #linear_input_size = convw * convh * 32
        linear_input_size = 10944
        
        self.fc1 = nn.Linear(linear_input_size, 1024)
        self.fc2 = nn.Linear(1024, 5)
        #self.optimizer = optim.SGD(self.parameters(), lr=self.ALPHA, momentum=0.9)
        self.optimizer = optim.RMSprop(self.parameters(), lr=ALPHA)
        #decay learning rate
        self.lr_scheduler = T.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.8)
        self.loss = nn.MSELoss()

    def forward(self, observation):
        observation = T.Tensor(observation)
  #      print(f"observation shape is {observation.shape}")
    
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))
        
  #      print(f"post conv obs shape {observation.shape}")
        observation = observation.view(observation.shape[0],-1)
        
        observation = F.relu(self.fc1(observation))
        actions = self.fc2(observation)
        return actions

class Agent(object):
    def __init__(self, gamma, epsilon, alpha,
                 maxMemorySize, epsEnd=0.05,
                 replace=10000, actionSpace=[0,1,2,3,4,5]):
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_END = epsEnd
        self.ALPHA = alpha
        self.actionSpace = actionSpace
        self.memSize = maxMemorySize
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = []
        self.memCntr = 0
        self.replace_target_cnt = replace
        self.Q_eval = DeepQNetwork(alpha)
        self.Q_next = DeepQNetwork(alpha)

    def storeTransition(self, state, action, reward, state_):
        if self.memCntr < self.memSize:
            self.memory.append([state, action, reward, state_])
        else:
            self.memory[self.memCntr%self.memSize] = [state, action, reward, state_]
        self.memCntr += 1

    def chooseAction(self, observation):
        rand = np.random.random()
        observation = np.expand_dims(observation, axis=0)
        actions = self.Q_eval.forward(observation)
        #print(actions.shape)
        #print(actions)
       # print(T.argmax(actions).item())
        if rand < 1 - self.EPSILON:
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.actionSpace)
        self.steps += 1

        return action

    def learn(self, batch_size):
        self.Q_eval.optimizer.zero_grad()
        if self.replace_target_cnt is not None and \
           self.learn_step_counter % self.replace_target_cnt == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

        if self.memCntr+batch_size < self.memSize:
            memStart = int(np.random.choice(range(self.memCntr)))
        else:
            memStart = int(np.random.choice(range(self.memSize-batch_size-1)))
        miniBatch=self.memory[memStart:memStart+batch_size]
        memory = np.array(miniBatch)

        # convert to list because memory is an array of numpy objects
      #  print(f"list memory is {len(list(memory[:,0][:]))} and {memory[:,0][:].shape}")
        Qpred = self.Q_eval.forward(list(memory[:,0][:]))
        Qnext = self.Q_next.forward(list(memory[:,3][:]))

        maxA = T.argmax(Qnext, dim=1)
        rewards = T.Tensor(list(memory[:,2]))
        Qtarget = Qpred.clone()
        indices = np.arange(batch_size)
        Qtarget[indices,maxA] = rewards + self.GAMMA*T.max(Qnext[1])

        if self.steps > 500:
            if self.EPSILON - 1e-4 > self.EPS_END:
                self.EPSILON -= 1e-4
            else:
                self.EPSILON = self.EPS_END

        #Qpred.requires_grad_()
        loss = self.Q_eval.loss(Qtarget, Qpred)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1
        
        self.Q_eval.lr_scheduler.step()
      #  self.Q_next.lr_scheduler.step()
      
        return loss