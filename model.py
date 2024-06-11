import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Linear_QNET(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
        
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() # loss
        
    def train_step(self, state, action, reward, next_state, done):
            """
            Perform a single training step for the model.

            Args:
                state (list): The current state of the environment.
                action (list): The action taken in the current state.
                reward (float): The reward received for taking the action.
                next_state (list): The next state of the environment after taking the action.
                done (bool): Whether the episode is done after taking the action.

            Returns:
                None
            """
            # Convert inputs to numpy arrays first
            state = np.array(state)
            next_state = np.array(next_state)
            action = np.array(action)
            reward = np.array(reward)
            # Convert arrays to tensors
            state = torch.tensor(state, dtype=torch.float)
            next_state = torch.tensor(next_state, dtype=torch.float)
            action = torch.tensor(action, dtype=torch.float)
            reward = torch.tensor(reward, dtype=torch.float)
            
            # If the state is 1D, (i.e. one state rather than multiple), add one dimension: [0, 1, 2] -> [[0, 1, 2]]
            if len(state.shape) == 1:
                state = torch.unsqueeze(state, 0)
                next_state = torch.unsqueeze(next_state, 0)
                action = torch.unsqueeze(action, 0)
                reward = torch.unsqueeze(reward, 0)
                done = (done, )
                
            # 1: Predicted Q values with current state
            prediction = self.model(state)
            
            target = prediction.clone()
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    # 2: Q_new = r + y * max(next_predicted_Q_value) -> only do this if not done
                    Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
                    
                # Update the target Q value for the selected action
                target[idx][torch.argmax(action).item()] = Q_new
            
            self.optimizer.zero_grad()  # Zeroes the gradients
            loss = self.criterion(target, prediction)
            loss.backward()  # Backpropagation: update gradients
            
            self.optimizer.step()
        