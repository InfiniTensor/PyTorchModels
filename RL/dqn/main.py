import torch,os 
import torch.nn as nn  
import torch.optim as optim  
import numpy as np  
import gym
import argparse
from torch.utils.data import DataLoader, TensorDataset 
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ['SDL_VIDEODRIVER'] = 'dummy'  
# 设定GPU  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
  
# DQN模型  
class DQN(nn.Module):  
    def __init__(self, n_states, n_actions):  
        super(DQN, self).__init__()  
        self.fc1 = nn.Linear(n_states, 128)  
        self.fc2 = nn.Linear(128, n_actions)  
  
    def forward(self, x):  
        x = torch.relu(self.fc1(x))  
        x = self.fc2(x)  
        return x  

def train(save_path,num_episodes,lr):  
    # 初始化环境  
    env = gym.make('CartPole-v1', render_mode="human")  
    n_states = env.observation_space.shape[0]  
    n_actions = env.action_space.n  
    
    # 创建DQN模型和优化器  
    q_net = DQN(n_states, n_actions).to(device)  
    optimizer = optim.Adam(q_net.parameters(), lr=lr)  
    loss_fn = nn.MSELoss()  
    
    # 训练参数  
    # num_episodes = 1000  
    gamma = 0.99  
    epsilon = 1.0  
    epsilon_decay = 0.995  
    epsilon_min = 0.01  
    
    # 训练过程  
    for i_episode in range(num_episodes):  
        state = env.reset()[0] 
        state = torch.tensor([state], dtype=torch.float32).to(device)  
        done = False  
        while not done:  
            if np.random.rand() > epsilon:  
                with torch.no_grad():  
                    action_values = q_net(state)  
                    action = action_values.argmax().item()  
            else:  
                action = np.random.randint(n_actions)  
    
            next_state, reward, done, info,_ = env.step(action)
            next_state = torch.tensor([next_state], dtype=torch.float32).to(device)  
    
            # 简化处理，这里不使用经验回放  
            # 直接用于训练  
            optimizer.zero_grad()  
            q_values = q_net(state)  
            q_value = q_values.gather(1, torch.tensor([[action]], dtype=torch.int64).to(device))  
            next_q_values = q_net(next_state).detach().max(1)[0].unsqueeze(1)
            done_float = torch.tensor(done, dtype=torch.float32).to(device)
            q_target = reward + gamma * next_q_values * (1 - done_float.float())  
            loss = loss_fn(q_value, q_target)  
            loss.backward()  
            optimizer.step()  
    
            state = next_state  
    
            # 递减epsilon  
            epsilon = max(epsilon_min, epsilon * epsilon_decay)  
    
        if i_episode % 10 == 0:  
            logger.info(f'Episode {i_episode}, Loss: {loss.item()}') 
            torch.save(q_net.state_dict(),os.path.join(save_path,"%s.pth"%i_episode))
    env.close()

def infer(model_path):
    env = gym.make('CartPole-v1', render_mode="human")  
    n_states = env.observation_space.shape[0]  
    n_actions = env.action_space.n  
    
    # 创建DQN模型和优化器  
    q_net = DQN(n_states, n_actions).to(device)
    # 加载模型  
    q_net_loaded = DQN(n_states, n_actions).to(device)  
    q_net_loaded.load_state_dict(torch.load(model_path, map_location=device))  
    q_net_loaded.eval()  # 设置为评估模式  
    
    # 现在你可以使用q_net_loaded进行推理了  
    # 假设我们有一个新的环境实例env_new  
    
    # 初始化状态  
    state = env.reset()[0]
    state = torch.tensor([state], dtype=torch.float32).to(device)  
    
    # 推理过程  
    done = False
    episode_return = 0.0
    while not done:  
        with torch.no_grad():  
            action_values = q_net_loaded(state)  
            action = action_values.argmax().item()  
    
        next_state, reward, done, info,xx = env.step(action)  
        next_state = torch.tensor([next_state], dtype=torch.float32).to(device)  

        # 如果需要，可以在这里添加代码来处理推理结果  
        episode_return +=reward
        state = next_state  
    logger.info("return:%s"%episode_return)
    # 完成后关闭环境  
    env.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",default="./output/0.pth",type=str)
    parser.add_argument("--num_episodes",default=1000,type=int)
    parser.add_argument("--lr",default=0.0001,type=float)
    parser.add_argument("--save_path",default="./output",type=str)
    parser.add_argument("--infer", action="store_true",default=False)

    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path,exist_ok=True)
    if args.infer:
        infer(model_path=args.model_path)
    else:
        train(args.save_path,args.num_episodes,args.lr)