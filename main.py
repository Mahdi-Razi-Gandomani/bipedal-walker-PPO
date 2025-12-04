import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from config import *
from models import PolNet, ValNet
from ppo_trainer import collect_traj, comp_gae, ppo_upd




if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    env = gym.make('BipedalWalker-v3')
    env.reset(seed=42)

    sd = env.observation_space.shape[0]
    ad = env.action_space.shape[0]
    pol = PolNet(sd, ad)
    vn = ValNet(sd)
    optim_policy = optim.Adam(pol.parameters(), lr=lr)
    optim_value = optim.Adam(vn.parameters(), lr=lr)

    total_point_history = []
    for i in range(MAX_EPISODES):
        states, acts, rews, loggs, vals, dns, avg_reward = collect_traj(env, pol, vn)
        total_point_history.append(avg_reward)
        
        av_latest_points = np.mean(total_point_history[-NUM_P_AV:])
        print(f"\rEpisode {i+1}.          Average of the last {NUM_P_AV} episodes: {av_latest_points:.1f}", end="")

        if (i+1) % NUM_P_AV == 0:
            print(f"\rEpisode {i+1}.          Average of the last {NUM_P_AV} episodes: {av_latest_points:.1f}")
        

        advs, returns = comp_gae(rews, vals, dns)
        ppo_upd(pol, vn, optim_policy, optim_value, states, acts, loggs, advs, returns)

    torch.save(pol.state_dict(), 'final.pth')
    env.close()



    # Testing
    test_env = gym.make('BipedalWalker-v3', render_mode='human')
    pol.eval()
    for ep_num in range(5):
        s, _ = test_env.reset()
        total = 0
        done = False
        while not done:
            st = torch.FloatTensor(s).unsqueeze(0)
            with torch.no_grad():
                m, ls = pol(st)
                a = m
            
            an = a.squeeze(0).numpy()
            s, r, term, trunc, _ = test_env.step(an)
            total += r
            done = term or trunc
        
        print(total)

    test_env.close()



