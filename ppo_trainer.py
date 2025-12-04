
from config import *
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal



def collect_traj(env, pol, vn):
    states = []
    acts = []
    rews = []
    loggs = []
    vals = []
    dns = []
    ep_rews = []
    s, _ = env.reset()
    er = 0
    
    for i in range(MAX_TIMESTEPS):
        st = torch.FloatTensor(s).unsqueeze(0)
        with torch.no_grad():
            m, ls = pol(st)
            std = ls.exp()
            d = Normal(m, std)
            a = d.sample()
            lp = d.log_prob(a).sum(-1)
            v = vn(st)
        
        an = a.squeeze(0).numpy()
        next, r, term, trunc, _ = env.step(an)
        flag = term or trunc
        
        states.append(s)
        acts.append(an)
        rews.append(r)
        loggs.append(lp.item())
        vals.append(v.item())
        dns.append(flag)
        
        er += r
        s = next
        last_flag = flag
        

        if flag:
            ep_rews.append(er)
            er = 0
            s, _ = env.reset()
            
    

    if er > 0:
        ep_rews.append(er)

       
    if last_flag:
        last_val = 0.0
    else:
        with torch.no_grad():
            st = torch.FloatTensor(s).unsqueeze(0)
            last_val = vn(st).item()
    

    avg_reward = np.mean(ep_rews) if ep_rews else 0
    return states, acts, rews, loggs, vals + [last_val], dns, avg_reward



def comp_gae(rews, vals, dns):
    advs = []
    gae = 0
    
    for i in reversed(range(len(rews))):
        nv = vals[i + 1]
        delta = rews[i] + GAMMA * nv * (1 - dns[i]) - vals[i]
        gae = delta + GAMMA * LAMBDA * (1 - dns[i]) * gae
        advs.insert(0, gae)

    returns = [adv + v for adv, v in zip(advs, vals[:-1])]
    return advs, returns


def ppo_upd(pol, vn, optim_policy, optim_value, states, acts, old_lps, advs, returns):
    states = torch.FloatTensor(np.array(states))
    acts = torch.FloatTensor(np.array(acts))
    old_lps = torch.FloatTensor(old_lps)
    advs = torch.FloatTensor(advs)
    returns = torch.FloatTensor(returns)
    
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    
    for _ in range(10):
        idx = np.arange(len(states))
        np.random.shuffle(idx)
        
        for st in range(0, len(states), BATCH_SIZE):
            end = st + BATCH_SIZE
            i = idx[st : end]
            bStates = states[i]
            bActs = acts[i]
            bOldLps = old_lps[i]
            bAdvs = advs[i]
            bReturns = returns[i]
            
            m, ls = pol(bStates)
            std = ls.exp()
            d = Normal(m, std)
            new_lps = d.log_prob(bActs).sum(-1)
            ratio = torch.exp(new_lps - bOldLps)
            s1 = ratio * bAdvs
            s2 = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * bAdvs
            
            ent = d.entropy().sum(-1).mean()  # Sum across action dims, mean across batch
            loss_policy = -torch.min(s1, s2).mean() - ENT_COEF * ent
            
            optim_policy.zero_grad()
            loss_policy.backward()
            nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
            optim_policy.step()
            
            vp = vn(bStates).squeeze(-1)
            loss_v = nn.MSELoss()(vp, bReturns)
            
            optim_value.zero_grad()
            loss_v.backward()
            nn.utils.clip_grad_norm_(vn.parameters(), 0.5)
            optim_value.step()