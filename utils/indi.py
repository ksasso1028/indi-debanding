import torch
import random
from tqdm import tqdm

def get_indi_step(dry, deterministic=True,steps=10,test=False):
    # deterministic approach to train on same steps.
    if test:
        t = torch.ones(dry.size(0))
        fct = t[:, None, None, None]
    else:
        if deterministic:
            bins = torch.linspace(0, 1, steps + 1)
            noise_levels = []
            for x in range(0, dry.size(0)):
                step = random.randint(0, steps)
                noise_levels.append(bins[step])

            t = torch.tensor(noise_levels).float()
            fct = t[:, None, None, None]
        else:
            # get random value between 0  and 1
            t = torch.rand(size=(dry.shape[0],))
            fct = t[:, None, None, None]
    return fct, t


def indi_transform(fct, clean, dirty):
    if fct.dim() > clean.dim():
        fct = fct.squeeze(1)
    transformed = (1 - fct) * clean + fct * dirty
    return transformed


def sample(net, x,steps):
    net.eval()
    with torch.no_grad():
        for t in tqdm(torch.linspace(1,0, steps+1,device=x.device)[:-1]):
            time = torch.tensor(t).unsqueeze(0)
            wav = net(x , time)
            fct = 1/(steps * t)
            x = (fct) * wav + (1-fct) * x
    return x
