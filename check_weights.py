import torch

org_path = "logs/market1501/bagtricks_R50_reproduce3/model_best.pth"
sr_path = 'logs/market1501/bagtricks_R50_prune/model_final.pth'

model_org = torch.load(org_path, map_location=torch.device("cpu"))['model']
model_sr = torch.load(sr_path, map_location=torch.device("cpu"))['model']


import pdb; pdb.set_trace()

""" all count weight"""

org_count = 0 
all_count = 0
for name in model_org.keys():
    if name.endswith('weight'):
        # import pdb; pdb.set_trace()
        weight = model_org[name]
        org_count += torch.sum( torch.where(torch.abs(weight)<=1e-10, 1, 0))
        all_count += torch.sum( torch.where(torch.abs(weight)>=1e-10, 1, 0))
print
print(org_count)


sr_count = 0 
for name in model_sr.keys():
    if name.endswith('weight'):
        # import pdb; pdb.set_trace()
        weight = model_sr[name]
        sr_count += torch.sum( torch.where(torch.abs(weight)<=1e-10, 1, 0))
print(sr_count)


# import pdb; pdb.set_trace()

print('++++++++++++++++++++++++++==')

