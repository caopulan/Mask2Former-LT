import torch


weights = torch.load("../output/model_0189999.pth", map_location=torch.device("cpu"))

for i in weights["model"]:
    print(i)

