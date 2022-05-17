import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model.autoencoder import deepAE, finenet
from train_lte_nr import return_loader
from train.train_val_test import get_device

####### Add additional linear layer to pretrained network #############

lte_file = 'dataset/data_lte.mat'
nr_file = 'dataset/data_nr.mat'

device = get_device()
pre_trained_net = deepAE().to(device)
path = 'model_save/pretrain_model2.pth'
pre_trained_net.load_state_dict(torch.load(path))
test_dataloader = return_loader(lte_file, nr_file)[2]

fine_model = finenet(pre_trained_net.encoder).to(device)
print(fine_model)

############ Fine-tuning (Training) #################

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(fine_model.parameters(), lr=1e-3)
# num_epochs = 2


# for epoch in range(num_epochs):
#     running_loss = 0
#     for data, label in test_dataloader:
#         data, label = data.to(device), label.to(device)
        
#         label = label.type(torch.LongTensor)
#         label = torch.flatten(label).to(device)
        
#         optimizer.zero_grad()
#         output = fine_model(data)
#         loss = criterion(output, label)
        
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
        
#     finetune_loss = running_loss/len(test_dataloader)
#     print(f'(Epoch {epoch+1}) \nFine-tuning Loss: {finetune_loss}')
# torch.save(fine_model.state_dict(), 'model_save/finetuned_model2.pth')
# torch.save(fine_model.state_dict(), 'model_save/no_pretrain.pth')
############ Classification accuracy on fine-tuned network #################

correct = 0
total = 0

path = 'model_save/finetuned_model2.pth'
fine_model.load_state_dict(torch.load(path))

print("Model's state_dict:")
for param_tensor in fine_model.state_dict():
    print(param_tensor, "\t", fine_model.state_dict()[param_tensor].size())

fine_model.eval()
with torch.no_grad():
    for data, label in test_dataloader:

        data, label = data.to(device), label.to(device)
        output = fine_model(data)
        _, predicted = torch.max(output.data, 1)
        
        label = label.type(torch.LongTensor)
        label = torch.flatten(label).to(device)

        total += label.size(0)
        correct += (predicted==label).sum().item()
print(f'Accuracy of the network on the {total} test data: {100 * correct / total} %')


###################################################################################
# 문제점이 fine-tuning 했을 때 accuracy가 test data에 NR data 넣어준 만큼 된다는거 (60%)
# 두 개의 class 구분 못하는듯 
