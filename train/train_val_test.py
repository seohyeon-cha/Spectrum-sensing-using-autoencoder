import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

def train(model, trainloader, validloader, epochs, optimizer, criterion):
    min_valid_loss = 1e-3
    train_loss_list = []
    device = get_device()

    for e in range(epochs):
        running_loss = 0.0
        model.train()     # Optional when not using Model Specific layer
        for data, label in trainloader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs,data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(trainloader)
        train_loss_list.append(train_loss)
        
        running_loss = 0.0
        model.eval()     # Optional when not using Model Specific layer

        for data, label in validloader:
            data, label = data.to(device), label.to(device)
            outputs = model(data)
            loss = criterion(outputs, data)
            running_loss += loss.item() 
        valid_loss = running_loss / len(validloader)


        print(f'(Epoch {e+1}) \nTraining Loss: {train_loss} \nValidation Loss: {valid_loss}')
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.8f}--->{valid_loss:.8f}) \nSaving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(model.state_dict(), 'model_save/pretrain_model2.pth')
    return train_loss_list

def test(model, testloader, criterion):
    device = get_device()
    running_loss = 0
    reconstructed_data = torch.Tensor().to(device)
    latent_vector = torch.Tensor().to(device)
    with torch.no_grad():
        for data, label in testloader:
            data, label = data.to(device), label.to(device)
            enc_output = model.encoder(data)
            recon_data = model(data)
            reconstructed_data = torch.cat((reconstructed_data, recon_data),0)
            latent_vector = torch.cat((latent_vector, enc_output), 0)
            loss = criterion(recon_data, data)
            running_loss += loss.item()
        test_loss = running_loss / len(testloader)
    
    print(f'Test Loss: {test_loss}')
    return reconstructed_data, latent_vector


