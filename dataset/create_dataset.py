import torch

class CustomDataset(): 
  def __init__(self,data,label):
    self.x_data = data
    self.y_data = label

  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.x_data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx): 
    x = torch.FloatTensor(self.x_data[idx])
    y = torch.Tensor(self.y_data[idx])
    return x, y