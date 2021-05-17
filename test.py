import os
import json
import builtins #dir()函数接受模块名作为参数，返回一个排好序的字符串列表，内容是一个模块里定义过的名字]
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import resnet18
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES =True #加载截断图片

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose(
        transforms.ToTensor(),
        transforms.Normalize([0.1317],[0.3081])
    )
    img  = builtins.open("./1.jpg","rb")
    img = Image.open(img).to(device)
    plt.imshow(img)
    img  = data_transform(img)
    img = torch.unsqueeze(img,dim=0)

    json_path = './class_indices.json'
    assert  os.path.exists(json_path),"file : '{}' does not exist.".format(json_path)
    json_file = open(json_path,"r")
    class_indict = json.load(json_file)

    model = resnet18(num_classes=10).to(device)
    weights_path = "./resNet18.pth"
    assert  os.path.exists(weights_path),"file: '{}' does not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path),map_location=device)
    model.eval()
    with torch.no_grad():
        output = torch.sequeeze(model(img))
        predict  = torch.softmax(output,dim=0)
        predict_cla = torch.argmax(predict).numpy()
    print_res = "class:{} prob:{:.3}".format(class_indict[str(predict_cla)],
                                             predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()

if __name__ =="__main__":
    main()











