import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms,datasets
from tqdm import tqdm
from model import resnet18
from torch.utils.data import DataLoader
import torchvision.models.resnet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    data_root = os.path.abspath(os.path.join(os.getcwd(),"../resnet18-mnist"))
    image_path = os.path.join(data_root,"data_set","mnist")
    assert  os.path.exists(image_path),"{} path does not exist.".format(image_path)
    train_dataset = datasets.MNIST(
        root='./data_set',
        train=True,
        transform=train_transforms,
        download=True
    )
    test_dataset=datasets.MNIST(
        root='./data_set',
        train=False,
        transform=val_transforms
    )

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {}  dataloader workers every process'.format(nw))
    train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True,num_workers= nw)
    test_loader = DataLoader(test_dataset,batch_size=16,shuffle=False,num_workers = nw)

    train_num = len(train_dataset)
    mnist_list = train_dataset.class_to_idx
    cla_dict = dict((val,key) for key,val in mnist_list.items())
    json_str = json.dumps(cla_dict,indent=4)## json.dumps()是把python对象转换成json对象的一个过程，
    # 生成的是字符串。cla_dict是转化成json的对象，indent:参数根据数据格式缩进显示，读起来更加清晰。
    with open('class_indices.json','w') as json_file:
        json_file.write(json_str)
    test_num = len(test_dataset)
    print("Using {} images for training,{} images for test".format(train_num,test_num))

    net = resnet18()
    # model_weight_path = "./resnet18-pre.pth"
    # assert  os.path.exists(model_weight_path),"file {} does not exist.".format(model_weight_path)
    # missing_keys,unexpected_keys = net.load_state_dict(torch.load(model_weight_path),strict=False)#载入模型权重
    in_channel = net.fc.in_features # # 这里的fc是model.ResNet里的fc;in_features是输入特征矩阵的深度
    net.fc = nn.Linear(in_channel,10)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.0001)
    epochs = 10
    best_acc = 0.0
    save_path ='./resnNet18.pth'
    trainn_steps = len(train_loader)
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step,data in enumerate(train_bar):
            images , labels =data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits,labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss +=loss.item()
            train_bar.desc = "train epoch[{}/{} loss:{:.3f}".format(epoch+1,epochs,loss)

        net.eval()
        acc = 0.0
        with torch.no_grad():
            test_bar = tqdm(test_loader,colour = 'green')
            for test_data  in  test_bar:
                test_images,test_labels =test_data
                outputs = net(test_images.to(device))
                predict_y = torch.max(outputs,dim=1)[1]
                acc +=torch.eq(predict_y,test_labels.to(device)).sum().item()
                test_bar.desc = "test epoch[{}/{}]".format(epoch+1,epochs)

        test_accurate = acc/test_num
        print('[epoch %d] train）loss : %.3f  test_accuracy'%(epoch+1,running_loss/trainn_steps,test_accurate) )

        if test_accurate>best_acc:
            best_acc = test_accurate
            torch.save(net.state_dict(),save_path)
    print('Finished Training')

if __name__ == '__main__':
    main()


























