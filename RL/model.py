import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ClaimingModel(nn.Module):
    def __init__(self):
        super(ClaimingModel, self).__init__()
        self.inputSize = 34*13
        self.outputSize = 6
        self.hiddenSize1 = 512
        self.hiddenSize2 = 256
        self.hiddenSize3 = 256
        
        self.network=nn.Sequential(
            nn.Linear(self.inputSize, self.hiddenSize1),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.hiddenSize1, self.hiddenSize2),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.hiddenSize2, self.hiddenSize3),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.hiddenSize3, self.outputSize),
            nn.Softmax()
        )
        self.apply(self.init_weight)

    def init_weight(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            module.bias.data.fill_(0)
    def forward(self, input):
        input=torch.tensor(input).float()
        return self.network(input)
    




if __name__=='__main__':
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else :"""
    device=torch.device('cpu')

    Model=ClaimingModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer=optim.Adam(Model.parameters(), lr=0.0001)
    batch_size=10

    pengSet=np.load('../dataset/peng.npy')
    chiSet=np.load('../dataset/chi.npy')
    gangSet=np.load('../dataset/gang.npy')
    bugangSet=np.load('../dataset/bugang.npy')
    angangSet=np.load('../dataset/angang.npy')
    PassSet=np.load('../dataset/Pass.npy')
    print('Finishing loading Data!!')
    # print( len(PassSet), len(chiSet), len(pengSet),  len(gangSet), len(bugangSet), len(angangSet), )
    # 2511705 232248 177092 7037 9655 5110 


    
    """
    claiming: pass chi peng gang bugang angang 
    label:    0    1   2    3    4      5
    """    
    
    # print('Finishing preparing training dataset!!')
    

    
    
    
    #testloader = torch.utils.data.DataLoader(TestingData, batch_size=batch_size,
    #                                shuffle=False,)
    
    runningLoss=0
    losses=[]
    pasAcc=[]
    chiAcc=[]
    pengAcc=[]
    bugangAcc=[]
    gangAcc=[]
    angangAcc=[]
    
    totAccuracy=[]
    Model.train()
    for epoch in range(10000):
        TrainingData=[]
        idx=np.random.randint(len(PassSet), size=100)
        T=PassSet[idx,:]
        for t in T:
            TrainingData.append((t,[1,0,0,0,0,0]))

        idx=np.random.randint(len(chiSet), size=100)
        T=chiSet[idx,:]
        for t in T:
            TrainingData.append((t,[0,1,0,0,0,0]))

        idx=np.random.randint(len(pengSet), size=100)
        T=pengSet[idx,:]
        for t in T:
            TrainingData.append((t,[0,0,1,0,0,0]))

        idx=np.random.randint(len(gangSet), size=70)
        T=gangSet[idx,:]
        for t in T:
            TrainingData.append((t,[0,0,0,1,0,0]))

        idx=np.random.randint(len(bugangSet), size=90)
        T=bugangSet[idx,:]
        for t in T:
            TrainingData.append((t,[0,0,0,0,1,0]))

        idx=np.random.randint(len(angangSet), size=50)
        T=angangSet[idx,:]
        for t in T:
            TrainingData.append((t,[0,0,0,0,0,1]))

        trainloader = torch.utils.data.DataLoader(TrainingData, batch_size=batch_size,
                                     shuffle=True)

        for i, data in enumerate(trainloader, 0):
            input,label=data
            optimizer.zero_grad()
            output=Model(input.float().to(device))
            # print(output, torch.stack(label, dim=1))

            loss=criterion(output, torch.stack(label, dim=1).float().to(device))
            loss.backward()
            optimizer.step()
            
            runningLoss+=loss.item()

        print('{} loss: {}'.format(i, runningLoss))
        runningLoss=0 
        losses.append(runningLoss)

        TestingData=[]
        idx=np.random.randint(len(PassSet), size=500)
        T=PassSet[idx,:]
        for t in T:
            TestingData.append((t,0))

        idx=np.random.randint(len(chiSet), size=500)
        T=chiSet[idx,:]
        for t in T:
            TestingData.append((t,1))

        idx=np.random.randint(len(pengSet), size=500)
        T=pengSet[idx,:]
        for t in T:
            TestingData.append((t,2))

        idx=np.random.randint(len(gangSet), size=500)
        T=gangSet[idx,:]
        for t in T:
            TestingData.append((t,3))

        idx=np.random.randint(len(bugangSet), size=500)
        T=bugangSet[idx,:]
        for t in T:
            TestingData.append((t,4))

        idx=np.random.randint(len(angangSet), size=500)
        T=angangSet[idx,:]
        for t in T:
            TestingData.append((t,5))
        correct=np.zeros(6)
        sum=np.zeros(6)
        Model.eval()
        for data, label in TestingData:
            output=Model(torch.tensor(data).float().to(device))
            if torch.argmax(output).item()==label:
                correct[label]+=1
            sum[label]+=1
        for ty in range(6):
            print('Accuracy of {}: {}'.format(ty, correct[ty]/sum[ty]))
        print('Accuracy: {}'.format(np.sum(correct)/np.sum(sum)))
        totAccuracy.append(np.sum(correct)/np.sum(sum))
    #torch.save(Model.state_dict(), './model/claiming.pth')





