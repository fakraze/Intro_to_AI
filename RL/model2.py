import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ClaimingModel(nn.Module):
    def __init__(self):
        super(ClaimingModel, self).__init__()
        self.inputSize = 102
        self.outputSize = 2
        self.hiddenSize1 = 128
        self.hiddenSize2 = 128
        self.hiddenSize3 = 64
        
        self.network=nn.Sequential(
            nn.Linear(self.inputSize, self.hiddenSize1),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.hiddenSize1, self.hiddenSize2),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.hiddenSize2, self.outputSize),
            nn.Softmax()
        )
        self.apply(self.init_weight)

    def init_weight(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            module.bias.data.fill_(0)
    def forward(self, input):
        return self.network(input)
    




if __name__=='__main__':
    """if torch.cuda.is_available():
        device = torch.device("cuda")
    else :"""
    device=torch.device('cpu')

    Model=ClaimingModel().to(device)
    criterion = nn.MSELoss()
    optimizer=optim.Adam(Model.parameters(), lr=0.01)
    batch_size=10

    pengSet=np.load('../dataset/peng.npy')
    chiSet=np.load('../dataset/chi.npy')
    gangSet=np.load('../dataset/gang.npy')
    bugangSet=np.load('../dataset/bugang.npy')
    angangSet=np.load('../dataset/angang.npy')
    PassSet=np.load('../dataset/Pass.npy')
    print('Finishing loading Data!!')
    # print( len(PassSet), len(chiSet), len(pengSet),  len(gangSet), len(bugangSet), len(angangSet), )
    # 4597819 232248 177092 7037 9655 5110 


    
    """
    claiming: pass chi peng gang bugang angang 
    label:    0    1   2    3    4      5
    """    
    
    # print('Finishing preparing training dataset!!')
    

    
    
    
    #testloader = torch.utils.data.DataLoader(TestingData, batch_size=batch_size,
    #                                shuffle=False,)
    
    runningLoss=0
    
    Model.train()
    for epoch in range(500):
        TrainingData=[]
        idx=np.random.randint(len(PassSet), size=500)
        T=PassSet[idx,:]
        for t in T:
            TrainingData.append((t,[1,0]))

        idx=np.random.randint(len(chiSet), size=500)
        T=chiSet[idx,:]
        for t in T:
            TrainingData.append((t,[1,0]))

        idx=np.random.randint(len(pengSet), size=500)
        T=pengSet[idx,:]
        for t in T:
            TrainingData.append((t,[1,0]))

        idx=np.random.randint(len(gangSet), size=500)
        T=gangSet[idx,:]
        for t in T:
            TrainingData.append((t,[1,0]))

        idx=np.random.randint(len(bugangSet), size=500)
        T=bugangSet[idx,:]
        for t in T:
            TrainingData.append((t,[1,0]))

        idx=np.random.randint(len(angangSet), size=500)
        T=angangSet[idx,:]
        for t in T:
            TrainingData.append((t,[0,1]))

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
            if i%100==99:
                print('{} loss: {}'.format(i, runningLoss))
                runningLoss=0 


    TestingData=[]
    idx=np.random.randint(len(PassSet), size=100)
    T=PassSet[idx,:]
    for t in T:
        TestingData.append((t,0))

    idx=np.random.randint(len(chiSet), size=100)
    T=chiSet[idx,:]
    for t in T:
        TestingData.append((t,0))

    idx=np.random.randint(len(pengSet), size=100)
    T=pengSet[idx,:]
    for t in T:
        TestingData.append((t,0))

    idx=np.random.randint(len(gangSet), size=100)
    T=gangSet[idx,:]
    for t in T:
        TestingData.append((t,0))

    idx=np.random.randint(len(bugangSet), size=100)
    T=bugangSet[idx,:]
    for t in T:
        TestingData.append((t,0))

    idx=np.random.randint(len(angangSet), size=100)
    T=angangSet[idx,:]
    for t in T:
        TestingData.append((t,1))
    correct=0
    sum=0
    Model.eval()
    for data, label in TestingData:
        output=Model(torch.tensor(data).float().to(device))
        if torch.argmax(output).item()==label:
            correct+=1
        sum+=1

    print('Accuracy: {}'.format(correct/sum))
    torch.save(Model.state_dict(), './model/angang.pth')

