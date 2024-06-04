import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
class ClaimingModel(nn.Module):
    def __init__(self):
        super(ClaimingModel, self).__init__()
        self.inputSize = 34*13
        self.outputSize = 2
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
        return self.network(input)

class DiscardModel(nn.Module):
    def __init__(self):
        super(DiscardModel, self).__init__()
        self.inputSize = 34*13
        self.outputSize = 34
        self.hiddenSize1 = 512
        self.hiddenSize2 = 256
        self.hiddenSize3 = 256
        
        self.network=nn.Sequential(
            nn.Linear(self.inputSize, self.hiddenSize1),
            #nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.hiddenSize1, self.hiddenSize2),
            #nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.hiddenSize2, self.hiddenSize3),
            #nn.Dropout(0.3),
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
        return self.network(input)
    
if __name__=='__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else :
        device=torch.device('cpu')

    Model=ClaimingModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer=optim.Adam(Model.parameters(), lr=0.001)
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

    #print(chiSet[789][0:34], chiSet[789][34*8:34*9])
    
    """
    claiming: pass chi peng gang bugang angang 
    label:    0    1   2    3    4      5
    """    
    
    # print('Finishing preparing training dataset!!')
    

    
    
    
    #testloader = torch.utils.data.DataLoader(TestingData, batch_size=batch_size,
    #                                shuffle=False,)
    
    claiming=['Chi', 'Peng', 'Gang', 'Bugang', 'Angang']
    for claim in claiming:
        print(claim)
        runningLoss=0    
        acu=[]
        losses=[]
        for epoch in range(1000):
            Model.train()
            TrainingData=[]
            one=[0,1]
            zero=[1,0]
            idx=np.random.randint(len(PassSet), size=50)
            T=PassSet[idx,:]
            for t in T:
                TrainingData.append((t,zero))
            
            idx=np.random.randint(len(chiSet), size=50)
            T=chiSet[idx,:]
            for t in T:
                TrainingData.append((t,one if claim=='Chi' else zero))

            idx=np.random.randint(len(pengSet), size=50)
            T=pengSet[idx,:]
            for t in T:
                TrainingData.append((t,one if claim=='Peng' else zero))

            idx=np.random.randint(len(gangSet), size=50)
            T=gangSet[idx,:]
            for t in T:
                TrainingData.append((t,one if claim=='Gang' else zero))

            idx=np.random.randint(len(bugangSet), size=50)
            T=bugangSet[idx,:]
            for t in T:
                TrainingData.append((t,one if claim=='Bugang' else zero))

            idx=np.random.randint(len(angangSet), size=50)
            T=angangSet[idx,:]
            for t in T:
                TrainingData.append((t,one if claim=='Angang' else zero))

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
            losses.append(runningLoss)
            runningLoss=0
            TestingData=[]
            idx=np.random.randint(len(PassSet), size=100)
            T=PassSet[idx,:]
            for t in T:
                TestingData.append((t,0))

            idx=np.random.randint(len(chiSet), size=100)
            T=chiSet[idx,:]
            for t in T:
                TestingData.append((t,1 if claim=='Chi' else 0))

            idx=np.random.randint(len(pengSet), size=100)
            T=pengSet[idx,:]
            for t in T:
                TestingData.append((t,1 if claim=='Peng' else 0))

            idx=np.random.randint(len(gangSet), size=100)
            T=gangSet[idx,:]
            for t in T:
                TestingData.append((t,1 if claim=='Gang' else 0))

            idx=np.random.randint(len(bugangSet), size=100)
            T=bugangSet[idx,:]
            for t in T:
                TestingData.append((t,1 if claim=='Bugang' else 0))

            idx=np.random.randint(len(angangSet), size=100)
            T=angangSet[idx,:]
            for t in T:
                TestingData.append((t,1 if claim=='Angang' else 0))
            correct=0
            sum=0
            Model.eval()
            for data, label in TestingData:
                output=Model(torch.tensor(data).float().to(device))
                if torch.argmax(output).item()==label:
                    correct+=1
                sum+=1
            print('Accuracy: {}'.format(correct/sum))
            acu.append(correct/sum)
        np.save('results/{}_losses'.format(claim),np.array(losses))
        np.save('results/{}_acu'.format(claim),np.array(acu))
    
