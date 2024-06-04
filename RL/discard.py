import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

    Model=DiscardModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer=optim.Adam(Model.parameters(), lr=0.0005)
    batch_size=50

    # len=4597819
    playSet=np.load('../dataset/discard.npy')
    labelSet=np.load('../dataset/labeldiscard.npy')
    
    """
    claiming: pass chi peng gang bugang angang 
    label:    0    1   2    3    4      5
    """    
    
    # print('Finishing preparing training dataset!!')
    
    #testloader = torch.utils.data.DataLoader(TestingData, batch_size=batch_size,
    #                                shu ffle=False,)
    
    runningLoss=0
    Model.train()
    Accuracy=0
    epoch=0
    acu=[]
    ewma=[]
    losses=[]
    lst=[i for i in range(34)]
    while Accuracy<0.7:
        #Model.train()
        TrainingData=[]
        idx=np.random.randint(len(playSet), size=70000)
        T=playSet[idx,:]
        L=labelSet[idx,:]
        for t,l in zip(T,L):
            TrainingData.append((t,l))


        trainloader = torch.utils.data.DataLoader(TrainingData, batch_size=batch_size,
                                     shuffle=True)

        for i, data in enumerate(trainloader, 0):
            input,label=data
            optimizer.zero_grad()
            output=Model(input.float().to(device))

            loss=criterion(output, label.float().to(device))
            loss.backward()
            optimizer.step()
        losses.append(loss)
            


        TestingData=[]

        idx=np.random.randint(len(playSet), size=1000)
        T=playSet[idx,:]
        L=labelSet[idx,:]
        for t,l in zip(T,L):
            TestingData.append((t,l))
        correct=0
        sum=0
        Model.eval()
        for data, label in TestingData:
            output=Model(torch.tensor(data).float().to(device))
            if label[torch.argmax(output).item()]==1:
                correct+=1
            sum+=1
            #print(data)
            #print(np.argmax(label), torch.argmax(output).item())
            #print(output[np.argmax(label)])
            #print()
        acu.append(correct/sum)
        
        print('Accuracy {}: {}'.format(epoch+1, correct/sum))
        print('{}\nModel Discard: {}\nLabel Discard: {}\n'.format( data[0:34], torch.argmax(output).item(), np.argmax(label)))
        Accuracy=0.95*Accuracy+0.05*correct/sum
        ewma.append(Accuracy)
        epoch+=1
    np.save('results/{}_losses'.format('Discard'),np.array(losses))
    np.save('results/{}_acu'.format('Discard'),np.array(acu))
    np.save('results/{}_ewma'.format('Discard'),np.array(ewma))
    
    torch.save(Model.state_dict(), './model/2/discard.pth')


