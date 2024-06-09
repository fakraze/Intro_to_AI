import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from argparse import ArgumentParser
from mahjong_env.nn import *

def trainNN1Discard(args, device):
    model=NN1Discard().to(device)
    criterion = nn.CrossEntropyLoss()  # Corrected loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    batch_size = args.batchSize
    playSet=np.load('./dataset/1/play.npy')
    labelSet=np.load('./dataset/1/labelPlay.npy')
    #runningLoss=0
    #Accuracy=0
    #acu=[]
    #ewma=[]
    #losses=[]
    lst=[i for i in range(34)]
    for epoch in range(args.epoch):
        model.train()
        TrainingData=[]
        idx=np.random.randint(len(playSet), size=70000)
        T=playSet[idx,:]
        L=labelSet[idx,:]
        for t,l in zip(T,L):
            TrainingData.append((t,l))

        trainloader = DataLoader(TrainingData, batch_size=batch_size,
                                     shuffle=True)

        for i, data in enumerate(trainloader, 0):
            input,label=data
            optimizer.zero_grad()
            output=model(input.float().to(device))
            loss=criterion(output, label.float().to(device))
            loss.backward()
            optimizer.step()
        #losses.append(loss)
        if args.eval and (epoch+1)%args.evalFreq==0:
            TestingData=[]
            idx=np.random.randint(len(playSet), size=1000)
            T=playSet[idx,:]
            L=labelSet[idx,:]
            for t,l in zip(T,L):
                TestingData.append((t,l))
            correct=0
            sum=0
            model.eval()
            for data, label in TestingData:
                output=model(torch.tensor(data).float().to(device))
                if label[torch.argmax(output).item()]==1:
                    correct+=1
                sum+=1
            #acu.append(correct/sum)
            print('Accuracy {}: {}'.format(epoch+1, correct/sum))
            # print('{}\nModel Discard: {}\nLabel Discard: {}\n'.format( data[0:34], torch.argmax(output).item(), np.argmax(label)))
            #Accuracy=0.95*Accuracy+0.05*correct/sum
            #ewma.append(Accuracy)
    #np.save('results/{}_losses'.format('Discard'),np.array(losses))
    #np.save('results/{}_acu'.format('Discard'),np.array(acu))
    #np.save('results/{}_ewma'.format('Discard'),np.array(ewma))
    if not os.path.exists('./model/0'):
        os.makedirs('./model/0')
    
    torch.save(model.state_dict(), './model/0/discard.pth')

def trainNN23Discard(args, device):
    model=NN23Discard().to(device)
    criterion = nn.CrossEntropyLoss()  # Corrected loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    batch_size = args.batchSize
    playSet=np.load('./dataset/2/discard.npy')
    labelSet=np.load('./dataset/2/labelDiscard.npy')
    #runningLoss=0
    #Accuracy=0
    #acu=[]
    #ewma=[]
    #losses=[]
    lst=[i for i in range(34)]
    for epoch in range(args.epoch):
        model.train()
        TrainingData=[]
        idx=np.random.randint(len(playSet), size=70000)
        T=playSet[idx,:]
        L=labelSet[idx,:]
        for t,l in zip(T,L):
            TrainingData.append((t,l))

        trainloader = DataLoader(TrainingData, batch_size=batch_size,
                                     shuffle=True)

        for i, data in enumerate(trainloader, 0):
            input,label=data
            optimizer.zero_grad()
            output=model(input.float().to(device))
            loss=criterion(output, label.float().to(device))
            loss.backward()
            optimizer.step()
        #losses.append(loss)
        if args.eval and (epoch+1)%args.evalFreq==0:
            TestingData=[]
            idx=np.random.randint(len(playSet), size=1000)
            T=playSet[idx,:]
            L=labelSet[idx,:]
            for t,l in zip(T,L):
                TestingData.append((t,l))
            correct=0
            sum=0
            model.eval()
            for data, label in TestingData:
                output=model(torch.tensor(data).float().to(device))
                if label[torch.argmax(output).item()]==1:
                    correct+=1
                sum+=1
            #acu.append(correct/sum)
            print('Accuracy {}: {}'.format(epoch+1, correct/sum))
            # print('{}\nModel Discard: {}\nLabel Discard: {}\n'.format( data[0:34], torch.argmax(output).item(), np.argmax(label)))
            #Accuracy=0.95*Accuracy+0.05*correct/sum
            #ewma.append(Accuracy)
    #np.save('results/{}_losses'.format('Discard'),np.array(losses))
    #np.save('results/{}_acu'.format('Discard'),np.array(acu))
    #np.save('results/{}_ewma'.format('Discard'),np.array(ewma))
    if not os.path.exists('./model/1'):
        os.makedirs('./model/1')
    
    torch.save(model.state_dict(), './model/1/discard.pth')

def trainNN1Claiming(args, device):
    Model = NN1Claiming().to(device)
    criterion = nn.CrossEntropyLoss()  # Corrected loss function
    optimizer = optim.Adam(Model.parameters(), lr=args.lr)
    batch_size = args.batchSize

    pengSet = np.load('../dataset/peng.npy')
    chiSet = np.load('../dataset/chi.npy')
    gangSet = np.load('../dataset/gang.npy')
    bugangSet = np.load('../dataset/bugang.npy')
    angangSet = np.load('../dataset/angang.npy')
    PassSet = np.load('../dataset/Pass.npy')
    
    #runningLoss = 0
    
    Model.train()
    claims=['chi', 'peng', 'gang', 'bugang', 'angang']
    
    for claim in claims:
        #print(claim)
        #acu=[]
        #losses=[]
        for epoch in range(args.epoch):
            TrainingData = []
            TrainingLabels = []

            for dataset, label in [(PassSet, 0), (chiSet, 1 if claim=='chi' else 0), (pengSet, 1 if claim=='peng' else 0), (gangSet, 1 if claim=='gang' else 0), (bugangSet, 1 if claim=='bugang' else 0), (angangSet, 1 if claim=='angang' else 0)]:
                idx = np.random.randint(len(dataset), size=500)
                T = dataset[idx, :]
                for t in T:
                    # print(t)
                    TrainingData.append(t)
                    TrainingLabels.append(label)

            trainloader = DataLoader(TensorDataset(torch.tensor(TrainingData, dtype=torch.float32), torch.tensor(TrainingLabels, dtype=torch.long)), batch_size=batch_size, shuffle=True)

            for i, (input, label) in enumerate(trainloader, 0):
                optimizer.zero_grad()
                output = Model(input.to(device))
                
                loss = criterion(output, label.to(device))
                loss.backward()
                optimizer.step()

                runningLoss += loss.item()
                
            #losses.append(runningLoss)
            runningLoss=0
            if args.eval and (epoch+1)%args.evalFreq==0:
                TestingData = []
                TestingLabels = []

                for dataset, label in [(PassSet, 0), (chiSet, 1 if claim=='chi' else 0), (pengSet, 1 if claim=='peng' else 0), (gangSet, 1 if claim=='gang' else 0), (bugangSet, 1 if claim=='bugang' else 0), (angangSet, 1 if claim=='angang' else 0)]:
                    if label == 1:
                        idx = np.random.randint(len(dataset), size=500)            
                    else:   
                        idx = np.random.randint(len(dataset), size=100)
                    T = dataset[idx, :]
                    for t in T:
                        TestingData.append(t)
                        TestingLabels.append(label)

                testloader = DataLoader(TensorDataset(torch.tensor(TestingData, dtype=torch.float32), torch.tensor(TestingLabels, dtype=torch.long)), batch_size=batch_size, shuffle=False)

                correct = 0
                total = 0
                Model.eval()
                with torch.no_grad():
                    for data, label in testloader:
                        output = Model(data.to(device))
                        predicted = torch.argmax(output, dim=1)  # Corrected dimension for argmax
                        correct += (predicted == label.to(device)).sum().item()
                        # print("predicted")
                        # print(predicted)
                        # print("label")
                        # print(label)
                        total += label.size(0)

                print(f'Accuracy: {correct / total:.4f}')
                #acu.append(correct / total)
        #np.save('./results/{}Acu'.format(claim), np.array(acu))
        #np.save('./results/{}Loss'.format(claim), np.array(losses))
        if not os.path.exists('./model/2'):
            os.makedirs('./model/2')
        
        torch.save(Model.state_dict(), f'./model/2/{claim}.pth')

def trainNN2Claiming(args, device):
    Model=NN2Claiming().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer=optim.Adam(Model.parameters(), lr=args.lr)
    batch_size=args.batchSize

    pengSet=np.load('../dataset/peng.npy')
    chiSet=np.load('../dataset/chi.npy')
    gangSet=np.load('../dataset/gang.npy')
    bugangSet=np.load('../dataset/bugang.npy')
    angangSet=np.load('../dataset/angang.npy')
    PassSet=np.load('../dataset/Pass.npy')
    
    """
    claiming: pass chi peng gang bugang angang 
    label:    0    1   2    3    4      5
    """    
    
    claiming=['Chi', 'Peng', 'Gang', 'Bugang', 'Angang']
    for claim in claiming:
        #print(claim)
        #runningLoss=0    
        #acu=[]
        #losses=[]
        for epoch in range(args.epoch):
            Model.train()
            TrainingData=[]
            one=[0,1]
            zero=[1,0]
            idx=np.random.randint(len(PassSet), size=500)
            T=PassSet[idx,:]
            for t in T:
                TrainingData.append((t,zero))
            
            idx=np.random.randint(len(chiSet), size=500)
            T=chiSet[idx,:]
            for t in T:
                TrainingData.append((t,one if claim=='Chi' else zero))

            idx=np.random.randint(len(pengSet), size=500)
            T=pengSet[idx,:]
            for t in T:
                TrainingData.append((t,one if claim=='Peng' else zero))

            idx=np.random.randint(len(gangSet), size=500)
            T=gangSet[idx,:]
            for t in T:
                TrainingData.append((t,one if claim=='Gang' else zero))

            idx=np.random.randint(len(bugangSet), size=500)
            T=bugangSet[idx,:]
            for t in T:
                TrainingData.append((t,one if claim=='Bugang' else zero))

            idx=np.random.randint(len(angangSet), size=500)
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
            #losses.append(runningLoss)
            runningLoss=0
            
            if args.eval and (epoch+1)%args.evalFreq==0:
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
                #acu.append(correct/sum)
        #np.save('results/{}_losses'.format(claim),np.array(losses))
        #np.save('results/{}_acu'.format(claim),np.array(acu))
        if not os.path.exists('./model/3'):
            os.makedirs('./model/3')
        
        torch.save(Model.state_dict(), f'./model/3/{claim}.pth')

def trainNN3Claiming(args, device):
    Model=NN3Claiming().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer=optim.Adam(Model.parameters(), lr=args.lr)
    batch_size=args.batchSize

    pengSet=np.load('../dataset/peng.npy')
    chiSet=np.load('../dataset/chi.npy')
    gangSet=np.load('../dataset/gang.npy')
    bugangSet=np.load('../dataset/bugang.npy')
    angangSet=np.load('../dataset/angang.npy')
    PassSet=np.load('../dataset/Pass.npy')
    #print('Finishing loading Data!!')
    """
    claiming: pass chi peng gang bugang angang 
    label:    0    1   2    3    4      5
    """    
    
    #runningLoss=0
    #losses=[]
    #pasAcc=[]
    #chiAcc=[]
    #pengAcc=[]
    #bugangAcc=[]
    #gangAcc=[]
    #angangAcc=[]
    
    #totAccuracy=[]
    Model.train()
    for epoch in range(args.epoch):
        TrainingData=[]
        idx=np.random.randint(len(PassSet), size=500)
        T=PassSet[idx,:]
        for t in T:
            TrainingData.append((t,[1,0,0,0,0,0]))

        idx=np.random.randint(len(chiSet), size=500)
        T=chiSet[idx,:]
        for t in T:
            TrainingData.append((t,[0,1,0,0,0,0]))

        idx=np.random.randint(len(pengSet), size=500)
        T=pengSet[idx,:]
        for t in T:
            TrainingData.append((t,[0,0,1,0,0,0]))

        idx=np.random.randint(len(gangSet), size=350)
        T=gangSet[idx,:]
        for t in T:
            TrainingData.append((t,[0,0,0,1,0,0]))

        idx=np.random.randint(len(bugangSet), size=450)
        T=bugangSet[idx,:]
        for t in T:
            TrainingData.append((t,[0,0,0,0,1,0]))

        idx=np.random.randint(len(angangSet), size=250)
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

        #print('{} loss: {}'.format(i, runningLoss))
        runningLoss=0 
        #losses.append(runningLoss)
        if args.eval and (epoch+1)%args.evalFreq==0:
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
            #totAccuracy.append(np.sum(correct)/np.sum(sum))
    if not os.path.exists('./model/4'):
        os.makedirs('./model/4')
    
    torch.save(Model.state_dict(), './model/4/claiming.pth')




if __name__=='__main__':
    parser=ArgumentParser()
    parser.add_argument('--model', default=0, type=int) # 0,1 Discard{1,23} 2,3,4 Claiming{1,2,3} 
    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batchSize', default=50, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--evalFreq', default=1, type=int)
    args = parser.parse_args()
    print(args)
    if not os.path.exists('./model'):
        os.makedirs('./model')
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model==0:
        trainNN1Discard(args, device)
    elif args.model==1:
        trainNN23Discard(args, device)
    elif args.model==2:
        trainNN1Claiming(args, device)
    elif args.model==3:
        trainNN2Claiming(args, device)
    elif args.model==4:
        trainNN3Claiming(args, device)
    



    