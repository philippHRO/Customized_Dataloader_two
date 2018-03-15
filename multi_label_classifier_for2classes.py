import sys
import os
import random
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import dataset_processing_for2classes
import torch.optim as optim
from torch.autograd import Variable
import torch

class MultiLabelNN(nn.Module):
    def __init__(self, nlabel):
        super(MultiLabelNN, self).__init__()
        self.nlabel = nlabel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(179776,1024)
        self.fc2 = nn.Linear(1024, nlabel)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        # x = self.pool(x)
        x = x.view(-1, 179776)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def main():
    """Main Func"""

    DATA_PATH = 'data_new/'
    TRAIN_DATA = 'train_img/'
    TEST_DATA = 'test_img/'

    # These are the actual strings the processing function should
    # look for in the filenames, in Ordner to create the target labels.
    search_classes = ["cat", "dog"]

    NLABELS = len(search_classes)
    WORKERS = 4
    my_batch_size = 3
    epochs = 1
    do_test_total = True

    # Info/Reminder:
    # Loss Functions need different tensors to work. The tensor is created automatically 
    # depending on the numpy array, that is created inside the class "DatasetProcessing".
    # Choose accordingly:
        # nn.MSELoss() needs FloatTensor as target, hence a np.float32 array is needed.
        # nn.MultiLabelMarginLoss() needs LongTensor as target, hence a np.in64 array is needed.

    #kwargs = {}                                        # für CPU
    #kwargs = {'num_workers': 1, 'pin_memory': True}     # für GPU/CUDA

    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

    dset_train = dataset_processing_for2classes.DatasetProcessing(
        DATA_PATH,
        TRAIN_DATA,
        #TRAIN_IMG_FILE,
        #TRAIN_LABEL_FILE,
        search_classes,
        transformations
        )

    dset_test = dataset_processing_for2classes.DatasetProcessing(
        DATA_PATH,
        TEST_DATA,
        #TEST_IMG_FILE,
        #TEST_LABEL_FILE,
        search_classes,
        transformations
        )

    train_loader = DataLoader(dset_train,
                            batch_size=my_batch_size,
                            shuffle=True,
                            num_workers=WORKERS
                            )

    test_loader = DataLoader(dset_test,
                            batch_size=my_batch_size,
                            shuffle=False,
                            num_workers=WORKERS
                            )

    use_gpu = torch.cuda.is_available()
    print("CUDA?: ", use_gpu)
    model = MultiLabelNN(NLABELS)
    if use_gpu:
        model = model.cuda()
    #print(model)

    optimizer = optim.SGD(model.parameters(), lr=0.0001)
    #criterion = nn.MultiLabelMarginLoss() # target tensor train_labels must be LongTensor.
    criterion = nn.MSELoss() # target tensor train_labels must be FloatTensor.
    #criterion = F.multilabel_margin_loss

    for epoch in range(1, epochs + 1):
        ### training phase
        total_training_loss = 0.0
        total = 0.0
        for it, traindata in enumerate(train_loader, 0):
            train_inputs, train_labels = traindata
            if use_gpu:
                train_inputs, train_labels = Variable(train_inputs.cuda()), Variable(train_labels.cuda())
            else: train_inputs, train_labels = Variable(train_inputs), Variable(train_labels)
            optimizer.zero_grad()
            train_outputs = model(train_inputs)
            #print("Train Output is:", train_outputs)
            #print("Train Labels is:", train_labels) #.type()
            loss = criterion(train_outputs, train_labels)
            loss.backward()
            optimizer.step()
            total += train_labels.size(0)
            total_training_loss += loss.data[0]
            #print(total_training_loss)
            # print('Training Phase: Epoch: [%2d|%2d]. Iteration: [%2d]. Iteration Loss: [%f]' %
            #     (epoch, epochs, it, loss.data[0]))
        print('Training Phase: Epoch: [%2d|%2d]. Loss: [%f]' %
            (epoch, epochs, total_training_loss)) # for last interation loss use: loss.data[0]

        ### testing phase
        total_testing_loss = 0.0
        total_t = 0.0
        for ite, testdata in enumerate(test_loader, 0):
            test_inputs, test_labels = testdata
            if use_gpu:
                test_inputs, test_labels = Variable(test_inputs.cuda()), Variable(test_labels.cuda())
            else: test_inputs, test_labels = Variable(test_inputs), Variable(test_labels)
            test_outputs = model(test_inputs)
            test_loss = criterion(test_outputs, test_labels)
            total_t += test_labels.size(0)
            total_testing_loss += test_loss.data[0]
            #print(total_testing_loss, test_loss.data[0])
            # print('Testing Phase:  Epoch: [%2d|%2d]. Iteration: [%2d]. Iteration Loss: [%f]' %
            #     (epoch, epochs, ite, test_loss.data[0]))
        print('Testing Phase:  Epoch: [%2d|%2d]. Loss: [%f]' %
            (epoch, epochs, total_testing_loss)) # for last interation loss use test_loss.data[0]

    if do_test_total:
        correct = 0
        test_total = 0
        for itera, testdata2 in enumerate(test_loader, 0):
        # for data in testloader:
            test_images2, test_labels2 = testdata2
            if use_gpu:
                test_images2 = Variable(test_images2.cuda())
                #test_labels2 = Variable(test_labels2.cuda())
            else:
                test_images2 = Variable(test_images2)
            outputs = model(test_images2)
            _, predicted = torch.max(outputs.data, 1)
            #predicted = outputs.data.max(1, keepdim=True)[1].cpu().numpy()[0][0]
            # out.data.max(1, keepdim=True)[1].cpu().numpy()[0][0] 
            #print("_: ", _)
            print("predicted: ", predicted)
            print("test_labels2: ", test_labels2)
            
            test_total += test_labels2.size(0)
            
            test_labels2 = test_labels2.type_as(predicted)
            #test_labels2.type_as(predicted) # convert to long tensor, doesn't work
            #test_labels2.type(predicted.type())
            print("test_labels2 after conv: ", test_labels2)
            print("test_labels2[0]: ", test_labels2[0])
            correct += (predicted == test_labels2[0]).sum()

        print('Accuracy of the network on all the test images: %d %%' % (
            100 * correct / test_total))




if __name__ == "__main__":
    main()
