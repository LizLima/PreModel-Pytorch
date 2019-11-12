import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import torchvision.utils as vutils
import pickle

# Model
import model as modelgen
from tqdm.autonotebook import tqdm
################################################
# CONFIG 
################################################
device = torch.device('cuda')

laten_sapce = 256
lr          = 0.001
num_epochs  = 500
batch_size  = 1
image_size  = 128
print_epoch = 25

################################################
# DATASET 
################################################
path = '/home/liz/Documents/Data/VGGFace'
path_result= '/home/liz/Documents/Model-Pretrained/results'
# Create the dataset
train_dataset = datasets.ImageFolder(root=path + '/vggface2_train/train',
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                           ]))

test_dataset = datasets.ImageFolder(root=path + '/vggface2_test/test',
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                           ]))                         
# Create the dataloader
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                          shuffle=True)

# train_size = int(0.7 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Plot some training images
real_batch = next(iter(trainloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()


################################################
# MODEL 
################################################
class_train = train_dataset.classes
class_test  = test_dataset.classes
num_classes = len(class_train) + len(class_test)
model       = modelgen.Generator(num_classes).to(device)
model.apply(modelgen.weights_init)
optimizer   = optim.Adam(model.parameters(), lr=lr)

lossMse     = nn.MSELoss()
lossCE      = nn.CrossEntropyLoss()


################################################
# TRAIN 
################################################
#Condificaton of label
def one_hot(label):
    # class is number
    y = torch.zeros(batch_size, num_classes, dtype=torch.long)
    y[range(y.shape[0]), label]=1
    return y

def train(epoch):
    model.train()
    train_loss = 0
    progress = tqdm(enumerate(trainloader), desc="Train", total=len(trainloader))
    for x in progress:
        
        data    = x[1]
        image   = data[0].to(device)
        # label   = one_hot(data[1]).to(device)
        label       = data[1].to(device)
        # update the gradients to zero
        optimizer.zero_grad()
        features_gen, image_synt = model(image)
        loss_mse    = lossMse(image_synt, image)
        loss_ce     = lossCE(features_gen, label)
        # update the weights
        loss_total = loss_ce + loss_mse
        loss_total.backward()
        optimizer.step()

        train_loss += loss_total.item()

        progress.set_description("Train epoch: %d, MSE: %.5f , CE: %.5f , T: %.5f " % (epoch, loss_mse, loss_ce, train_loss))
    
    # Save model
    if (epoch + 1) % print_epoch == 0:

        # Save the model
        # https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/2
        state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
        torch.save(state, path_result + "/vgg_checkpoint" + str(epoch) +"_" + ".pth.tar" ) 

    return train_loss
################################################
# TEST 
################################################
def test(epoch):
    model.eval()
    test_loss   = 0
    image       = None
    image_synt  = None
    progress = tqdm(enumerate(testloader), desc="Train", total=len(testloader))
    with torch.no_grad():
        for x in progress:
            data    = x[1]
            image   = data[0].to(device)
            # label   = one_hot(data[1]).to(device)
            label       = data[1].to(device)

            features_gen, image_synt = model(image)
            loss_mse    = lossMse(image_synt, image)
            loss_ce     = lossCE(features_gen, label)
            # update the weights
            loss_total = loss_ce + loss_mse
            
            test_loss += loss_total.item()

            progress.set_description("Test epoch: %d, MSE: %.5f , CE: %.5f , T: %.5f " % (epoch, loss_mse, loss_ce, train_loss))
        
        vutils.save_image(image_synt.data, path_result + '/vgg_synt_%03d.jpg' % epoch, normalize=True)
        vutils.save_image(image.data, path_result + '/vgg_input_%03d.jpg' % epoch, normalize=True)
    return test_loss

best_test_loss = float('inf')
patience_counter = 0

loss_Train = []
loss_Test = []
for e in range(num_epochs):

    train_loss = train(e)
    test_loss = test(e)

    train_loss /= len(train_dataset)
    test_loss /= len(test_dataset)
    loss_Train.append(train_loss)
    loss_Test.append(test_loss)

#plot 
fig = plt.figure()
x = np.arange(num_epochs + 1)
ax = plt.subplot(111)
ax.plot(x, loss_Train, 'mediumvioletred', label='Generator Training')
ax.plot(x, loss_Test, 'pink', label='Generator Test')

plt.title('Function loss')
ax.legend()
fig.savefig(path_result + '/vgg_plot' + str(num_epochs+1) + '.png')

# Save loss an test values to plot comparison
fichero = open(path_result + '/files_vgg19_train.pckl', 'wb')
pickle.dump(loss_Train, fichero)
fichero.close()
fichero = open(path_result + '/files_vgg19_test.pckl', 'wb')
pickle.dump(loss_Test, fichero)
fichero.close()
