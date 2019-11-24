# Define loss functions
import torch

class Loss():

    @staticmethod
    def PixelWiseLoss(x_array, y_array):
        # x -> input, y -> target 
        sum = 0
        for i in range(9): # W
            for j in range(9): # H
                sum += abs( (x_array[0][i][j] - y_array[0][i][j]))


        return sum
# Calculate Pixels wise loss one chanel

x = torch.rand(1, 10, 10) # input
x.fill_(0.7)
y = torch.ones(1, 10, 10) # target
c = torch.zeros(1, 10, 10) # other class

loss = torch.nn.L1Loss(reduction='mean')
print(x)
print(y)
print(c)
print("i - t: ", loss(x,y))
print("c - t: ", loss(c,y) )
print("t - t: ", loss(y,y) )
print("------------------------")
print("i - t: ", Loss.PixelWiseLoss(x,y))
print("c - t: ", Loss.PixelWiseLoss(c,y) )
print("t - t: ", Loss.PixelWiseLoss(y,y) )