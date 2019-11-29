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


    @staticmethod
    def SymmetryLoss(image_synt):
        
        # Lsym=(1/W/2×H) ∑_x=1 ^W/2 ∑_y=1 ^ H |Ipred_(x,y) − Ipred_(W−(x−1)),y|
        # B, C, W, H = image_synt.shape  (batch, channel, W, H)
        # Use L1Loss  
        inv_idx = torch.arange(image_synt.size()[3]-1, -1, -1).long().to(self.device)
        img_inverse = image_synt.index_select(3, inv_idx)
        img_inverse.detach_()
        value_loss = self.l1_loss(image_synt, img_inverse)

        return value_loss

    @staticmethod
    def TotalVariationLoss(img):
        value_loss = torch.mean(torch.abs(img[:, :, : -1, :] - img[:, :, 1: , :])) + \
                     torch.mean(torch.abs(img[:, :, :, : -1] - img[:, :, :, 1 :]))
        return value_loss 

    @staticmethod
    def IdentityPreserving(input_a, input_b):

        diff = torch.abs(input_a - input_b)

        return torch.mean(diff)

    @staticmethod
    def Norm(value, p):
        b = torch.norm(value, p=2, dim=1)
        result = torch.mean(b)

        return result**2

    @staticmethod
    def FNorm(matrix):
        matrix = torch.abs(matrix)
        mult = matrix**2
        result = torch.mean(mult)

        return result**2

# x = torch.rand(1, 10, 10) # input
# x.fill_(0.7)
# y = torch.ones(1, 10, 10) # target
# c = torch.zeros(1, 10, 10) # other class

# loss = torch.nn.L1Loss(reduction='mean')
# print(x)
# print(y)
# print(c)
# print("i - t: ", loss(x,y))
# print("c - t: ", loss(c,y) )
# print("t - t: ", loss(y,y) )
# print("------------------------")
# print("i - t: ", Loss.PixelWiseLoss(x,y))
# print("c - t: ", Loss.PixelWiseLoss(c,y) )
# print("t - t: ", Loss.PixelWiseLoss(y,y) )