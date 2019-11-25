import argparse
import pickle
import math 
import numpy as np
import matplotlib.pyplot as plt

# Read values from console
parser = argparse.ArgumentParser(description='arg')
parser.add_argument("--function")
args = parser.parse_args()

def main():
    print("hello")

def compare():

    # Load pickle file
    path = '/media/liz/Files/Model-Pretrained/'

    list_file = [
        ("resnet-train-64", "/PreTrained_Resnet50_b64_lfw/files_resnet50_b64_train.pckl"),
        ("vgg19-train-64","/PreTrained_VGG19bn_b64_lfw/files_vgg19_train.pckl"),
        ("vgg19-test-64","/PreTrained_VGG19bn_b64_lfw/files_vgg19_test.pckl"),
        ("resnet-train-128", "/results_resnet50_batch128_lfw/files_resnet50_b128_train.pckl"),
        ("resnet-test-128", "/results_resnet50_batch128_lfw/files_resnet50_b128_test.pckl")
    ]
    
    # Plot information
    for label, path_l in list_file:
        with open(path + path_l, 'rb') as f:
            l = pickle.load(f) # List o values
            x = np.arange(len(l))
            plt.plot(x, l, label=label)
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE loss')
    plt.legend()
    plt.show()

def compare_gan():
      # Load pickle file
    path = '/media/liz/Files/Model-Pretrained/GAN_64batch/'
    
    list_file = [
        ("Test2", "Test4/files_gan_test.pckl")
        # ("vgg19-train-64","/PreTrained_VGG19bn_b64_lfw/files_vgg19_train.pckl"),
        # ("vgg19-test-64","/PreTrained_VGG19bn_b64_lfw/files_vgg19_test.pckl"),
        # ("resnet-train-128", "/results_resnet50_batch128_lfw/files_resnet50_b128_train.pckl"),
        # ("resnet-test-128", "/results_resnet50_batch128_lfw/files_resnet50_b128_test.pckl")
    ]
    
    # Plot information
    for label, path_l in list_file:
        with open(path + path_l, 'rb') as f:
            l = pickle.load(f) # List o values
            listG, listD = zip(*l)

            list_G = [ round(float(x), 2) for x in listG[1:] ]
            x = np.arange(len(list_G))
            plt.plot(x, list_G, label=listG[0])

            list_D = [ round(float(x), 2) for x in listD[1:] ]
            x = np.arange(len(list_D))
            plt.plot(x, list_D, label=listD[0])
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE loss')
    plt.legend()
    plt.show()

# INIT functions
if __name__ == '__main__':
    if args.function == 'compare':
        compare()
    if args.function == 'main':
        main()
    if args.function == 'compare_gan':
        compare_gan()



