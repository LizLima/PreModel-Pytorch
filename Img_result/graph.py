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
    path = '/media/liz/Files/Model-Pretrained/GAN_64batch_'

    list_file = [
        ("Test1", "Test1/files_gan_train.pckl"),
        # ("Test2", "Test2/files_gan_train.pckl"),
        ("Test3", "Test3/files_gan_train_499.pckl"),
        ("Test4", "Test4/files_gan_train_499.pckl"),
        ("Test5", "Test5/files_gan_train_499.pckl"),
        ("Test5_v1", "Test5_v1/files_gan_train_224.pckl"),
        ("Test6", "Test6/files_gan_train_499.pckl"),
        ("TestX_v1", "TestX_v1/files_gan_train_149.pckl"),
        ("Test7", "Test7/files_gan_train_499.pckl"),
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
    path = '/media/liz/Files/Model-Pretrained/GAN_64batch_'

    list_file = [
        ("Test1", "Test1/files_gan_train.pckl"),
        # ("Test2", "Test2/files_gan_train.pckl"),
        ("Test3", "Test3/files_gan_train_499.pckl"),
        ("Test4", "Test4/files_gan_train_499.pckl"),
        ("Test5", "Test5/files_gan_train_499.pckl"),
        # ("Test5_v1", "Test5_v1/files_gan_train_224.pckl"),
        # ("Test6", "Test6/files_gan_train_499.pckl"),
        # ("TestX_v1", "TestX_v1/files_gan_train_149.pckl"),
        # ("Test7", "Test7/files_gan_train_499.pckl"),
    ]
    # Plot information
    for label, path_l in list_file:
        with open(path + path_l, 'rb') as f:
            l = pickle.load(f) # List o values
            listG, listD = zip(*l)

            # list_G = [ round(float(x), 2) for x in listG[1:] ]
            # x = np.arange(len(list_G))
            # plt.plot(x, list_G, label=label + "G")

            list_D = [ round(float(x), 2) for x in listD[1:] ]
            x = np.arange(len(list_D))
            plt.plot(x, list_D, label=label + 'D')
    
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



