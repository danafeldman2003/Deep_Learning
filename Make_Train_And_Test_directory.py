# -*- coding: utf-8 -*-
"""
Dana Feldman
YB4
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shutil
import os 



def Make_Directory():
    '''
    the function gets a directory name and directory path from the user and creates a directory according to that input,
    if the directory doesn't already exist.
    if the user presses enter when inputing directory path, default is desktop
    the function creates sub-direcories named "train" and "test" in the input directory
    '''
 
    # input directory name
    print("enter directory")
    directory = input()
  
    # input directory path 
    print("enter path")
    parent_dir =input()#directory path
    if parent_dir == "":#default is desktop
        parent_dir=r"C:\Users\danaf\OneDrive\Desktop"
        
    # final path of directory
    path = os.path.join(parent_dir,directory) 
    
    #create directory if doesn't exist
    isFile = os.path.isdir(path)  
    if isFile is False:
        os.mkdir(path) 
        print("made") 
    else:
        print("already there")

    #make train sub-directory
    directory = "train"
    parent_dir = path
    path1 = os.path.join(parent_dir,directory) 
    os.mkdir(path1)

    #make test sub directory
    directory = "test"
    parent_dir = path
    path2 = os.path.join(parent_dir,directory) 
    os.mkdir(path2)




def Move_Pictures(dir_path,pic_path):
    '''
    inputs:destination directory to move files to, directory to move files from
    
    the function moves files to directories:
        -if the destination directory is train, move 70% of files from pic_path to train
        -if the destination directory is test, move all of the files from pic_path to test
    '''
    files = os.listdir(pic_path)  #file list from pic_path
    if os.path.basename(dir_path) =="train": #move 70% of files
        i=(int)(len(files)*0.7)
        for file in files:
            if(i>0):
                shutil.move(os.path.join(pic_path, file), dir_path)
                i-=1
      
            
    elif os.path.basename(dir_path) =="test":#move all files
        for file in files:
            shutil.move(os.path.join(pic_path, file), dir_path)

    


def Print_plot_pics(path):
    '''
    input:path of directory
    the function prints all images in a directory and their names
    '''
    files = os.listdir(path) 
    for file in files:
        print(file)#image name
        plt.imshow(mpimg.imread(os.path.join(path,file)))
        plt.show()
        
        
        
def Main():
    Make_Directory()
    Move_Pictures(r"C:\Users\student\make_dir_and_2_subdic\train",r"C:\Users\move_pics")
    Print_plot_pics(r"C:\Users\student\make_dir_and_2_subdic\train")
    
    
    
    
    
    
if __name__ == '__main__':
    Main()
    
    
    
    
    

  
