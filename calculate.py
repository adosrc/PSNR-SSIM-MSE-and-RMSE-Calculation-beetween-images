import cv2 
from skimage.metrics import structural_similarity
from skimage.measure import compare_mse as mse
from skimage import img_as_float
import math
import numpy as np
from glob import glob
from natsort import natsorted
import os


highdir = []
testdir = []


highdir = os.listdir('sunuHR/')
highdir = sorted(highdir,key=lambda x: int(os.path.splitext(x)[0]))

testdir = os.listdir('sunuSR/')
testdir = sorted(testdir,key=lambda x: int(os.path.splitext(x)[0][3:])) 
"""Change splitext depends on the name of files"""


def rmse(img1, img2):
    """Calculates the root mean square error (RSME) between two images"""
    return math.sqrt(mse(img_as_float(img1), img_as_float(img2))) * 255

def psnr(img1,img2):
    """Calculates the peak signal-to-noise ratio (PSNR) between two images"""
    return cv2.PSNR(img1,img2)

def ssim(img1,img2):    
    """Calculates the structural similarity index measure (SSIM) between two images"""
    grayA = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    (score, diff) = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    return score
    
def mse(img1,img2):
    """Calculates the mean square error (MSE) between two images"""
    return np.square(np.subtract(img1,img2)).mean()


totpeak=0
totstruc=0
totroot=0
toterr=0

for idx in range(len(highdir)):
    img2 = cv2.imread('sunuHR/'+highdir[idx])
    img1 = cv2.imread('sunuSR/'+testdir[idx])
    
    if img1 is None or img2 is None:
        continue  
    
    root=rmse(img1,img2)
    peak=psnr(img1,img2)
    struc=ssim(img1,img2)
    err=mse(img1,img2)
  
    totpeak=totpeak+peak
    totstruc=totstruc+struc
    totroot=totroot+root
    toterr=toterr+err
  
    print("Psnr : {:.3f} ; Ssim : {:.3f} ; Mse : {:.3f} ; Rmse : {:.3f} ".format(peak,struc,err,root))
  
avgpeak=totpeak/len(highdir)
avgstruc=totstruc/len(highdir)
avgroot=totroot/len(highdir)
avgerr=toterr/len(highdir)
print("Total Image : {:d}  ".format(len(highdir)))

print("Average Values---->    Avg Psnr : {:.3f} ; Avg Ssim : {:.3f} ; Avg Mse : {:.3f} ; Avg Rmse : {:.3f} ".format(avgpeak,avgstruc,avgerr,avgroot))
