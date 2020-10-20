import cv2 as cv
import glob
# pip install opencv-python
# https://docs.opencv.org/master/df/dfb/group__imgproc__object.html#gga3a7850640f1fe1f58fe91a2d7583695dac6677e2af5e0fae82cc5339bfaef5038

# définition des ficheirs images
sampleList = glob.glob("samples-bank/*.png")

imgFile = 'spectro/mesange_charb.png'
img = cv.imread(imgFile,0)

best_corr_val = 0
best_corr_sample = ""

for sample in sampleList :
    # import du sample
    template = cv.imread(sample,0)

    #w, h = template.shape[::-1] # définition de la taille de l'image pour output

    meth = 'cv.TM_CCOEFF_NORMED'
    method = eval(meth)

    # Apply template Matching
    res = cv.matchTemplate(img,template,method)
    # Get Value from template matching
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    print(f"with sample {sample} the coef of corr is : {max_val} ")

    # If better correlation save values
    if max_val > best_corr_val :
        best_corr_val = max_val
        best_corr_sample = sample

# Test if corr coef is good inof
if best_corr_val < 0.65 :
    print("No correlation found")
else :
    print(f"Bird by correlation is : {best_corr_sample} with a Coef of Corr : {best_corr_val}")

