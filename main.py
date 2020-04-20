from GrayRumatrix import getGrayRumatrix 
from PIL import Image
import numpy as np
from itertools import groupby
import csv
import warnings
warnings.filterwarnings("ignore")
test = getGrayRumatrix()

test.read_img("test.jpg")#Read in the picture file test.jpg, if you need to open another file, put the picture in this folder, and replace test.jpg with the new file name

DEG = [['deg0'], ['deg45'], ['deg90'], ['deg135']]
f = open('data.csv','w',encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow(["function","deg","value"])
for deg in DEG:
    now_deg = deg[0]
    test_data = test.getGrayLevelRumatrix(test.data,deg)
    #1
    SRE = test.getShortRunEmphasis(test_data) 
    SRE = np.squeeze(SRE)
    csv_writer.writerow(["SRE",now_deg,str(SRE)])
    #2
    LRE = test.getLongRunEmphasis(test_data)
    LRE = np.squeeze(LRE)
    csv_writer.writerow(["LRE",now_deg,str(LRE)])
    #3
    GLN = test.getGrayLevelNonUniformity(test_data)
    GLN = np.squeeze(GLN)
    csv_writer.writerow(["GLN",now_deg,str(GLN)])
    #4
    RLN = test.getRunLengthNonUniformity(test_data)
    RLN = np.squeeze(RLN)
    csv_writer.writerow(["RLN",now_deg,str(RLN)])
    #5
    RP = test.getRunPercentage(test_data)
    RP = np.squeeze(RP)
    csv_writer.writerow(["RP",now_deg,str(RP)])
    #6
    LGLRE = test.getLowGrayLevelRunEmphasis(test_data)
    LGLRE = np.squeeze(LGLRE)
    csv_writer.writerow(["LGLRE",now_deg,str(LGLRE)])
    #7
    HGL = test.getHighGrayLevelRunEmphais(test_data)
    HGL = np.squeeze(HGL)
    csv_writer.writerow(["HGL",now_deg,str(SRE)])
    #8
    SRLGLE = test.getShortRunLowGrayLevelEmphasis(test_data)
    SRLGLE = np.squeeze(SRLGLE)
    csv_writer.writerow(['SRLGLE',now_deg,str(SRLGLE)])
    #9
    SRHGLE = test.getShortRunHighGrayLevelEmphasis(test_data)
    SRHGLE = np.squeeze(SRHGLE)
    csv_writer.writerow(['SRHGLE',now_deg,str(SRHGLE)])
    #10
    LRLGLE = test.getLongRunLow(test_data)
    LRLGLE = np.squeeze(LRLGLE)
    csv_writer.writerow(['LRLGLE',now_deg,str(LRLGLE)])
    #11
    LRHGLE = test.getLongRunHighGrayLevelEmphais(test_data)
    LRHGLE = np.squeeze(LRHGLE)
    csv_writer.writerow(['LRHGLE',now_deg,str(LRHGLE)])