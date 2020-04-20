import matplotlib.pyplot as plt 
from PIL import Image 
import numpy as np
from itertools import groupby
class getGrayRumatrix:
    data = 0 
    def read_img(self,path=" "):
        
        try:
            img = Image.open(path) 
            img = img.convert('L')
            self.data=np.array(img)
            
        except:
            img = None
            
    def getGrayLevelRumatrix(self, array, theta):
            '''
            
            array: the numpy array of the image
            theta: Input, the angle used when calculating the gray scale run matrix, list type, can contain fields:['deg0', 'deg45', 'deg90', 'deg135']
            glrlm: output,the glrlm result
            '''
            P = array
            x, y = P.shape
            min_pixels = np.min(P)   # the min pixel
            run_length = max(x, y)   # Maximum parade length in pixels
            num_level = np.max(P) - np.min(P) + 1   # Image gray level
    
            deg0 = [val.tolist() for sublist in np.vsplit(P, x) for val in sublist]   # 0deg
            deg90 = [val.tolist() for sublist in np.split(np.transpose(P), y) for val in sublist]   # 90deg
            diags = [P[::-1, :].diagonal(i) for i in range(-P.shape[0]+1, P.shape[1])]   #45deg
            deg45 = [n.tolist() for n in diags]
            Pt = np.rot90(P, 3)   # 135deg
            diags = [Pt[::-1, :].diagonal(i) for i in range(-Pt.shape[0]+1, Pt.shape[1])]
            deg135 = [n.tolist() for n in diags]
    
            def length(l):
                if hasattr(l, '__len__'):
                    return np.size(l)
                else:
                    i = 0
                    for _ in l:
                        i += 1
                    return i
    
            glrlm = np.zeros((num_level, run_length, len(theta)))   
            for angle in theta:
                for splitvec in range(0, len(eval(angle))):
                    flattened = eval(angle)[splitvec]
                    answer = []
                    for key, iter in groupby(flattened):  
                        answer.append((key, length(iter)))   
                    for ansIndex in range(0, len(answer)):
                        glrlm[int(answer[ansIndex][0]-min_pixels), int(answer[ansIndex][1]-1), theta.index(angle)] += 1   
            return glrlm
    '''
  The gray scale run matrix is only the measurement and statistics of the image pixel information. In the actual use process, the generated
     The gray scale run matrix is calculated to obtain image feature information based on the gray level co-occurrence matrix.
     First write a few common functions to complete the calculation of subscripts i and j (calcuteIJ ()), multiply and divide according to the specified dimension (apply_over_degree ())
     And calculate the sum of all pixels (calcuteS ())
    '''        
            
    def apply_over_degree(self,function, x1, x2):
        rows, cols, nums = x1.shape
        result = np.ndarray((rows, cols, nums))
        for i in range(nums):
                #print(x1[:, :, i])
                result[:, :, i] = function(x1[:, :, i], x2)
               # print(result[:, :, i])
                result[result == np.inf] = 0
                result[np.isnan(result)] = 0
        return result 
    def calcuteIJ (self,rlmatrix):
        gray_level, run_length, _ = rlmatrix.shape
        I, J = np.ogrid[0:gray_level, 0:run_length]
        return I, J+1

    def calcuteS(self,rlmatrix):
        return np.apply_over_axes(np.sum, rlmatrix, axes=(0, 1))[0, 0]
    '''
    The following code realizes the extraction of 11 gray runoff matrix features
    '''
    #1.SRE
    def getShortRunEmphasis(self,rlmatrix):
            I, J = self.calcuteIJ(rlmatrix)
            numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.divide, rlmatrix, (J*J)), axes=(0, 1))[0, 0]
            S = self.calcuteS(rlmatrix)
            return numerator / S
    #2.LRE
    def getLongRunEmphasis(self,rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.multiply, rlmatrix, (J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S
    #3.GLN
    def getGrayLevelNonUniformity(self,rlmatrix):
        G = np.apply_over_axes(np.sum, rlmatrix, axes=1)
        numerator = np.apply_over_axes(np.sum, (G*G), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S
    # 4. RLN
    def getRunLengthNonUniformity(self,rlmatrix):
            R = np.apply_over_axes(np.sum, rlmatrix, axes=0)
            numerator = np.apply_over_axes(np.sum, (R*R), axes=(0, 1))[0, 0]
            S = self.calcuteS(rlmatrix)
            return numerator / S

        # 5. RP
    def getRunPercentage(self,rlmatrix):
            gray_level, run_length,_ = rlmatrix.shape
            num_voxels = gray_level * run_length
            return self.calcuteS(rlmatrix) / num_voxels

        # 6. LGLRE
    def getLowGrayLevelRunEmphasis(self,rlmatrix):
            I, J = self.calcuteIJ(rlmatrix)
            numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.divide, rlmatrix, (I*I)), axes=(0, 1))[0, 0]
            S = self.calcuteS(rlmatrix)
            return numerator / S

        # 7. HGL   
    def getHighGrayLevelRunEmphais(self,rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.multiply, rlmatrix, (I*I)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

        # 8. SRLGLE
    def getShortRunLowGrayLevelEmphasis(self,rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.divide, rlmatrix, (I*I*J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S
    # 9. SRHGLE
    def getShortRunHighGrayLevelEmphasis(self,rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        temp = self.apply_over_degree(np.multiply, rlmatrix, (I*I))
        print('-----------------------')
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.divide, temp, (J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S
 
    # 10. LRLGLE
    def getLongRunLow(self,rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        temp = self.apply_over_degree(np.multiply, rlmatrix, (J*J))
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.divide, temp, (J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S
 
    # 11. LRHGLE
    def getLongRunHighGrayLevelEmphais(self,rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.multiply, rlmatrix, (I*I*J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S
