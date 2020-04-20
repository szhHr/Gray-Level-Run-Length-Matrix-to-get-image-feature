# Image-segmentation-Gray-Level-Run-Length-Matrix
use GLRLM to get image segmentation and feature

idea from https://blog.csdn.net/Beans___Lee/article/details/86579227

environments: python3
              numpy
              pillow
              itertools
              matplotlib
              csv
use:
      there are 11 kinds of functions in the file GrayRumatrix.py,
      " python3 main.py " to test code in the image "test.gif",
      the result will be stored in the data.csv.
      You can change the gif and rewrite the image path in the file"main.py"
      to test this code on your own image.

fuctions:

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
            
         
