import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from glob import glob
from torch import nn
import torchvision
import torch.nn.functional as F


##create model 
device="cuda" if torch.cuda.is_available() else "cpu"
print ("Using {} device".format(device))


##define Unet architecture
## Let us first define a block as the UNet structure has quite a lot of repeating operation. 

class Block(nn.Module): ##class this as a nn.module so you can recall this with nn.ModuleList
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1=nn.Conv2d(in_ch,out_ch,3, stride=1, padding=0) ## all the kernel used in UNet are 3x3 size. no padding as shown in the papaer
        self.conv2=nn.Conv2d(out_ch,out_ch,3, stride=1, padding=0) ## same channel convolution happends in all the blocks.
        self.relu=nn.ReLU() #relu operation is always the final step in each block
    def forward(self,x):
        return self.relu(self.conv2(self.relu(self.conv1(x)))) 

# note here some people will use relu everytime after convolution, weather this is used is still agruable
'''
enc_block = Block(1, 64)
x= torch.randn(1, 1, 572, 572) 
#note that the input shape of conv2d is 
#Input: (N, C_{in}, H_{in}, W_{in})
#Output: (N, C_{out}, H_{out}, W_{out})
'''
#print (enc_block(x).shape)

class Encoder(nn.Module):
    def __init__(self, chs):
        super().__init__() ##todo: find out why this is needed
        self.enc_blocks=nn.ModuleList([Block(chs[i],chs[i+1]) for i in range (len(chs)-1)])
        ## now the self.enc_block has many blocks (2dcov+relu+2dconv+relu)) that can be recalled when needed. 
        self.pool =nn.MaxPool2d(2)
    def forward(self,x):
        ftrs = [] # the reason to have every output stored is that we need to use this in the decoder session. other than that there is no need to store them, we just need to most updated x, as what we do in the decoder block
        for block in self.enc_blocks:
            x=block(x) # given x, x will go through a 2dconv+relu+2dconv+relu
            ftrs.append(x) #x will then be saved into ftrs
            x=self.pool(x) #x will then go through maxpooling
            # the processed x will go through the whole process again.
        return ftrs
'''
encoder=Encoder()
x= torch.randn(1, 3, 572, 572)
ftrs = encoder(x)
for ftr in ftrs: print(ftr.shape)
'''
##great! let's define the decoder than we can put them altogether and have the whole model ready

class Decoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.chs=chs # we have to define this as an parameter of Decoder as we used it in other functions
        self.upconvs=nn.ModuleList([nn.ConvTranspose2d(chs[i],chs[i+1],2,stride=2) for i in range(len(chs)-1)]) #why nn.Moduleslist is necessary
        #self.upconvs=nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)
        self.dec_blocks=nn.ModuleList([Block(chs[i],chs[i+1]) for i in range (len(chs)-1)])
        #basically here a series of upconvs and decovn block are defined with different dimensions
    def forward(self, x, encoder_features): # the inputs include x and encoder_features what are the enconder features
        for i in range (len(self.chs)-1):
            x=self.upconvs[i](x)
            enc_ftrs=self.crop (encoder_features[i],x) # how to crop, will be defined later
            x=torch.cat([x,enc_ftrs], dim=1) # concatanate them together 
            x= self.dec_blocks[i](x)
        return x #as mentioned before, only need to most updataed x 
    def crop(self, enc_ftrs_input, x):
        _,_, H,W=x.shape # get the height and weight of shape 
        enc_ftrs= torchvision.transforms.CenterCrop([H,W])(enc_ftrs_input)# what do transforms do?
        #transforms are common image transformations  torchvision.transforms.CenterCrop, parameters are the h and w to crop, input is the image
        return enc_ftrs

'''
decoder = Decoder()
x = torch.randn(1, 1024, 28, 28)
print (decoder(x, ftrs[::-1][1:]).shape) 
'''
# ftrs[::-1] it starts from the end towards the first taking each element. 
# ftrs[::-1][1:] it starts from the end towards the first taking each element and ignore the first one.

#Now, let us assumble the UNet with all the module we built.


class UNet(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024,512,256,128,64),num_class=1,retain_dim=False,out_sz=(572,572)):
        super().__init__()
        self.encoder=Encoder(enc_chs) 
        self.decoder =Decoder(dec_chs) 
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1) #input channel is last dim and output channel is the class number, using 1x1 kernal. This is the last step of unet, converting the 64 channel to segmentaion channels. size remain the same so use 1x1 convolution
        self.retain_dim=retain_dim #what is this?
        self.out_sz=out_sz
    def forward(self,x): 
        enc_ftrs=self.encoder(x) #run encode with x as input, output the results for all blocks on the ftrs
        out= self.decoder(enc_ftrs[::-1][0],enc_ftrs[::-1][1:]) # input as the last one of the ftrs, and the encoder_features is the other output in ftrs
        out= self.head (out) #after decoder, do a final 2d convolution 
        if self.retain_dim: #rescale the sample to the original size
            out=F.interpolate(out,self.out_sz) #what does F mean here
        return out


model=UNet()

#######