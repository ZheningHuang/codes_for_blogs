import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import cv2
import torchvision
from loss.losses import DiceLoss

class Block(nn.Module): 
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1=nn.Conv2d(in_ch,out_ch,3, stride=1, padding=0) 
        self.conv2=nn.Conv2d(out_ch,out_ch,3, stride=1, padding=0) 
        self.relu=nn.ReLU() 
    def forward(self,x):
        return self.relu(self.conv2(self.relu(self.conv1(x)))) 

class Encoder(nn.Module):
    def __init__(self, chs):
        super().__init__() 
        self.enc_blocks=nn.ModuleList([Block(chs[i],chs[i+1]) for i in range (len(chs)-1)])
        self.pool =nn.MaxPool2d(2)
    def forward(self,x):
        ftrs = [] 
        for block in self.enc_blocks:
            x=block(x) 
            ftrs.append(x) 
            x=self.pool(x)   
        return ftrs

class Decoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.chs=chs 
        self.upconvs=nn.ModuleList([nn.ConvTranspose2d(chs[i],chs[i+1],2,stride=2) for i in range(len(chs)-1)]) 
        
        self.dec_blocks=nn.ModuleList([Block(chs[i],chs[i+1]) for i in range (len(chs)-1)])
        
    def forward(self, x, encoder_features): 
        for i in range (len(self.chs)-1):
            x=self.upconvs[i](x)
            enc_ftrs=self.crop (encoder_features[i],x) 
            x=torch.cat([x,enc_ftrs], dim=1) 
            x= self.dec_blocks[i](x)
        return x 
    def crop(self, enc_ftrs_input, x):
        _,_, H,W=x.shape 
        enc_ftrs= torchvision.transforms.CenterCrop([H,W])(enc_ftrs_input)
        
        return enc_ftrs

class UNet(pl.LightningModule):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024,512,256,128,64),num_class=1,retain_dim=True,out_sz=( 384,544)):
        super().__init__()
        self.encoder=Encoder(enc_chs) 
        self.decoder =Decoder(dec_chs) 
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1) 
        self.retain_dim=retain_dim 
        self.out_sz=out_sz
        self.loss_dice=DiceLoss()

    def forward(self,x): 
        enc_ftrs=self.encoder(x) 
        out= self.decoder(enc_ftrs[::-1][0],enc_ftrs[::-1][1:]) 
        out= self.head (out) 
        if self.retain_dim: 
            out=F.interpolate(out,self.out_sz) 
        return out

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_dice(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_dice(y_hat, y)
        self.log('val_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
       x, y = batch
       y_hat = self.forward(x)
       loss = self.loss_dice(y_hat, y)
       self.log('test_loss', loss, on_step=True)
       print (loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
