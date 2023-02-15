import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(
        self
    ):
        super(CNNEncoder, self).__init__()
        conv1 = nn.Conv2d(2, 4, kernel_size=(6,3), padding = 1, stride = 2)
        nn.init.kaiming_normal_(conv1.weight, a = 0.1)
    
        self.cnn_block1 = nn.Sequential(
            conv1,
            nn.BatchNorm2d(4),
            nn.GELU()
        )
        
        conv2 = nn.Conv2d(4, 8, kernel_size=(5,4), padding = 1, stride = 2)
        nn.init.kaiming_normal_(conv2.weight, a = 0.1)
    
        self.cnn_block2 = nn.Sequential(
            conv2,
            nn.BatchNorm2d(8),
            nn.GELU()
        )
        
        conv3 = nn.Conv2d(8, 16, kernel_size=3, padding = 1)
        nn.init.kaiming_normal_(conv3.weight, a = 0.1)
    
        self.cnn_block3 = nn.Sequential(
            conv3,
            nn.BatchNorm2d(16),
            nn.GELU()
        )
        
        conv4 = nn.Conv2d(16,32, kernel_size=3,padding = 1)
        nn.init.kaiming_normal_(conv4.weight, a = 0.1)
        
        self.cnn_block4 = nn.Sequential(
            conv4,
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        
        conv5 = nn.Conv2d(32,64, kernel_size=3,padding = 1)
        nn.init.kaiming_normal_(conv5.weight, a = 0.1)
        
        self.cnn_block5 = nn.Sequential(
            conv5,
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        conv6 = nn.Conv2d(64,64, kernel_size=3,padding = 1)
        nn.init.kaiming_normal_(conv6.weight, a = 0.1)
        
        self.cnn_block6 = nn.Sequential(
            conv6
        )
        
        
        

    def forward(self, x):
        x1 = self.cnn_block1(x)
        #print(x1.shape)
        x2 = self.cnn_block2(x1)
        #print(x2.shape)
        x3 = self.cnn_block3(x2)
        #print(x3.shape)
        x4 = self.cnn_block4(x3)
        #print(x4.shape)
        x5 = self.cnn_block5(x4)
        #print(x5.shape)
        x6 = self.cnn_block6(x5)

        return x6
    
class CNNDecoder(nn.Module) :
    
    def __init__(
        self
    ) :
        super(CNNDecoder, self).__init__()

        conv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding = 1)
        nn.init.kaiming_normal_(conv1.weight, a = 0.1)
    
        self.cnn_block1 = nn.Sequential(
            conv1,
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        conv2 = nn.ConvTranspose2d(64,32, kernel_size=3,padding = 1)
        nn.init.kaiming_normal_(conv2.weight, a = 0.1)
        
        self.cnn_block2 = nn.Sequential(
            conv2,
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        
        conv3 = nn.ConvTranspose2d(32,16, kernel_size=3,padding = 1)
        nn.init.kaiming_normal_(conv3.weight, a = 0.1)
        
        self.cnn_block3 = nn.Sequential(
            conv3,
            nn.BatchNorm2d(16),
            nn.GELU()
        )
        
        conv4 = nn.ConvTranspose2d(16,8, kernel_size=3,padding = 1)
        nn.init.kaiming_normal_(conv4.weight, a = 0.1)
        
        self.cnn_block4 = nn.Sequential(
            conv4,
            nn.BatchNorm2d(8),
            nn.GELU()
        )
        
        conv5 = nn.ConvTranspose2d(8,4, kernel_size=(5,4),padding = 1, stride = 2)
        nn.init.kaiming_normal_(conv5.weight, a = 0.1)
        
        self.cnn_block5 = nn.Sequential(
            conv5,
            nn.BatchNorm2d(4),
            nn.GELU()
        )
        
        conv6 = nn.ConvTranspose2d(4,2, kernel_size=(6,3), padding = 1, stride=2)
        nn.init.kaiming_normal_(conv6.weight, a = 0.1)
        
        
        self.cnn_block6 = nn.Sequential(
            conv6,    
        )
    
    def forward(self, x) :
        x1 = self.cnn_block1(x)
        #print(x1.shape)
        x2 = self.cnn_block2(x1)
        #print(x2.shape)
        x3 = self.cnn_block3(x2)
        #print(x3.shape)
        x4 = self.cnn_block4(x3)
        #print(x4.shape)
        x5 = self.cnn_block5(x4)
        #print(x5.shape)
        x6 = self.cnn_block6(x5)

        return x6
    
class CNNAutoEncoder(nn.Module) :
    
    def __init__(
        self
    ) :
        super(CNNAutoEncoder, self).__init__()
        self.encoder = CNNEncoder()
        self.decoder = CNNDecoder()
        self.ln = nn.MSELoss()
    
    def encode(self, x) :
        
        return self.encoder(x)
    
    def decode(self, z) :
        
        return self.decoder(z)
    
    def reconstruct(self, x) :
        
        return self.decode(self.encode(x)) 
    
    def compute_reconstruction_loss(self, x, type = "positive") :
        loss = self.ln(x, self.reconstruct(x))
        if type == "positive" :
            return loss 
        elif type == "negative" :
            return -loss 
        else :
            raise Exception(f"{type} type of data is not supported")