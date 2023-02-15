import torch.nn as nn 

class CNNClassifier(nn.Module) :
    
    def __init__(
        self,
        n_category : int = 5,
        h_fc : int = 256
        
    ) :
        super(CNNClassifier, self).__init__()
        conv1 = nn.Conv2d(2, 4, kernel_size=(6,3), padding = 1, stride = 2)
        nn.init.kaiming_normal_(conv1.weight, a = 0.1)
    
        self.cnn_block1 = nn.Sequential(
            conv1,
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(4),
            nn.GELU()
        )
        
        conv2 = nn.Conv2d(4, 8, kernel_size=(5,4), padding = 1, stride = 2)
        nn.init.kaiming_normal_(conv2.weight, a = 0.1)
    
        self.cnn_block2 = nn.Sequential(
            conv2,
            nn.BatchNorm2d(8),
            nn.Dropout2d(0.2),
            nn.GELU()
        )
        
        conv3 = nn.Conv2d(8, 16, kernel_size=3, padding = 1, stride = 2)
        nn.init.kaiming_normal_(conv3.weight, a = 0.1)
    
        self.cnn_block3 = nn.Sequential(
            conv3,
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(16),
            nn.GELU()
        )
        
        conv4 = nn.Conv2d(16, 16, kernel_size=3, padding = 1, stride = 2)
        nn.init.kaiming_normal_(conv4.weight, a = 0.1)
    
        self.cnn_block4 = nn.Sequential(
            conv4,
            nn.Dropout2d(0.4),
            nn.Flatten()
        )
        
        self.linear = nn.Sequential(
            nn.Linear(1024, h_fc),
            nn.LeakyReLU(),
            nn.Linear(h_fc, 2 * h_fc),
            nn.GELU(),
            nn.Linear(2 * h_fc, h_fc),
            nn.LeakyReLU(),
            nn.Linear(h_fc, n_category)
        )
        
    def forward(self, x) :
        x1 = self.cnn_block1(x)
        x2 = self.cnn_block2(x1)
        x3 = self.cnn_block3(x2)
        x4 = self.cnn_block4(x3)
        z = self.linear(x4)
        return z 