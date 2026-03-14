import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self , num_classes = 1000):
        super(AlexNet, self).__init__()

        # feature extractor : 5 convolutional layers
        self.features = nn.Sequential(

            # layer 1
            nn.Conv2d(3 , 96, kernel_size= 11, stride = 4, padding=2),
            nn.ReLU(inplace= True),
            nn.LocalResponseNorm(size= 5 , alpha= 0.0001 , beta = 0.75, k = 2),
            nn.MaxPool2d(kernel_size= 3 , stride= 2),

            #layer 2
            nn.Conv2d(96, 256, kernel_size= 5 , padding=2),
            nn.ReLU(inplace= True),
            nn.LocalResponseNorm(size= 5 , alpha= 0.0001, beta = 0.75, k = 2),
            nn.MaxPool2d(kernel_size=3 , stride=2),

            # layer 3, 4 and 5 are connected directly without pooling
            nn.Conv2d(256 , 384, kernel_size= 3 , padding=1),
            nn.ReLU(inplace = True),

            nn.Conv2d(384 , 384, kernel_size= 3 , padding = 1),
            nn.ReLU(inplace= True),

            nn.Conv2d(384 , 256 , kernel_size= 3 , padding=1),
            nn.ReLU(inplace= True),

            nn.MaxPool2d(kernel_size=3 , stride= 2),
        )

        # fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(p= 0.5) ,
            nn.Linear(256 * 6 * 6 , 4096),
            nn.ReLU(inplace = True),

            nn.Dropout(p = 0.5),
            nn.Linear(4096 , 4096),
            nn.ReLU(inplace= True),

            nn.Dropout(p = 0.5),
            nn.Linear(4096 , num_classes),
        )

        self._initialize_weights()

    def forward(self , x):
        x = self.features(x)
        x = torch.flatten(x , 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m ,nn.Linear):
                nn.init.normal_(m.weight, mean = 0 , std = 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias , 0)

        nn.init.constant_(self.features[4].bias, 1)   # 2nd Conv (index 4 in Sequential)
        nn.init.constant_(self.features[10].bias, 1)  # 4th Conv (index 10)
        nn.init.constant_(self.features[12].bias, 1)  # 5th Conv (index 12)
        nn.init.constant_(self.classifier[1].bias, 1) # 1st FC (index 1)
        nn.init.constant_(self.classifier[4].bias, 1) # 2nd FC (index 4)
        nn.init.constant_(self.classifier[6].bias, 1) # 3rd FC (index 6)