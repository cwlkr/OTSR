"""
Model definition Context Aware Network
"""import torch
from torch import nn

class ContextNet(nn.Module):
    def __init__(self, nclasses=2, width_factor=1, slope=0.2, l2=0):
        super(ContextNet,self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 6*width_factor, kernel_size=3),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(6*width_factor),
            nn.Conv2d(6*width_factor,8*width_factor,stride=2,kernel_size=3),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*width_factor),
            nn.Conv2d(8*width_factor,8*width_factor,stride=1,kernel_size=3),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*width_factor),
            nn.Conv2d(8*width_factor,8*width_factor,stride=2,kernel_size=3),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*width_factor),
            nn.Conv2d(8*width_factor,8*width_factor,stride=1,kernel_size=3),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*width_factor))

        self.encoder3 = nn.Sequential(
            nn.Conv2d(3, 6*(width_factor+l2), kernel_size=3),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(6*(width_factor+l2)),
            nn.Conv2d(6*(width_factor+l2),8*(width_factor+l2),stride=2,kernel_size=3),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*(width_factor+l2)),
            nn.Conv2d(8*(width_factor+l2),8*(width_factor+l2),stride=2,kernel_size=3),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*(width_factor+l2)),
            nn.Conv2d(8*(width_factor+l2),8*(width_factor+l2),stride=1,kernel_size=3),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*(width_factor+l2)),
            nn.Conv2d(8*(width_factor+l2),8*(width_factor+l2),stride=2,kernel_size=3),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*(width_factor+l2)),
            nn.Conv2d(8*(width_factor+l2),8*(width_factor+l2),stride=2,kernel_size=3),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*(width_factor+l2)),
            nn.Conv2d(8*(width_factor+l2),8*width_factor,kernel_size=2),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*width_factor)

            )

        self.fc = nn.Sequential(
            nn.Conv2d(8*width_factor*2,8*width_factor*2,kernel_size=2, stride=2),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*width_factor*2),
            nn.Conv2d(8*width_factor*2,8*width_factor*2,kernel_size=2),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*width_factor*2),
            nn.Conv2d(8*width_factor*2,8*width_factor,kernel_size=2),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*width_factor),
            nn.Conv2d(8*width_factor,8*width_factor,kernel_size=2),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*width_factor),
            nn.Conv2d(8*width_factor,8*width_factor,kernel_size=2),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*width_factor),
            nn.Conv2d(8*width_factor,8*width_factor,kernel_size=1),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*width_factor),
            nn.Conv2d(8*width_factor,8*width_factor,kernel_size=1),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*width_factor)
            )
        self.classifier = nn.Sequential(nn.Conv2d(8*width_factor, nclasses, kernel_size=1))

    def forward(self, x, y):
        x = torch.cat([self.encoder1(x), self.encoder3(y)], 1)
        x = self.fc(x)
        return self.classifier(x).squeeze()

    def test1(self, x):
        x = self.encoder1(x)
        return x

    def test3(self, x):
        x = self.encoder3(x)
        return x

class ContextNetS(nn.Module):
    def __init__(self, nclasses=3, width_factor=5, slope=0.2, l2=0):
        super(ContextNetS,self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 6*width_factor, kernel_size=3),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(6*width_factor),
            nn.Conv2d(6*width_factor,8*width_factor,stride=2,kernel_size=3),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*width_factor),
            nn.Conv2d(8*width_factor,8*width_factor,stride=1,kernel_size=3),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*width_factor),
            nn.Conv2d(8*width_factor,8*width_factor,stride=2,kernel_size=3),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*width_factor),
            nn.Conv2d(8*width_factor,8*width_factor,stride=1,kernel_size=3),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*width_factor))

        self.encoder3 = nn.Sequential(
            nn.Conv2d(3, 6*(width_factor+l2), kernel_size=3),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(6*(width_factor+l2)),
            nn.Conv2d(6*(width_factor+l2),8*(width_factor+l2),stride=2,kernel_size=3),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*(width_factor+l2)),
            nn.Conv2d(8*(width_factor+l2),8*(width_factor+l2),stride=2,kernel_size=3),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*(width_factor+l2)),
            nn.Conv2d(8*(width_factor+l2),8*(width_factor+l2),stride=1,kernel_size=3),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*(width_factor+l2)),
            nn.Conv2d(8*(width_factor+l2),8*(width_factor+l2),stride=2,kernel_size=3),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*(width_factor+l2)),
            nn.Conv2d(8*(width_factor+l2),8*(width_factor+l2),stride=2,kernel_size=3),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*(width_factor+l2)),
            nn.Conv2d(8*(width_factor+l2),8*width_factor,kernel_size=2),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*width_factor)

            )

        self.fc = nn.Sequential(
            nn.Conv2d(8*width_factor*2,8*width_factor*2,kernel_size=2, stride=2),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*width_factor*2),
            nn.Conv2d(8*width_factor*2,8*width_factor*2,kernel_size=2),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*width_factor*2),
            nn.Conv2d(8*width_factor*2,8*width_factor,kernel_size=2),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*width_factor),
            nn.Conv2d(8*width_factor,8*width_factor,kernel_size=2),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*width_factor),
            nn.Conv2d(8*width_factor,8*width_factor,kernel_size=2),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*width_factor),
            nn.Conv2d(8*width_factor,8*width_factor,kernel_size=1),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*width_factor),
            nn.Conv2d(8*width_factor,8*width_factor,kernel_size=1),
            nn.LeakyReLU(slope, True),
            nn.BatchNorm2d(8*width_factor)
            )
        self.classifier = nn.Sequential(nn.Conv2d(8*width_factor, nclasses, kernel_size=1))
    def forward(self, x):
        x = torch.cat([self.encoder1(x[:,:,80:224-80,80:224-80]), self.encoder3(x)], 1)
        x = self.fc(x)
        return self.classifier(x).squeeze()

    def test1(self, x):
        x = self.encoder1(x)
        return x

    def test3(self, x):
        x = self.encoder3(x)
        return x
# +
if __name__ == '__main__':
    from torchsummary import summary
    gpuid=0
    print(torch.cuda.get_device_properties(gpuid))
    torch.cuda.set_device(gpuid)
    device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')
    model=ContextNetS(nclasses = 3, width_factor=6).to(device)
    summary(model, (3,224,224))

    model.test1(torch.zeros((5,3,64,64)).to(device)).shape
    model.test3(torch.zeros((5,3,224,224)).to(device)).shape
    model.forward(torch.zeros((5,3,224,224)).to(device)).shape
