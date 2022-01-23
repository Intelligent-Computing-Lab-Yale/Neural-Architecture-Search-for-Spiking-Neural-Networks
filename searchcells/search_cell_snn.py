import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional, layer, surrogate, neuron


class ScaleLayer(nn.Module):
   def __init__(self):
       super().__init__()
       self.scale = torch.tensor(0.)

   def forward(self, input):
       return input * self.scale

class Neuronal_Cell(nn.Module):
    def __init__(self,args,  in_channel, out_channel, con_mat):
        super(Neuronal_Cell, self).__init__()
        self.cell_architecture = nn.ModuleList([])
        self.con_mat = con_mat
        for col in range(1,4):
            for row in range(col):
                op = con_mat[row,col]
                if op==0:
                    self.cell_architecture.append(ScaleLayer())
                elif op == 1:
                    self.cell_architecture.append(nn.Identity())
                elif op == 2:
                    self.cell_architecture.append(nn.Sequential(
                        neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau,
                                       surrogate_function=surrogate.ATan(),
                                       detach_reset=True),
                        nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1),
                                  stride=(1, 1), bias=False),
                        nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1,
                                       affine=True, track_running_stats=True)))
                elif op == 3:
                    self.cell_architecture.append(nn.Sequential(
                                neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau,
                                               surrogate_function=surrogate.ATan(),
                                               detach_reset=True),
                                nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3),
                                          stride=(1, 1), padding=(1,1), bias=False),
                                nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1,
                                               affine=True, track_running_stats=True)))
                elif op == 4:
                    self.cell_architecture.append(nn.AvgPool2d(kernel_size=3, stride=1, padding=1))


    def forward(self, x_in):
        x_1 = self.cell_architecture[0](x_in)
        x_2 = self.cell_architecture[1](x_in) + self.cell_architecture[2](x_1)
        x_3 = self.cell_architecture[3](x_in) + self.cell_architecture[4](x_1) + self.cell_architecture[5](x_2)

        return x_3




class Neuronal_Cell_backward(nn.Module):
    def __init__(self,args,  in_channel, out_channel, con_mat):
        super(Neuronal_Cell_backward, self).__init__()

        self.cell_architecture = nn.ModuleList([])
        self.con_mat = con_mat
        self.cell_architecture_back = nn.ModuleList([])

        self.last_xin = 0.
        self.last_x1 = 0.
        self.last_x2 = 0.

        for col in range(1,4):
            for row in range(col):
                op = con_mat[row,col]
                if op==0:
                    self.cell_architecture.append(ScaleLayer())
                elif op == 1:
                    self.cell_architecture.append(nn.Identity())
                elif op == 2:
                    self.cell_architecture.append(nn.Sequential(
                        neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau,
                                       surrogate_function=surrogate.ATan(),
                                       detach_reset=True),
                        nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1),
                                  stride=(1, 1), bias=False),
                        nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1,
                                       affine=True, track_running_stats=True)))
                    # l_idx +=1
                elif op == 3:
                    self.cell_architecture.append(nn.Sequential(
                                neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau,
                                               surrogate_function=surrogate.ATan(),
                                               detach_reset=True),
                                nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3),
                                          stride=(1, 1), padding=(1,1), bias=False),
                                nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1,
                                               affine=True, track_running_stats=True)))

                elif op == 4:
                    self.cell_architecture.append(nn.AvgPool2d(kernel_size=3, stride=1, padding=1))

        for col in range(0, 3):
            for row in range(col+1, 4):
                op = con_mat[row, col]
                if op == 0:
                    self.cell_architecture_back.append(ScaleLayer())
                elif op == 1:
                    self.cell_architecture_back.append(nn.Identity())
                elif op == 2:
                    self.cell_architecture_back.append(nn.Sequential(
                        neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau,
                                       surrogate_function=surrogate.ATan(),
                                       detach_reset=True),
                        nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1),
                                  stride=(1, 1), bias=False),
                        nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1,
                                       affine=True, track_running_stats=True)))
                elif op == 3:
                    self.cell_architecture_back.append(nn.Sequential(
                        neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau,
                                       surrogate_function=surrogate.ATan(),
                                       detach_reset=True),
                        nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3),
                                  stride=(1, 1), padding=(1, 1), bias=False),
                        nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1,
                                       affine=True, track_running_stats=True)))
                elif op == 4:
                    self.cell_architecture_back.append(nn.AvgPool2d(kernel_size=3, stride=1, padding=1))

    def forward(self, x_in):
        x_1 = self.cell_architecture[0](x_in + self.last_xin)
        x_2 = self.cell_architecture[1](x_in+ self.last_xin ) + self.cell_architecture[2](x_1 + self.last_x1)
        x_3 = self.cell_architecture[3](x_in+ self.last_xin) + self.cell_architecture[4](x_1+ self.last_x1) + self.cell_architecture[5](x_2+ self.last_x2)

        self.last_xin = self.cell_architecture_back[0](x_1+ self.last_x1)+ self.cell_architecture_back[1](x_2+ self.last_x2)+ self.cell_architecture_back[2](x_3)
        self.last_x1 = self.cell_architecture_back[3](x_2+ self.last_x2)+ self.cell_architecture_back[4](x_3)
        self.last_x2 =  self.cell_architecture_back[5](x_3)

        return x_3





