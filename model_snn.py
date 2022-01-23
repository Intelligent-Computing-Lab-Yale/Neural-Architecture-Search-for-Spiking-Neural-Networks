import torch
import torch.nn as nn
import numpy as np
from spikingjelly.clock_driven import functional, layer, surrogate, neuron
from searchcells.search_cell_snn import Neuronal_Cell,Neuronal_Cell_backward

def logdet(K):
    s, ld = np.linalg.slogdet(K)
    return  ld

def find_best_neuroncell(args, trainset):

    search_batchsize = 256
    repeat = 2

    train_data = torch.utils.data.DataLoader(trainset, batch_size=search_batchsize,
                                               shuffle=True, pin_memory=True, num_workers=4)
    scores = []
    history = []
    neuron_type = 'LIFNode'


    with torch.no_grad():
        for i in range(args.num_search):
            while (1):
                con_mat =connection_matrix_gen(args, num_node=4, num_options=5)

                # sanity check on connection matrix
                neigh2_cnts = torch.mm(con_mat, con_mat)
                neigh3_cnts = torch.mm(neigh2_cnts, con_mat)
                neigh4_cnts = torch.mm(neigh3_cnts, con_mat)
                connection_graph = con_mat + neigh2_cnts + neigh3_cnts + neigh4_cnts

                for node in range(3):
                    if connection_graph[node,3] ==0: # if any node doesnt send information to the last layer, remove it
                        con_mat[:, node] = 0
                        con_mat[node,:] = 0
                for node in range(3):
                    if connection_graph[0,node+1] ==0: # if any node doesnt get information from the input layer, remove it
                        con_mat[:, node+1] = 0
                        con_mat[node+1,:] = 0

                if con_mat[0, 3] != 0: # ensure direct connection between input=>output for fast information propagation
                    break


            searchnet = SNASNet(args, con_mat)
            searchnet.cuda()

            searchnet.K = np.zeros((search_batchsize, search_batchsize))
            searchnet.num_actfun = 0

            def computing_K_eachtime(module, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                out = out.view(out.size(0), -1)
                batch_num , neuron_num = out.size()
                x = (out > 0).float()

                full_matrix = torch.ones((search_batchsize, search_batchsize)).cuda() * neuron_num
                sparsity = (x.sum(1)/neuron_num).unsqueeze(1)
                norm_K = (sparsity @ (1-sparsity.t())) + ((1-sparsity) @ sparsity.t())
                rescale_factor = torch.div(0.5* torch.ones((search_batchsize, search_batchsize)).cuda(), norm_K+1e-3)
                K1_0 = (x @ (1 - x.t()))
                K0_1 = ((1-x) @ x.t())
                K_total = (full_matrix - rescale_factor * (K0_1 + K1_0))

                searchnet.K = searchnet.K + (K_total.cpu().numpy())
                searchnet.num_actfun += 1

            s = []
            for name, module in searchnet.named_modules():
                if neuron_type in str(type(module)):
                    module.register_forward_hook(computing_K_eachtime)

            for j in range(repeat):
                searchnet.K = np.zeros((search_batchsize, search_batchsize))
                searchnet.num_actfun = 0
                data_iterator = iter(train_data)
                inputs, targets = next(data_iterator)
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = searchnet(inputs)
                s.append(logdet(searchnet.K/ (searchnet.num_actfun)))

            scores.append(np.mean(s))
            history.append(con_mat)

        print ("mean / var:", np.mean(scores), np.var(scores))
        print ("max score:", max(scores))
        best_idx = (np.argsort(scores))[-1]
        best_policy = history[best_idx]
    return best_policy



def connection_matrix_gen(args, num_node = 4, num_options = 5):

    if args.celltype == 'forward':
        upper_cnts = torch.triu(torch.randint(num_options, size=(num_node, num_node)), diagonal=1)
        cnts = upper_cnts
    elif args.celltype == 'backward':
        upper_cnts = torch.triu(torch.randint(num_options, size=(num_node, num_node)), diagonal=1)
        lower_cnts = torch.tril(torch.randint(num_options, size=(num_node, num_node)), diagonal=-1)
        selection_mask = torch.triu(torch.randint(2, size=(num_node, num_node)), diagonal=1)
        tr_selection_mask = 1 - selection_mask.permute(1,0)
        cnts = selection_mask * upper_cnts + tr_selection_mask * lower_cnts

    return cnts


class SNASNet(nn.Module):
    def __init__(self, args, con_mat):
        super(SNASNet, self).__init__()

        self.con_mat = con_mat
        self.total_timestep = args.timestep
        self.second_avgpooling = args.second_avgpooling

        if args.dataset == 'cifar10':
            self.num_class = 10
            self.num_final_neuron = 100
            self.num_cluster = 10
            self.in_channel = 3
            self.img_size = 32
            self.first_out_channel = 128
            self.spatial_decay = 2 *self.second_avgpooling
            self.classifier_inter_ch = 1024
            self.stem_stride = 1
        elif args.dataset == 'cifar100':
            self.num_class = 100
            self.num_final_neuron = 500
            self.num_cluster = 5
            self.in_channel = 3
            self.img_size = 32
            self.first_out_channel = 128
            self.spatial_decay = 2 *self.second_avgpooling
            self.classifier_inter_ch = 1024
            self.stem_stride = 1
        elif args.dataset == 'tinyimagenet':
            self.num_class = 200
            self.num_final_neuron = 1000
            self.num_cluster = 5
            self.in_channel = 3
            self.img_size = 64
            self.first_out_channel = 128
            self.spatial_decay = 4 * self.second_avgpooling
            self.classifier_inter_ch = 4096
            self.stem_stride = 2

        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channel, self.first_out_channel, kernel_size=3, stride=self.stem_stride, padding=1, bias=False),
            nn.BatchNorm2d(self.first_out_channel, affine=True),
        )

        if args.celltype == "forward":
            self.cell1 = Neuronal_Cell(args, self.first_out_channel, self.first_out_channel, self.con_mat)
        elif args.celltype == "backward":
            self.cell1 = Neuronal_Cell_backward(args, self.first_out_channel, self.first_out_channel, self.con_mat)
        else:
            print ("not implemented")
            exit()

        self.downconv1 = nn.Sequential(
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau= args.tau,
                                                      surrogate_function=surrogate.ATan(),
                                                      detach_reset=True),
                                        nn.Conv2d(128, 256, kernel_size=(3, 3),
                                                  stride=(1, 1), padding=(1,1), bias=False),
                                        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1,
                                                       affine=True, track_running_stats=True)
                                        )
        self.resdownsample1 = nn.AvgPool2d(2,2)

        if args.celltype == "forward":
            self.cell2 = Neuronal_Cell(args, 256, 256, self.con_mat)
        elif args.celltype == "backward":
            self.cell2 = Neuronal_Cell_backward(args, 256, 256, self.con_mat)
        else:
            print ("not implemented")
            exit()

        self.last_act = nn.Sequential(
                        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau,
                                       surrogate_function=surrogate.ATan(),
                                       detach_reset=True)
        )
        self.resdownsample2 = nn.AvgPool2d(self.second_avgpooling,self.second_avgpooling)

        self.classifier = nn.Sequential(
            layer.Dropout(0.5),
            nn.Linear(256*(self.img_size//self.spatial_decay)*(self.img_size//self.spatial_decay), self.classifier_inter_ch, bias=False),
            neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True),
        nn.Linear(self.classifier_inter_ch, self.num_final_neuron, bias=True))
        self.boost = nn.AvgPool1d(self.num_cluster, self.num_cluster)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a =2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        self.neuron_init()

        acc_voltage = 0
        batch_size = input.size(0)
        static_x = self.stem(input)

        for t in range(self.total_timestep):
            x = self.cell1(static_x)
            x = self.downconv1(x)
            x = self.resdownsample1(x)
            x = self.cell2(x)
            x = self.last_act(x)
            x = self.resdownsample2(x)
            x = x.view(batch_size, -1)
            x = self.classifier(x)
            acc_voltage = acc_voltage + self.boost(x.unsqueeze(1)).squeeze(1)
        acc_voltage = acc_voltage/self.total_timestep
        return acc_voltage


    def neuron_init(self):
        self.cell1.last_xin =0.
        self.cell1.last_x1 =0.
        self.cell1.last_x2 =0.
        self.cell2.last_xin = 0.
        self.cell2.last_x1 = 0.
        self.cell2.last_x2 = 0.
        neuron_type = 'LIFNode'
        for name, module in self.cell1.named_modules():
            if neuron_type in str(type(module)):
                module.v = 0.
            if 'Dropout' in str(type(module)):
                module.mask = None
        for name, module in self.downconv1.named_modules():
            if neuron_type in str(type(module)):
                module.v = 0.
            if 'Dropout' in str(type(module)):
                module.mask = None
        for name, module in self.resdownsample1.named_modules():
            if neuron_type in str(type(module)):
                module.v = 0.
            if 'Dropout' in str(type(module)):
                module.mask = None
        for name, module in self.cell2.named_modules():
            if neuron_type in str(type(module)):
                module.v = 0.
            if 'Dropout' in str(type(module)):
                module.mask = None
        for name, module in self.resdownsample2.named_modules():
            if neuron_type in str(type(module)):
                module.v = 0.
            if 'Dropout' in str(type(module)):
                module.mask = None
        for name, module in self.last_act.named_modules():
            if neuron_type in str(type(module)):
                module.v = 0.
            if 'Dropout' in str(type(module)):
                module.mask = None
        for name, module in self.classifier.named_modules():
            if neuron_type in str(type(module)):
                module.v = 0.
            if 'Dropout' in str(type(module)):
                module.mask = None
