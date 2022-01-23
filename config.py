import argparse


def get_args():
    parser = argparse.ArgumentParser("SNASNet")
    parser.add_argument('--exp_name', type=str, default='snn_test',  help='experiment name')
    parser.add_argument('--data_dir', type=str, default='dataset/', help='path to the dataset')
    parser.add_argument('--dataset', type=str, default='cifar100', help='[cifar10, cifar100]')
    parser.add_argument('--seed', default=1234, type=int)

    parser.add_argument('--timestep', type=int, default=5, help='timestep for SNN')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=300, help='epoch')
    parser.add_argument('--tau', type=float, default=4/3, help='neuron decay time factor')
    parser.add_argument('--threshold', type=float, default=1.0, help='neuron firing threshold')
    parser.add_argument('--celltype', type=str, default='backward', help='[forward, backward]')
    parser.add_argument('--second_avgpooling', type=int, default=2, help='momentum')

    parser.add_argument('--optimizer', type=str, default='sgd', help='[sgd, adam]')
    parser.add_argument('--scheduler', type=str, default='cosine', help='[step, cosine]')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learnng rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--val_interval', type=int, default=20, help='validate and save frequency')
    parser.add_argument('--num_search', type=int, default=5000, help='number of search samples')

    # parser.add_argument('--cnt_mat', type=str, default=5000, help='number of search samples')
    parser.add_argument('--cnt_mat', type=list, nargs='+', default=None)
    parser.add_argument('--savemodel_pth', type=str,  default=None)

    args = parser.parse_args()
    print(args)

    return args
