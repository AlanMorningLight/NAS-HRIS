import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

class MixedOp(nn.Module):
    def __init__(self, C, stride, i, j):
        super(MixedOp, self).__init__()
        node_str = '{:}<-{:}'.format(i, j)
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op[ node_str ] = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.C_prev = C_prev
        self.C = C

        #判断reduction逻辑
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        self._steps = steps
        self._multiplier = multiplier
        self.edges = nn.ModuleDict()
        self._bns = nn.ModuleList()
        for i in range(2,self._steps+2):
            for j in range(i):
                stride = 2 if reduction and j < 2 else 1
                node_str = '{:}<-{:}'.format(i, j)
                oplist = [OPS[primitive](C, stride, False) for primitive in PRIMITIVES]
                self.edges[ node_str ] = nn.ModuleList( oplist )
        self.edge_keys = sorted(list(self.edges.keys()))
        '''print("**********")
        print(self.edges.keys())'''
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

    def forward(self, s0, s1, hardwts, index):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        '''print("**********")
        print(states)'''
        for i in range(2, self._steps+2):
            inter_nodes = []
            for j in range(i):
                # 记录
                node_str = '{:}<-{:}'.format(i, j)
                '''print("problem:***********")
                print(node_str)'''
                weights = hardwts[self.edge2index[node_str]]
                argmaxs = index[self.edge2index[node_str]].item()
                weigsum = sum(weights[_ie] * edge(states[j]) if _ie == argmaxs else weights[_ie]
                              for _ie, edge in enumerate(self.edges[node_str]))
                inter_nodes.append(weigsum)
            states.append(sum(inter_nodes))
        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):
    def __init__(self, gpu, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C  # 宽度 通道数
        self._num_classes = num_classes
        self._layers = layers  # 深度
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        #self.final = nn.Conv2d(C_prev, num_classes ,1, stride = 1, padding = 0, bias=False)
        if gpu == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(gpu))

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        #layer_channels   = [C    ] * layers + [C*2 ] + [C*2  ] * layers + [C*4 ] + [C*4  ] * layers
        #layer_reductions = [False] * layers + [True] + [False] * layers + [True] + [False] * layers

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        num_edge, edge2index = None, None
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            else:
                reduction = False
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
                if num_edge is None:num_edge, edge2index = cell.num_edges, cell.edge2index
                else:assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)

            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        #self.final = nn.Conv2d(C_prev, num_classes ,1, stride = 1, padding = 0, bias=False)
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.Conv2d(C_prev, num_classes ,1, stride = 1, padding = 0, bias=False), nn.ReLU(inplace=True))
        #self.arch_parameters = nn.Parameter(1e-3 * torch.randn(num_edge, len(search_space)))
        self._initialize_alphas()
        self.tau = 10


    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).to(self.device)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def bilinear_kernel(in_channels, out_channels, kernel_size):
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
        weight[range(in_channels), range(out_channels), :, :] = filt
        return torch.from_numpy(weight)

    def forward(self, input):

        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                while True:
                    #返回与input相同size，	从指数分布中抽取数字，取one hot
                    gumbels = -torch.empty_like(self.arch_parameters()[0]).exponential_().log()
                    logits = (self.arch_parameters()[0].log_softmax(dim=1) + gumbels) / self.tau
                    probs = nn.functional.softmax(logits, dim=1)
                    index = probs.max(-1, keepdim=True)[1]
                    one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
                    hardwts = one_h - probs.detach() + probs
                    if (torch.isinf(gumbels).any()) or (torch.isinf(probs).any()) or (torch.isnan(probs).any()):
                        continue
                    else:
                        break
            else:
                while True:
                    gumbels = -torch.empty_like(self.arch_parameters()[1]).exponential_().log()
                    logits = (self.arch_parameters()[1].log_softmax(dim=1) + gumbels) / self.tau
                    probs = nn.functional.softmax(logits, dim=1)
                    index = probs.max(-1, keepdim=True)[1]
                    one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
                    hardwts = one_h - probs.detach() + probs
                    if (torch.isinf(gumbels).any()) or (torch.isinf(probs).any()) or (torch.isnan(probs).any()):
                        continue
                    else:
                        break
            s0, s1 = s1, cell(s0, s1, hardwts, index)
        s1 = self.lastact(s1)
        result = F.upsample(s1, input.size()[-2:], mode='bilinear', scale_factor=None)
        return result

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).to(self.device), requires_grad=True)
        self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops).to(self.device), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    #选出来α，把结构从连续的又变回离散的。
    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

