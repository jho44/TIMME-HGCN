from model.layer import GraphConvolution
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import model.manifolds as manifolds
import math
from model.attention import ScaledDotProductSelfAttention
from model.decoders import LinearDecoder

import torch

from utils import slicing
import numpy as np

class WeightClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'c'):
            c = module.c.data
            c = c.clamp(min=0.1)
        if hasattr(module, 'c_in'):
            c = module.c_in.data
            c = c.clamp(min=0.1)
        if hasattr(module, 'linear') and hasattr(module.linear, 'c'):
            c = module.linear.c.data
            c = c.clamp(min=0.1)
        if hasattr(module, 'agg') and hasattr(module.agg, 'c'):
            c = module.agg.c.data
            c = c.clamp(min=0.1)
        if hasattr(module, 'hyp_act') and hasattr(module.hyp_act, 'c_in'):
            c = module.hyp_act.c_in.data
            c = c.clamp(min=0.1)
        if hasattr(module, '0'):
            if hasattr(getattr(module,'0'), 'linear') and hasattr(getattr(module,'0').linear, 'c'):
                c = getattr(module,'0').linear.c.data
                c = c.clamp(min=0.1)
            if hasattr(getattr(module,'0'), 'agg') and hasattr(getattr(module,'0').agg, 'c'):
                c = getattr(module,'0').agg.c.data
                c = c.clamp(min=0.1)
            if hasattr(getattr(module,'0'), 'hyp_act') and hasattr(getattr(module,'0').hyp_act, 'c_in'):
                c = getattr(module,'0').hyp_act.c_in.data
                c = c.clamp(min=0.1)
        if hasattr(module, 'layers'):
            if hasattr(module.layers, '0'):
                if hasattr(getattr(module.layers,'0'), 'linear') and hasattr(getattr(module.layers,'0').linear, 'c'):
                    c = getattr(module.layers,'0').linear.c.data
                    c = c.clamp(min=0.1)
                if hasattr(getattr(module.layers,'0'), 'agg') and hasattr(getattr(module.layers,'0').agg, 'c'):
                    c = getattr(module.layers,'0').agg.c.data
                    c = c.clamp(min=0.1)
                if hasattr(getattr(module.layers,'0'), 'hyp_act') and hasattr(getattr(module.layers,'0').hyp_act, 'c_in'):
                    c = getattr(module.layers,'0').hyp_act.c_in.data
                    c = c.clamp(min=0.1)

clipper = WeightClipper()

def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))

    n_curvatures = args.num_layers - 1
    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([0.2])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures


class DenseAtt(nn.Module):
    def __init__(self, in_features, dropout):
        super(DenseAtt, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(2 * in_features, 1, bias=True)
        self.in_features = in_features

    def forward (self, x, adj):
        n = x.size(0)
        # n x 1 x d
        x_left = torch.unsqueeze(x, 1)
        x_left = x_left.expand(-1, n, -1)
        # 1 x n x d
        x_right = torch.unsqueeze(x, 0)
        x_right = x_right.expand(n, -1, -1)

        x_cat = torch.cat((x_left, x_right), dim=2)
        att_adj = self.linear(x_cat).squeeze()
        att_adj = F.sigmoid(att_adj)

        att_adj = torch.mul(adj.to_dense(), att_adj)
        return att_adj

class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)

        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )

class HypAgg(nn.Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg, num_adjs):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

        self.adj_weight = nn.Parameter(nn.init.kaiming_uniform_(torch.FloatTensor(num_adjs, in_features, 16))) # TODO: pass out_features in less heathen way
        self.bias = nn.Parameter(torch.FloatTensor(16)) # TODO: pass args.bias and args.dim in

    def forward(self, x, adj):
        # to tangent space
        outputs = []
        x_tangent = self.manifold.logmap0(x, c=self.c)

        for i in range(len(self.adj_weight)):
            support_t = torch.mm(x_tangent, self.adj_weight[i])
            support_t = torch.spmm(adj[i], support_t)
            output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
            outputs.append(output)

        outputs_raw = torch.stack(outputs)
        outputs = torch.stack(outputs, 2)

        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att, x_tangent)
        else:
            outputs /= 11 # TODO: num_adjs

        output = torch.sum(outputs, 2)
        return output + self.bias

    def extra_repr(self):
        return 'c={}'.format(self.c)

class HypAct(nn.Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )

class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg, 11) # pass num_adjs through in a less heathen way
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        # print("AFTER LINEAR")
        # print(h)
        h = self.agg.forward(h, adj) # want neighbors from diff relations to be aggregated properly
        # print("AFTER AGG")
        # print(h)
        h = self.hyp_act.forward(h)
        output = h, adj
        # print("AFTER ACT")
        # print(h)
        return output

class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        # print(output)
        return output

class HGCN(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, c, args):
        super(HGCN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        args.num_layers = args.num_layers
        assert args.num_layers > 1
        dims, acts, self.curvatures = get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, self.out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    HyperbolicGraphConvolution(
                            self.manifold, in_dim, self.out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att, args.local_agg
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        # clipper = WeightClipper()
        # self.apply(clipper)

        # print("ACTUAL CURVATURE")
        # print(self.curvatures[0])
        # if self.curvatures[0] <= 0.1:
        #     print("LESS THAN")
        #     for param in self.parameters():
        #         param.requires_grad = False
        #     self.curvatures[0] = torch.tensor([0.01])
        # print("HGCN ENCODE PROJ")
        # print(x)
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        # print("TAN")
        # print(x_tan)
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        # print("EXP")
        # print(x_hyp)
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        # print("HYP PROJ")
        # print(x_hyp)
        return super(HGCN, self).encode(x_hyp, adj)

class GCN_multirelation(nn.Module):
    """
    The multi-relational encoder of TIMME
    """
    def __init__(self, num_relation, num_entities, num_adjs, nfeat, nhid, dropout, skip_mode="none", attention_mode="none"):
        super(GCN_multirelation, self).__init__()

        self.gc1 = GraphConvolution(num_relation, num_entities, num_adjs, nfeat, nhid, attention_mode=attention_mode)
        self.gc2 = GraphConvolution(num_relation, num_entities, num_adjs, self.gc1.out_features, nhid, attention_mode=attention_mode)
        self.dropout = dropout
        if skip_mode not in ["add", "concat", "none"]:
            print("skip mode {} unknown, use default option 'none'".format(skip_mode))
            skip_mode = "add"
        elif skip_mode in ["concat"]:
            self.ff = nn.Linear(self.gc1.out_features + self.gc2.out_features, self.gc2.out_features)
        self.skip_mode = skip_mode
        self.out_dim = self.gc2.out_features

    def skip_connect_out(self, x2, x1):
        return self.ff(torch.cat((x2, x1), 1)) if self.skip_mode=="concat" else x2+x1

    def forward(self, x, adjs):
        x1 = F.relu(self.gc1(x, adjs))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, adjs))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        return x2 if self.skip_mode == "none" else self.skip_connect_out(x2, x1)

class Classification(nn.Module):
    def __init__(self, num_relation, num_entities, num_adjs, nfeat, nhid, nclass, dropout, regularization=None, gcn=None, skip_mode="none", attention_mode="none", trainable_features=None):
        super(Classification, self).__init__()
        self.gcn = GCN_multirelation(num_relation, num_entities, num_adjs, nfeat, nhid, dropout, skip_mode=skip_mode, attention_mode=attention_mode) if gcn is None else gcn
        self.classifier = nn.Linear(self.gcn.out_dim, nclass)
        self.reg_param = regularization if regularization else 0
        self.trainable_features = trainable_features if trainable_features else None

    def forward(self, x, adjs, calc_gcn=True):
        x = self.gcn(x, adjs) if calc_gcn else x
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x

    def regularization_loss(self, embedding):
        if not self.reg_param:
            return 0
        return self.reg_param * torch.mean(embedding.pow(2))

    def get_loss(self, output, labels, idx_lst):
        reg_loss = self.regularization_loss(output) # regularize the embeddings
        return F.nll_loss(output[idx_lst], labels[idx_lst]) + reg_loss

class LinkPrediction(nn.Module):
    def __init__(self, num_relation, gcn=None, relations=None, regularization=None, weightless=False, add_layer=True, trainable_features=None):
        super(LinkPrediction, self).__init__()

        self.gcn = gcn
        self.trainable_features = trainable_features if trainable_features else None
        if add_layer:
            self.additional_layer = nn.Linear(self.gcn.out_dim, self.gcn.out_dim)
        else:
            self.register_parameter('additional_layer', None)
        self.reg_param = regularization if regularization else 0
        self.num_relation = num_relation
        self.relation_names = relations if relations else [""] * num_relation
        # relations to predict using weight: could be 1 ~ N relations when we use DistMult
        # each relation's embedding is trained differently anyway
        self.n_relations = num_relation
        self.w_relation = nn.Parameter(torch.Tensor(num_relation, self.gcn.out_dim), requires_grad=(not weightless))
        self.w_standard = nn.Parameter(torch.Tensor(num_relation, self.gcn.out_dim * 2), requires_grad=(not weightless))
        self.bias = nn.Parameter(torch.Tensor(num_relation,1), requires_grad=(not weightless))
        # initialization wouldn't affect if it is trainable or not
        # relations have to be somewhat different from each other to make a difference
        nn.init.xavier_uniform_(self.w_relation,
                            gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_standard,
                            gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.bias,
                            gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        '''
        NTN with diag-weight and k=1
        Called TIMME-NTN for convenience in our paper
        '''
        # tensor layer with k = 1 and w being diagonal
        s = embedding[triplets[0]]
        r = self.w_relation[triplets[1]]
        o = embedding[triplets[2]]
        # standard layer
        v = self.w_standard[triplets[1]]
        c = torch.cat([s,o], dim=1)  # concatenation
        # bias term
        b = self.bias[triplets[1]]
        # final score
        score = torch.sum(s * r * o, dim=1) + torch.sum(v * c, dim=1) + torch.sum(b, dim=1)
        return score

    def forward(self, x, adjs, calc_gcn=True):
        '''
        forward without calculating loss
        '''
        # print("X")
        # print(x)
        # print(x.shape)
        # print("ADJS")
        # print(adjs[0])
        # print(len(adjs))
        # print(adjs[0].shape)
        embeddings = self.gcn.encode(x, adjs) if calc_gcn else x
        # print("EMBEDDINGS")
        # print(embeddings)
        if self.additional_layer:
            embeddings = self.additional_layer(embeddings)
        return embeddings

    def regularization_loss(self, embedding):
        if not self.reg_param:
            return 0
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, embeddings, labels, triplets):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets consists of [source, relation, destination]
        # embeddings = self.forward(x, adjs)
        score = self.calc_score(embeddings, triplets)
        reg_loss = self.regularization_loss(embeddings)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        loss = predict_loss + self.reg_param * reg_loss
        return loss

    def calc_score_by_relation(self, batches, embeddings, cuda=False):
        '''
        batches is a batch generator, from sampler
        see examples in training and testing script
        embeddings is the embedding result of the nodes
        '''
        all_scores = [list() for i in range(self.num_relation)]
        all_labels = [list() for i in range(self.num_relation)]
        for batch_id, triplets, labels, relation_indexes, _ in batches:
            triplets = torch.from_numpy(triplets)
            if cuda:
                triplets = triplets.cuda()
            scores = self.calc_score(embeddings, triplets).detach().cpu().numpy()
            # print(slicing(triplets.numpy().transpose(), relation_indexes[1]))
            for r in range(self.n_relations):
                score_r = slicing(scores, relation_indexes[r])
                label_r = slicing(labels, relation_indexes[r])
                all_scores[r].append(score_r)
                all_labels[r].append(label_r)
        # get the scores of different relation and their labels
        all_scores = [np.concatenate(scores_r) for scores_r in all_scores]
        all_labels = [np.concatenate(labels_r) for labels_r in all_labels]
        return all_scores, all_labels

class TIMME(nn.Module):
    def __init__(self, num_relation, num_entities, num_adjs, nfeat, nhid, nclass, dropout, relations, args, regularization=None, skip_mode="none", attention_mode="none",trainable_features=None):
        super(TIMME, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([0.2]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        self.nnodes = args.n_nodes
        self.gcn = HGCN(self.c, args)
        self.trainable_features = trainable_features if trainable_features else None
        # the last model is always node classification, following the R relations samples
        self.models = nn.ModuleList(list())
        self.num_relation = num_relation
        self.relation_names = relations
        # treat each relation separately
        for i in range(num_relation):
            self.models.append(LinkPrediction(1, gcn=self.gcn, regularization=regularization))
        self.models.append(Classification(num_relation, num_entities, num_adjs, nfeat, nhid, nclass, dropout, gcn=self.gcn))

    def encode(self, x, adjs):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.gcn.encode(x, adjs)
        return [m(h, adjs, calc_gcn=False) for m in self.models]

    def forward(self, x, adjs):
        gcn_embedding = self.gcn(x, adjs)
        return [m(gcn_embedding, adjs, calc_gcn=False) for m in self.models]

    def calc_joint_loss(self, embeddings, losses):
        # no lambda here
        return sum(losses)

    def get_loss(self, embeddings, labels, triplets, mask_info, class_index, class_labels):
        link_loss = [self.models[i].get_loss(embeddings[i], labels[i], triplets[i]) for i in range(self.num_relation)]
        mask_idxs = set(np.concatenate(np.array(np.concatenate(mask_info))))
        valid_idxs = set(class_index.tolist()).intersection(mask_idxs)
        valid_idx_idxs = [i for i,idx in enumerate(class_index.tolist()) if idx in valid_idxs]
        class_index = class_index[valid_idx_idxs]
        node_loss = self.models[-1].get_loss(embeddings[-1], class_labels, class_index)
        # calculate the loss
        return self.calc_joint_loss(embeddings[:-1], link_loss + [node_loss])

    def calc_score_by_relation(self, batches, embeddings, cuda=False, get_triplets=False):
        '''
        batches is a batch generator, from sampler
        see examples in training and testing script
        embeddings is the embedding result of the nodes
        '''
        all_scores = [list() for i in range(self.num_relation)]
        all_labels = [list() for i in range(self.num_relation)]
        all_from = [list() for i in range(self.num_relation)] if get_triplets else None
        all_to = [list() for i in range(self.num_relation)] if get_triplets else None
        for batch_id, triplets, labels, relation_indexes, _ in batches:
            if cuda:
                triplets = [torch.from_numpy(t).cuda(0) for t in triplets]
            else:
                triplets = [torch.from_numpy(t) for t in triplets]
            scores = [self.models[i].calc_score(embeddings[i], triplets[i]).detach().cpu().numpy() for i in range(self.num_relation)]
            for r in range(self.num_relation):
                all_scores[r].append(scores[r][:])
                all_labels[r].append(labels[r][:])
                if get_triplets:
                    all_from[r].extend(list(triplets[r][0].numpy()))
                    all_to[r].extend(list(triplets[r][2].numpy()))
        # get the scores of different relation and their labels
        all_scores = [np.concatenate(scores_r) for scores_r in all_scores]
        all_labels = [np.concatenate(labels_r) for labels_r in all_labels]
        all_triplets = (all_from, all_to)
        return all_scores, all_labels, all_triplets

class TIMMEhierarchical(TIMME):
    def __init__(self, num_relation, num_entities, num_adjs, nfeat, nhid, nclass, dropout, relations, args, regularization=None, skip_mode="none", attention_mode="none", trainable_features=None):
        super(TIMMEhierarchical, self).__init__(num_relation, num_entities, num_adjs, nfeat, nhid, nclass, dropout, relations, args, regularization=regularization, skip_mode=skip_mode, attention_mode=attention_mode,trainable_features=trainable_features)
        self.decoder = LinearDecoder(self.c, args)
        self._lambda = ScaledDotProductSelfAttention(nhid, num_entities)
        self.attention_weight = None

    def forward(self, x, adjs):
        gcn_embedding = self.gcn.encode(x, adjs)
        # print("GCN EMBEDDING")
        # print(gcn_embedding[0])
        # print(gcn_embedding.shape)
        link_embeddings = [m(gcn_embedding, adjs, calc_gcn=False) for m in self.models[:-1]]
        # print("LINK EMBEDDINGS")
        # print(link_embeddings)
        attention_weight = self._lambda(torch.stack(link_embeddings))
        node_x_in = torch.sum(attention_weight * torch.stack(link_embeddings, 2), 2)
        node_embedding = self.models[-1](node_x_in, adjs, calc_gcn=False)
        self.attention_weight = attention_weight.detach().numpy()
        return link_embeddings + [node_embedding]

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output[idx], dim=1)

    def acc_f1(output, labels, average='binary'):
        from sklearn.metrics import accuracy_score, f1_score

        preds = output.max(1)[1].type_as(labels)
        accuracy = accuracy_score(preds, labels)
        f1 = f1_score(preds, labels, average=average)
        return accuracy, f1

    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        acc, f1 = self.acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]