import abc
from fedrec.preprocessor import DLRMPreprocessor
import sys

import numpy as np
import torch
from fedrec.utilities import registry
from torch import nn
from torch.nn.parameter import Parameter


def xavier_init(layer: nn.Linear):
    # initialize the weights
    # with torch.no_grad():
    # custom Xavier input, output or two-sided fill
    mean = 0.0
    std_dev = np.sqrt(2 / (layer.out_features + layer.in_features))
    W = np.random.normal(mean, std_dev, size=(
        layer.out_features, layer.in_features)).astype(np.float32)
    std_dev = np.sqrt(1 / layer.out_features)  # np.sqrt(2 / (m + 1))
    bt = np.random.normal(
        mean, std_dev, size=layer.out_features).astype(np.float32)
    # approach 1
    layer.weight.data = torch.tensor(W, requires_grad=True)
    layer.bias.data = torch.tensor(bt, requires_grad=True)
    return layer
### define dlrm in PyTorch ###


class BaseModel(abc.ABC):
    @abc.abstractstaticmethod
    def parse_args(args):
        pass


@registry.load("model", "dlrm")
@BaseModel.register
class DLRM_Net(nn.Module):
    Preproc = DLRMPreprocessor

    def create_mlp(self, ln, sigmoid_layer):
        # build MLP layer by layer
        layers = []
        for in_f, out_f in zip(ln, ln[1:]):
            layers += [xavier_init(nn.Linear(in_f, out_f, True)),
                       registry.load('sigmoid_layer', sigmoid_layer)]
        return nn.ModuleList(layers)

    def create_emb(self, m, ln, weighted_pooling=None):
        emb_l = nn.ModuleList()
        v_W_l = []
        for i in range(0, ln.size):
            # construct embedding operator
            registry.construct("embedding",
                               num_embeddings=ln[i],
                               embedding_dim=m, **dict)
            if self.qr_flag and ln[i] > self.qr_threshold:
                EE = QREmbeddingBag(
                    ln[i],
                    m,
                    **dict
                )
            elif self.md_flag and ln[i] > self.md_threshold:
                EE = PrEmbeddingBag(ln[i], m[i], max(m))
            else:
                EE = nn.EmbeddingBag(ln[i], m, **dict)

            if weighted_pooling is None:
                v_W_l.append(None)
            else:
                v_W_l.append(torch.ones(ln[i], dtype=torch.float32))
            emb_l.append(EE)
        return emb_l, v_W_l

    def __init__(
        self,
        preproc : DLRMPreprocessor,
        arch_sparse_feature_size=None,
        ln_emb=None,
        arch_mlp_bot=None,
        arch_mlp_top=None,
        arch_interaction_op=None,
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        sync_dense_params=True,
        loss_weights=None,
        loss_threshold=0.0,
        ndevices=-1,
        qr_dict={},
        md_dict={},
        weighted_pooling=None,
        loss_function="bce"
    ):
        super(DLRM_Net, self).__init__()
        self.preproc = preproc

        if (
            (arch_sparse_feature_size is not None)
            and (ln_emb is not None)
            and (arch_mlp_bot is not None)
            and (arch_mlp_top is not None)
            and (arch_interaction_op is not None)
        ):

            # save arguments
            self.ndevices = ndevices
            self.m_spa = arch_sparse_feature_size
            self.ln_emb = self.preproc.ln_emb
            self.ln_bot = np.fromstring(arch_mlp_bot, dtype=int, sep="-")
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            self.loss_function = loss_function

            # create variables for QR embedding if applicable
            self.qr_dict = qr_dict
            self.md_dict = md_dict

            ### parse command line arguments ###
            num_fea = self.ln_emb.size + 1  # num sparse + num dense features
            m_den = self.preproc.m_den
            self.ln_bot[0] = m_den
            if arch_interaction_op == "dot":
                if arch_interaction_itself:
                    num_int = (num_fea * (num_fea + 1)) // 2 + self.ln_bot[-1]
                else:
                    num_int = (num_fea * (num_fea - 1)) // 2 + self.ln_bot[-1]
            elif arch_interaction_op == "cat":
                num_int = num_fea * self.ln_bot[-1]
            else:
                sys.exit(
                    "ERROR: --arch-interaction-op="
                    + arch_interaction_op
                    + " is not supported"
                )
            self.ln_top = np.fromstring(str(num_int) + "-" + arch_mlp_top,
                                        dtype=int, sep="-")
            self.sanity_check()

            # create operators
            self.emb_l, w_list = self.create_emb(
                arch_sparse_feature_size, ln_emb, weighted_pooling)

            if weighted_pooling is not None and weighted_pooling != "fixed":
                self.weighted_pooling = "learned"
                self.v_W_l = nn.ParameterList()
                for w in w_list:
                    self.v_W_l.append(Parameter(w))
            else:
                self.weighted_pooling = weighted_pooling
                self.v_W_l = w_list

            self.bot_l = self.create_mlp(self.ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(arch_mlp_top, sigmoid_top)

            # specify the loss function
            if self.loss_function == "mse":
                self.loss_fn = torch.nn.MSELoss(reduction="mean")
            elif self.loss_function == "bce":
                self.loss_fn = torch.nn.BCELoss(reduction="mean")
            elif self.loss_function == "wbce":
                self.loss_ws = torch.tensor(
                    np.fromstring(loss_weights, dtype=float, sep="-")
                )
                self.loss_fn = torch.nn.BCELoss(reduction="none")
            else:
                sys.exit(
                    "ERROR: --loss-function=" + self.loss_function + " is not supported"
                )

    def toGPU(self):
        # Custom Model-Data Parallel
        # the mlps are replicated and use data parallelism, while
        # the embeddings are distributed and use model parallelism
        if self.ndevices > 1:
            self.emb_l, self.v_W_l = self.create_emb(
                self.m_spa, self.ln_emb, self.weighted_pooling
            )
        else:
            if self.weighted_pooling == "fixed":
                for k, w in enumerate(self.v_W_l):
                    self.v_W_l[k] = w.cuda()

    def sanity_check(self):
        # sanity check: feature sizes and mlp dimensions must match
        if self.qr_dict['qr_flag']:
            if self.qr_dict['qr_operation'] == "concat" and 2 * self.m_spa != self.ln_bot[-1]:
                sys.exit(
                    "ERROR: 2 arch-sparse-feature-size "
                    + str(2 * self.m_spa)
                    + " does not match last dim of bottom mlp "
                    + str(self.ln_bot[-1])
                    + " (note that the last dim of bottom mlp must be 2x the embedding dim)"
                )
            if self.qr_dict['qr_operation'] != "concat" and self.m_spa != self.ln_bot[-1]:
                sys.exit(
                    "ERROR: arch-sparse-feature-size "
                    + str(self.m_spa)
                    + " does not match last dim of bottom mlp "
                    + str(self.ln_bot[-1])
                )
        else:
            if self.m_spa != self.ln_bot[-1]:
                sys.exit(
                    "ERROR: arch-sparse-feature-size "
                    + str(self.m_spa)
                    + " does not match last dim of bottom mlp "
                    + str(self.ln_bot[-1])
                )

    def apply_mlp(self, x, layers):
        return layers(x)

    def apply_emb(self, lS_o, lS_i, emb_l, v_W_l):
        ly = [None]*len(lS_i)
        for i, (emb, lsi, lso, vwl) in enumerate(zip(emb_l, lS_i, lS_o, v_W_l)):
            per_sample_weights = vwl.gather(
                0, lsi) if vwl is not None else None
            ly[i] = emb(
                lsi,
                lso,
                per_sample_weights=per_sample_weights,
            )
        return ly

    def interact_features(self, x, ly):

        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            _, ni, nj = Z.shape
            offset = 1 if self.arch_interaction_itself else 0
            li = torch.tensor([i for i in range(ni)
                              for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj)
                              for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R

    def forward(self, dense_x, lS_o, lS_i):
        # process dense features (using bottom mlp), resulting in a row vector
        x = self.apply_mlp(dense_x, self.bot_l)
        ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l)
        z = self.interact_features(x, ly)
        out = self.apply_mlp(z, self.top_l)
        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            out = torch.clamp(out, min=self.loss_threshold,
                              max=(1.0 - self.loss_threshold))
        return out
    
    def loss(self, output, true_label):
        if self.loss_function == "mse" or self.loss_function == "bce":
            return self.loss_fn(output, true_label)
        elif self.loss_function == "wbce":
            loss_ws_ = self.loss_ws[true_label.data.view(-1).long()].view_as(true_label)
            loss_fn_ = self.loss_fn(output, true_label)
            loss_sc_ = loss_ws_ * loss_fn_
            return loss_sc_.mean()
