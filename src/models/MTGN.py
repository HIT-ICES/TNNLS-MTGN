from cmath import log
import re
from importlib_metadata import metadata
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import Sequential, BatchNorm
from src.models.modules.TGCN import TGCNConv
import torch.distributions as D
import torch.nn.functional as F
from src.models.modules.dists import UpperTailTruncatedLogNormalMixtureDistribution, LogNormalMixtureDistribution, TruncatedNormal, MixtureSameFamily
import time
from torchmetrics import RetrievalMRR, MeanAbsoluteError
from src.utils.utils import HITS
import numpy as np
from torch_geometric.utils import to_undirected

class MTGN(LightningModule):
    def __init__(self, num_nodes, embed_size, num_gcn_layers, num_mixture_components, missing_mum, missing_component=True, missing_stru='hypo1', undirected=False,sample_among_shown_entity=[True,False], lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_size = embed_size
        self.num_gcn_layers = num_gcn_layers
        self.num_mixture_components = num_mixture_components
        self.missing_mum = missing_mum
        self.missing_stru = missing_stru
        self.undirected=undirected
        if type(sample_among_shown_entity) == bool:
            self.sample_among_shown_entity = [sample_among_shown_entity, sample_among_shown_entity]
        else:
            self.sample_among_shown_entity = sample_among_shown_entity
        self.lr = lr
        self.weight_decay = weight_decay
        self.tbpp = True
        self.truncated_bptt_steps = 5
        self.missing_component = missing_component
        self.save_hyperparameters()
        
        # static node embedding for observed events
        self.register_parameter('observed_embeddings', torch.nn.Parameter(torch.randn(num_nodes, embed_size)))
        glorot(self.observed_embeddings)
        if self.missing_component:
            # missing node embedding for unobserved events
            self.register_parameter('missing_embeddings', torch.nn.Parameter(torch.randn(num_nodes, embed_size)))
            glorot(self.missing_embeddings)
        
        self.register_buffer('t_l', -torch.ones(num_nodes) * 1.0)
        if self.missing_component:
            self.register_buffer('r_l', -torch.ones(num_nodes) * 1.0)
        self.enlarge_size = 2
        # evolving node embeddings for observed events
        self.register_buffer('observed_hiddens', torch.zeros(num_nodes, embed_size))
        # evolving node embeddings for missing events
        if self.missing_component:
            self.register_buffer('missing_hiddens', torch.zeros(num_nodes, embed_size))
        
        self.mean_absolute_error = MeanAbsoluteError()
        self.mrr = RetrievalMRR()
        self.hit3 = HITS(topk=3)
        self.hit5 = HITS(topk=5)
        self.hit10  = HITS(topk=10)
        
        self._build_layers()
        
    def _build_layers(self):
        observed_layers, missing_layers = [], []
        for layer_idx in range(self.num_gcn_layers):
            observed_layers.append((TGCNConv(self.embed_size, self.embed_size), 'x, edge_index, t -> x'))
            missing_layers.append((TGCNConv(self.embed_size, self.embed_size), 'x, edge_index, t -> x'))
            if layer_idx != self.num_gcn_layers - 1:
                observed_layers.append(nn.ReLU(inplace=True))
                observed_layers.append(nn.Dropout(p=0.5))
                missing_layers.append(nn.ReLU(inplace=True))
                missing_layers.append(nn.Dropout(p=0.5))

        # gnn layers for observed events
        self.observed_gnn_layers = Sequential('x, edge_index, t', observed_layers)
        self.observed_rnn = nn.GRUCell(self.embed_size, self.embed_size)
        
        if self.missing_component:
            # gnn layers for missing events
            self.missing_gnn_layers = Sequential('x, edge_index, t', missing_layers)
            self.missing_rnn = nn.GRUCell(self.embed_size, self.embed_size)
        
        # distribution for observed event entity u
        # p(u | o_t_bar*, m_t*)
        if self.missing_component:
            self.observed_mlp_u = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.embed_size * 2 * self.enlarge_size, self.embed_size * 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_size * 2, self.num_nodes)
            ) # 2: m, o
            
            # distribution for missing event entity v|u
            # p(v | u*, o_t_bar*, m_t*)
            self.observed_mlp_v = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.embed_size * 4 * self.enlarge_size, self.embed_size * 4),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_size * 4, self.num_nodes)
            ) # 4: m_u, o_u, m, o

            # observed event time TPP parameterization
            self.observed_log_norm_mix_param_mlp = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.embed_size * 4, self.embed_size * 4),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_size * 4, self.num_mixture_components * 3) # 4: m_u, o_u, m_v, o_v  3: w, mu, sigma
            )
        else:
            self.observed_mlp_u = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.embed_size * self.enlarge_size, self.embed_size * 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_size * 2, self.num_nodes)
            )
            self.observed_mlp_v = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.embed_size * 2 * self.enlarge_size, self.embed_size * 4),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_size * 4, self.num_nodes)
            ) # 4: o_u, o

            # observed event time TPP parameterization
            self.observed_log_norm_mix_param_mlp = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.embed_size * 2, self.embed_size * 4),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_size * 4, self.num_mixture_components * 3) # 4: m_u, o_u, m_v, o_v  3: w, mu, sigma
            )
        
        
        # prior distribution for missing event entity u
        # p_{prior} (u | o_t_bar, m_t_bar)
        if self.missing_component:
            self.missing_prior_mlp_u = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.embed_size * 2 * self.enlarge_size, self.embed_size * 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_size * 2, self.num_nodes if self.missing_stru=="hypo1" else 1)
            )  # 2: m, o
            # distribution for missing event entity v|u
            self.missing_prior_mlp_v = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.embed_size * 4 * self.enlarge_size, self.embed_size * 4),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_size * 4, self.num_nodes)
            ) # 4: m_u, o_u, m, o
            # missing event time prior TPP parameterization
            self.missing_prior_log_norm_mix_param_mlp = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.embed_size * 4, self.embed_size * 4),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_size * 4, self.num_mixture_components * 3) # 4: m_u, o_u, m_v, o_v  3: w, mu, sigma
            )
            ##### New added: not trainable parameters #####
            '''
            for params in self.missing_prior_mlp_u.parameters():
                params.requires_grad = False
            for params in self.missing_prior_mlp_v.parameters():
                params.requires_grad = False
            for paras in self.missing_prior_log_norm_mix_param_mlp.parameters():
                params.requires_grad = False
            '''
            ###############################################
            
            # posterior distribution for missing events u
            # q_{\psi} (u | o_t, o_t_bar, m_t_bar)
            self.missing_posterior_mlp_u = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.embed_size * 3 * self.enlarge_size, self.embed_size * 3),    # 3: o_t, o_t_bar, m_t_bar
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_size * 3, self.num_nodes if self.missing_stru=="hypo1" else 1)
            )
            # posterior distribution for missing events v|u
            self.missing_posterior_mlp_v = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.embed_size * 6 * self.enlarge_size, self.embed_size * 6),    # 6: o_t, o_t_bar, m_t_bar, u -> each ....
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_size * 6, self.num_nodes)
            )
            # missing event time posterior TPP parameterization
            self.missing_posterior_trunc_log_norm_mix_mlp = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.embed_size * 6, self.embed_size * 6),    # 6: o_t, o_t_bar, m_t_bar, u -> each ....
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_size * 6, self.num_mixture_components * 3)
            )
    
    def tbptt_split_batch(self, batch, split_size: int) -> list:
        splits = []
        timestamps, prev_t = [], -1
        for index, t in enumerate(batch.t):
            if t != prev_t:
                timestamps.append(index)
            prev_t = t
        num_split = len(timestamps) // split_size
        for i in range(1, num_split):
            splits.append(batch[timestamps[(i-1)*split_size]:timestamps[i*split_size]])
        if (len(timestamps) % split_size) != 0:
            splits.append(batch[timestamps[num_split*split_size]:])
        return splits
    
    def sample_missing_events(self, O_t, o_active_t, O_t_bar, o_active_t_bar, M_t_bar, m_active_t_bar, delta_t_max, Q, shown_entities=None):
        """sample missing events from the posterior distribution and calculate the KL-divergence between the posterior and the prior distribution

        Args:
            O_t (Tensor): [N, D]
            o_active_t (Tensor): [N,]
            O_t_bar (Tensor): [N, D]
            o_active_t_bar (Tensor): [N,]
            M_t (Tensor): [N, D]
            m_active_t (Tensor): [N,]
            delta_t_max (float): t + tau - delta --> sampled time
            Q (torch.Size): Number of missing events to sample
        """
        # generate graph level embeddings
        O_t_star = torch.cat([O_t, self.observed_embeddings], dim=1)
        o_t = torch.max(O_t_star, dim=0)[0] if torch.all(o_active_t == -1) else torch.max(O_t_star[o_active_t > -1], dim=0)[0]
        
        O_t_bar_star = torch.cat([O_t_bar, self.observed_embeddings], dim=1)
        o_t_bar = torch.max(O_t_bar_star, dim=0)[0] if torch.all(o_active_t_bar == -1) else torch.max(O_t_bar_star[o_active_t_bar > -1], dim=0)[0]

        M_t_bar_star = torch.cat([M_t_bar, self.missing_embeddings], dim=1)
        m_t_bar = torch.max(M_t_bar_star, dim=0)[0] if torch.all(m_active_t_bar == -1) else torch.max(M_t_bar_star[m_active_t_bar > -1], dim=0)[0]
        #o_t, o_t_bar, m_t_bar = torch.max(O_t[o_active_t > -1], dim=0)[0], torch.max(O_t_bar[o_active_t_bar > -1], dim=0)[0], torch.max(M_t_bar[m_active_t_bar > -1], dim=0)[0]
        o_t, o_t_bar, m_t_bar = o_t.view(1, -1), o_t_bar.view(1, -1), m_t_bar.view(1, -1)
        
        if self.sample_among_shown_entity[0]:
            mask = torch.where(o_active_t > -1, torch.zeros_like(self.t_l), torch.ones_like(self.t_l) * float('-inf'))

        # distribution for missing event entity u (posterior and prior)
        if self.missing_stru == "hypo1":
            posterior_u_score = self.missing_posterior_mlp_u(torch.cat([o_t, o_t_bar, m_t_bar], dim=-1)).view(-1) # [N]
        else:
            posterior_u_score = self.missing_posterior_mlp_u(torch.cat([O_t_star, O_t_bar_star, M_t_bar_star], dim=-1)).view(-1) # [N]
        #######
        if self.sample_among_shown_entity[0]:
            posterior_u_score = posterior_u_score + mask
        #######
        posterior_probs = torch.softmax(posterior_u_score, dim=-1)
        
        
        posterior_u_cat_dist = D.Categorical(probs=posterior_probs)
        
        if self.missing_stru == "hypo1":
            prior_u_score = self.missing_prior_mlp_u(torch.cat([o_t_bar, m_t_bar], dim=-1)).view(-1) # [N]
        else:
            prior_u_score = self.missing_prior_mlp_u(torch.cat([O_t_bar_star, M_t_bar_star], dim=-1))
            prior_u_score = prior_u_score.view(-1) # [N]
        
        ########
        if self.sample_among_shown_entity[1]:
            prior_u_score = prior_u_score + mask
        ########
        prior_probs = torch.softmax(prior_u_score, dim=-1)
        
        prior_u_cat_dist = D.Categorical(probs=prior_probs)
        
        kl_div_u = D.kl_divergence(posterior_u_cat_dist, prior_u_cat_dist) # [1,]
        
        # 1. sample entity u from the posterior distribution
        sample_entity_u = posterior_u_cat_dist.sample(Q)
        sample_entity_u = sample_entity_u.view(-1) # [Q]
        
        o_u_t, o_u_t_bar, m_u_t_bar = O_t_star[sample_entity_u], O_t_bar_star[sample_entity_u], M_t_bar_star[sample_entity_u]
        o_t_b, o_t_bar_b, m_t_bar_b = o_t.repeat(sample_entity_u.size(0), 1), o_t_bar.repeat(sample_entity_u.size(0), 1), m_t_bar.repeat(sample_entity_u.size(0), 1)
        
        # distribution for missing event entity v|u (posterior and prior)
        posterior_v_score = self.missing_posterior_mlp_v(torch.cat([o_u_t, o_u_t_bar, m_u_t_bar, o_t_b, o_t_bar_b, m_t_bar_b], dim=-1)) # [Q, N]
        prior_v_score = self.missing_prior_mlp_v(torch.cat([o_u_t_bar, m_u_t_bar, o_t_bar_b, m_t_bar_b], dim=-1)) 
        
        #
        if self.sample_among_shown_entity[0]:
            posterior_v_score = posterior_v_score + mask.repeat(sample_entity_u.size(0), 1)
        #
        posterior_probs = torch.softmax(posterior_v_score, dim=-1)
    
        posterior_v_cat_dist = D.Categorical(probs=posterior_probs)
       
        
        
            
        #
        if self.sample_among_shown_entity[1]:
            prior_v_score = prior_v_score + mask.repeat(sample_entity_u.size(0), 1)
        #
        prior_probs = torch.softmax(prior_v_score, dim=-1)
        
        prior_v_cat_dist = D.Categorical(probs=prior_probs)
        
        kl_div_v = D.kl_divergence(posterior_v_cat_dist, prior_v_cat_dist)  # (Q,)
        
        # 2. sample entity v from the posterior distribution
        sample_entity_v = posterior_v_cat_dist.sample()
        sample_entity_v = sample_entity_v.view(-1) #[Q]
        
        # Temporal dont need to cat embeddings.
        o_u_t, o_u_t_bar, m_u_t_bar = O_t[sample_entity_u], O_t_bar[sample_entity_u], M_t_bar[sample_entity_u]
        o_v_t, o_v_t_bar, m_v_t_bar = O_t[sample_entity_u], O_t_bar[sample_entity_u], M_t_bar[sample_entity_u]

        # distribution for missing event \Delta|u, v
        posterior_tpp_params = self.missing_posterior_trunc_log_norm_mix_mlp(torch.cat([o_u_t, o_u_t_bar, m_u_t_bar, o_v_t, o_v_t_bar, m_v_t_bar], dim=-1)) #[Q, 3K]
        posterior_log_w, posterior_loc, posterior_log_scale = posterior_tpp_params.chunk(3, 1)  # [Q, K]
        ##clamp the log_scale
        posterior_log_scale = torch.clamp(posterior_log_scale, min=-5, max=5)
        ##
        posterior_loc = F.softplus(posterior_loc)
        posterior_log_w = F.softplus(posterior_log_w)
        posterior_log_w = torch.softmax(posterior_log_w, dim=-1)
        ##########
        
        mix = D.Categorical(probs=posterior_log_w)
        componet_dist = TruncatedNormal(posterior_loc, posterior_log_scale.exp(), b=torch.log(delta_t_max))
        componet_dist = D.TransformedDistribution(componet_dist, D.ExpTransform())
        #posterior_tpp_dist = D.MixtureSameFamily(mix, componet_dist)
        posterior_tpp_dist = MixtureSameFamily(mix, componet_dist)
        ##########
        
        prior_tpp_params = self.missing_prior_log_norm_mix_param_mlp(torch.cat([o_u_t_bar, m_u_t_bar, o_v_t_bar, m_v_t_bar], dim=-1)) #[Q, 3K]
        prior_log_w, prior_loc, prior_log_scale = prior_tpp_params.chunk(3, 1)  # [Q, K]
        ##
        prior_log_scale = torch.clamp(prior_log_scale, min=-5, max=5)
        ##
        prior_loc = F.softplus(prior_loc)
        prior_log_w = torch.softmax(prior_log_w, dim=-1)
        prior_log_w = F.softplus(prior_log_w)
        prior_tpp_dist = LogNormalMixtureDistribution(prior_loc, prior_log_scale, prior_log_w)
        
        
        kl_div_delta = D.kl_divergence(posterior_tpp_dist, prior_tpp_dist) # [Q,]
        
        # 3. sample \Delta from the posterior distribution
        sample_delta = posterior_tpp_dist.sample()
        
        #return kl_div_u * sample_entity_u.size(0), kl_div_v.sum(), kl_div_delta.sum(), sample_entity_u, sample_entity_v, sample_delta
        #return kl_div_u, kl_div_v.mean(), kl_div_u, sample_entity_u, sample_entity_v, sample_delta
        return kl_div_u, kl_div_v.mean(), kl_div_delta.mean(), sample_entity_u, sample_entity_v, sample_delta
    
    def observed_events_log_prob(self, edge_index, edge_t):
        observed_star = torch.cat([self.observed_hiddens, self.observed_embeddings], dim=1)
        o = torch.max(observed_star, dim=0)[0] if torch.all(self.t_l == -1) else torch.max(observed_star[self.t_l > -1], dim=0)[0]

        if self.missing_component:
            missing_star = torch.cat([self.missing_hiddens, self.missing_embeddings], dim=1)
            m = torch.max(missing_star, dim=0)[0] if torch.all(self.r_l == -1) else torch.max(missing_star[self.r_l > -1], dim=0)[0]
            o, m = o.view(1, -1), m.view(1, -1)
        else:
            o = o.view(1, -1)
        # distribution for observed event entity u
        if self.missing_component:
            u_score = self.observed_mlp_u(torch.cat([o, m], dim=-1)).view(-1) # [N]
        else:
            u_score = self.observed_mlp_u(o).view(-1) # [N]
            
        u_probs = torch.softmax(u_score, dim=-1)
        u_log_probs = torch.log(u_probs[edge_index[0]] + 1e-14).view(-1) # avoid 0 #[E]
        
        # distribution for observed event entity v|u 
        if self.missing_component:
            ob, mb = o.repeat(edge_t.size(0), 1), m.repeat(edge_t.size(0), 1)
            o_u, m_u = observed_star[edge_index[0]], missing_star[edge_index[0]]
            v_score = self.observed_mlp_v(torch.cat([o_u, m_u, ob, mb], dim=-1)) # [B, N]
        else:
            ob = o.repeat(edge_t.size(0), 1)
            o_u = observed_star[edge_index[0]]
            v_score = self.observed_mlp_v(torch.cat([o_u, ob], dim=-1)) # [B, N]
            
        v_probs = torch.softmax(v_score, dim=-1) # [B, N]
        v_log_probs = torch.log(v_probs.gather(1, edge_index[1].view(-1, 1)) + 1e-14).view(-1) # [E]
        
        if self.missing_component:
            o_u, m_u = self.observed_hiddens[edge_index[0]], self.missing_hiddens[edge_index[0]]
            o_v, m_v = self.observed_hiddens[edge_index[1]], self.missing_hiddens[edge_index[1]]
        else:
            o_u, o_v = self.observed_hiddens[edge_index[0]], self.observed_hiddens[edge_index[1]]
        
        # distribution for observed event \Delta|u, v
        if self.missing_component:
            tpp_params = self.observed_log_norm_mix_param_mlp(torch.cat([o_u, m_u, o_v, m_v], dim=-1)) # [E, 3K]
        else:
            tpp_params = self.observed_log_norm_mix_param_mlp(torch.cat([o_u, o_v], dim=-1)) # [E, 3K]
        log_w, loc, log_scale = tpp_params.chunk(3, 1)  # [E, K]
        
        ##
        log_scale = torch.clamp(log_scale,min=-5, max=5)
        ##
        loc = F.softplus(loc)
        log_w = torch.softmax(log_w, dim=-1)
        log_w = F.softplus(log_w)
        tpp_dist = LogNormalMixtureDistribution(loc, log_scale, log_w)
        t_log_probs = tpp_dist.log_prob(edge_t.view(-1, 1)).diag().view(-1) # [E]
        return u_log_probs, v_log_probs, t_log_probs
        
        
        

    def on_train_epoch_start(self) -> None:
        zeros(self.observed_hiddens)
        if self.missing_component:
            zeros(self.missing_hiddens)
        self.t_l = -1.0 * torch.ones_like(self.t_l).type_as(self.t_l)
        if self.missing_component:
            self.r_l = -1.0 * torch.ones_like(self.r_l).type_as(self.r_l)
        self.train_escape_time = 0
    
    def training_step(self, batch, batch_idx, hiddens=None):
        start_time = time.time()
        edge_src_batch, edge_dst_batch, t_batch = batch.src, batch.dst, batch.t
        edge_index_batch = torch.stack([edge_src_batch, edge_dst_batch])
        time_stamps = torch.unique(t_batch, sorted=True)
        # operate one time stemp
        loss = 0
        for t in time_stamps:
            loss_t = self.train_temporal_step(edge_index_batch, t_batch, t)
            loss += loss_t
        self.train_escape_time += time.time() - start_time
        self.log('train/loss', loss)
        self.log('train/time', self.train_escape_time, on_step=False, on_epoch=True)
        if self.tbpp:
            return {"loss": loss, "hiddens": None}  # hiddens are maintained in the model
        else:
            return loss
        

    def train_temporal_step(self, edge_index_batch, t_batch, t):
        temporal_mask = t_batch == t
        edge_index, edge_abs_t = edge_index_batch[:,temporal_mask], t_batch[temporal_mask]
        edge_index, edge_abs_t = to_undirected(edge_index, edge_abs_t, reduce='mean')
        # calculate the \delta = t - \bar{t}
        delta_t = t - torch.max(self.t_l)
        # tau = t - max(t_l_u, t_l_v)
        edge_t = torch.minimum(edge_abs_t - self.t_l[edge_index[0]], edge_abs_t - self.t_l[edge_index[1]])
        
        observered_embeddings_L = self.observed_gnn_layers(self.observed_embeddings, edge_index, edge_t)
        observered_hidden_t = self.observed_rnn(observered_embeddings_L, self.observed_hiddens)
        # TODO: design do we need to concat the observed hidden state with the observed embeddings?
        o_active_t = self.t_l * 1. # avoid change t_l value
        o_active_t[edge_index[0]] = t # update the observed time stamp
        if self.missing_component:
            Q = max(1, int(self.missing_mum * edge_t.size(0)))
            kl_div_u, kl_div_v, kl_div_delta, sample_entity_u, sample_entity_v, sample_delta = self.sample_missing_events(
                observered_hidden_t, o_active_t, self.observed_hiddens, self.t_l, self.missing_hiddens, self.r_l, delta_t, (Q,))
            self.log("train/kl_div_u", kl_div_u, on_step=True)
            self.log("train/kl_div_v", kl_div_v, on_step=True)
            self.log("train/kl_div_delta", kl_div_delta, on_step=True)
            self.log("train/kl_div_total", kl_div_u + kl_div_v + kl_div_delta, on_step=True)
        
            # generate missing event graph
            missing_entity_u = torch.cat([sample_entity_u, sample_entity_v], dim=0)
            missing_entity_v = torch.cat([sample_entity_v, sample_entity_u], dim=0)
            missing_edge_index = torch.stack([missing_entity_u, missing_entity_v])
            missing_edge_abs_t = t + delta_t - torch.cat([sample_delta, sample_delta], dim=0)
            missing_edge_t = torch.minimum(missing_edge_abs_t - self.r_l[missing_entity_u], missing_edge_abs_t - self.r_l[missing_entity_v])
        
            # update the missing time stamp and missing hiddens
            missing_embeddings_L = self.missing_gnn_layers(self.missing_embeddings, missing_edge_index, missing_edge_t)
            self.missing_hiddens = self.missing_rnn(missing_embeddings_L, self.missing_hiddens).detach()
            self.r_l[missing_entity_u] = missing_edge_abs_t

        # update the observed time stamp and observed hiddens
        self.observed_hiddens = observered_hidden_t.detach()
        
        u_log_probs, v_log_probs, t_log_probs = self.observed_events_log_prob(edge_index, edge_t) 
        self.t_l[edge_index[0]] = t
        self.log("train/u_log_probs", u_log_probs.mean(), on_step=True)
        self.log("train/v_log_probs", v_log_probs.mean(), on_step=True)
        self.log("train/t_log_probs", t_log_probs.mean(), on_step=True)
        
        # calculate the loss
        if self.missing_component:
            loss = -(torch.mean(u_log_probs) + torch.mean(v_log_probs) + torch.mean(t_log_probs)) + (kl_div_u + kl_div_v + kl_div_delta)
        else:
            loss = -(torch.mean(u_log_probs) + torch.mean(v_log_probs) + torch.mean(t_log_probs))
        return loss
    
    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
                else:
                    torch.nn.utils.clip_grad_norm_(param, 0.5)
        if not valid_gradients:
            print("NaN or Inf gradients detected. not updating parameters")
            self.optimizers().param_groups[1]['lr'] = self.optimizers.param_groups[1]['lr'] * 0.5
            
            self.optimizers().zero_grad()
    
     
    def on_validation_epoch_start(self) -> None:
        self.val_escape_time = 0
        
    def validation_step(self, batch, batch_idx):
        start_time = time.time()
        edge_src_batch, edge_dst_batch, edge_abs_t = batch.src, batch.dst, batch.t
        edge_index = torch.stack([edge_src_batch, edge_dst_batch])
        if self.undirected:
            edge_index, edge_abs_t = to_undirected(edge_index, edge_abs_t, reduce='mean')
        #edge_index, edge_abs_t = batch.edge_index, batch.t
        delta_t = torch.minimum(edge_abs_t - self.t_l[edge_index[0]], edge_abs_t - self.t_l[edge_index[1]])
        source, target = edge_index[0], edge_index[1]
        
        observed_star = torch.cat([self.observed_hiddens, self.observed_embeddings], dim=1)
                        

        o = torch.max(observed_star, dim=0)[0] if torch.all(self.t_l == -1) else torch.max(observed_star[self.t_l > -1], dim=0)[0]
        if self.missing_component:
            missing_star = torch.cat([self.missing_hiddens, self.missing_embeddings], dim=1)
            m = torch.max(missing_star, dim=0)[0] if torch.all(self.r_l == -1) else torch.max(missing_star[self.r_l > -1], dim=0)[0]
            o, m = o.view(1, -1), m.view(1, -1) 
            ob, mb = o.repeat(source.size(0), 1), m.repeat(source.size(0), 1)
            o_u, m_u = observed_star[edge_index[0]], missing_star[edge_index[0]]
            predict_v_score = self.observed_mlp_v(torch.cat([o_u, m_u, ob, mb], dim=-1)) # [E, N]
        else:
            o = o.view(1, -1)
            ob = o.repeat(source.size(0), 1)
            o_u = observed_star[edge_index[0]]
            predict_v_score = self.observed_mlp_v(torch.cat([o_u, ob], dim=-1)) # [E, N]
        
        if self.missing_component:
            o_u, m_u = self.observed_hiddens[source], self.missing_hiddens[source]
            o_v, m_v = self.observed_hiddens[target], self.missing_hiddens[target]
        else:
            o_u, o_v = self.observed_hiddens[source], self.observed_hiddens[target]
        
        if self.missing_component:
            tpp_params = self.observed_log_norm_mix_param_mlp(torch.cat([o_u, m_u, o_v, m_v], dim=-1)) # [E, 3K]
        else:
            tpp_params = self.observed_log_norm_mix_param_mlp(torch.cat([o_u, o_v], dim=-1)) # [E, 3K]
        log_w, loc, log_scale = tpp_params.chunk(3, 1)  # [E, K]
        ##
        log_scale = torch.clamp(log_scale, min=-5, max=5)
        
        ##
        loc = F.softplus(loc)
        log_w = torch.softmax(log_w, dim=-1)
        log_w = F.softplus(log_w)
        tpp_dist = LogNormalMixtureDistribution(loc, log_scale, log_w)
        predict_delta_t = tpp_dist.mean
        predict_delta_t = predict_delta_t.view(-1)
        
        
        self.val_escape_time += time.time() - start_time
        self.log("val/time", self.val_escape_time, on_step=False, on_epoch=True)
        ############# Metrics update#############
        # not sandity check
        
        self.mean_absolute_error.update(predict_delta_t, delta_t)
        #num_events = predict_v_score.size(0)
        #indexes = torch.arange(num_events).view(-1, 1).repeat(1, predict_v_score.size(1)).type_as(predict_v_score).long()
        #logits = torch.zeros_like(predict_v_score).scatter(1, target.view(-1, 1), 1)
        #logits = logits.type_as(predict_v_score).long()
        #self.mrr.update(predict_v_score, logits, indexes)
        
        self.hit3.update(predict_v_score, target.view(-1, 1))
        self.hit5.update(predict_v_score, target.view(-1, 1))
        self.hit10.update(predict_v_score, target.view(-1, 1))
        ############ Metrics log ################
        self.log("val/mae", self.mean_absolute_error.compute(), on_step=False, on_epoch=True)
        #self.log("val/mrr", self.mrr.compute(), on_step=False, on_epoch=True)
        self.log("val/hit3", self.hit3.compute(), on_step=False, on_epoch=True)
        self.log("val/hit5", self.hit5.compute(), on_step=False, on_epoch=True)
        self.log("val/hit10", self.hit10.compute(), on_step=False, on_epoch=True)    
    
        
    def configure_optimizers(self):
        if self.missing_component: 
            ob_structure_mlp_params = list(map(id, self.observed_mlp_u.parameters())) + list(map(id, self.observed_mlp_v.parameters())) + list(map(id, self.missing_posterior_mlp_u.parameters())) + list(map(id, self.missing_posterior_mlp_v.parameters())) + list(map(id, self.missing_prior_mlp_u.parameters())) + list(map(id, self.missing_prior_mlp_v.parameters()))
        else:
            ob_structure_mlp_params =  list(map(id, self.observed_mlp_u.parameters())) + list(map(id, self.observed_mlp_v.parameters()))
            
        base_params = filter(lambda p: id(p) in ob_structure_mlp_params, self.parameters())
        other_params = filter(lambda p: id(p) not in ob_structure_mlp_params, self.parameters())
        params = [
            {"params": base_params, "lr": 5e-4}, # 5e-4
            {"params": other_params, "lr": self.lr},
        ]
        #opt = torch.optim.SGD(params, momentum=0.9, weight_decay=5e-4)
        opt = torch.optim.AdamW(params, weight_decay=self.weight_decay)
        #lambda1 = lambda epoch: 1e-3
        #lambda2 = lambda epoch: 5e-4 if epoch < 70 else 1e-5
        #scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=[lambda1, lambda2])
        #return [opt], [scheduler]
        return opt