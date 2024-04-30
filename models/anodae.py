import torch 
import torch.nn as nn
import torch.nn.functional as F

class NodeAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NodeAttention, self).__init__()
        self.emb_conv = nn.Conv1d(input_dim, output_dim, kernel_size=1)
        self.q_conv = nn.Conv1d(output_dim, 1, kernel_size=1)
        self.k_conv = nn.Conv1d(output_dim, 1, kernel_size=1)


    def forward(self, input, adj):
        X = input.transpose(1,2) # (B, hidden1, N)
        seq_fts = self.emb_conv(X) # (B, D, N)

        f_1_t = self.q_conv(seq_fts) #(B, 1, N)
        f_2_t = self.k_conv(seq_fts) #(B ,1, N)

        f = f_1_t + f_2_t.transpose(1,2) # (B,N,N)
        f = F.leaky_relu(adj * f) #(B,N,N)
        coefs = F.softmax(f,dim = 1) 
        vals = torch.matmul(coefs, seq_fts.transpose(1,2))
        ## 是否需要加偏置
        return vals

class GCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()
        self.emb_conv = nn.Conv1d(input_dim, output_dim, kernel_size=1)
        # self.q_conv = nn.Conv1d(output_dim, 1, kernel_size=1)
        # self.k_conv = nn.Conv1d(output_dim, 1, kernel_size=1)

    def forward(self, input, adj):
        X = input.transpose(1,2) # (B, hidden1, N)
        seq_fts = self.emb_conv(X) # (B, D, N)

        # f_1_t = self.q_conv(seq_fts) #(B, 1, N)
        # f_2_t = self.k_conv(seq_fts) #(B ,1, N)

        # f = f_1_t + f_2_t.transpose(1,2) # (B,N,N)
        # f = F.leaky_relu(adj * f) #(B,N,N)
        # coefs = F.softmax(f,dim = 1) 
        vals = torch.matmul(adj, seq_fts.transpose(1,2))
        ## 是否需要加偏置
        return vals

class TCN(nn.Module):
    def __init__(self, input_dim, output_dim, L = 4):
        super(TCN, self).__init__()
        self.layers = nn.ModuleList()  # []会存储在cpu上
        self.L = L
        window=2
        dilation = 1
        self.layers.append(nn.Conv2d(in_channels=input_dim,
                                     out_channels=output_dim,
                                     kernel_size=(1, window),
                                     dilation = dilation))
        for i in range(L):
            self.layers.append(nn.Conv2d(in_channels=output_dim,
                                         out_channels=output_dim,
                                         kernel_size=(1, window),
                                         dilation = dilation))
            dilation *= 2

    def forward(self, x):
        '''x: (B, T, N, d)
        '''
        x = x.transpose(1,3) #(B, d, N, T)
        for i in range(self.L):
            x = self.layers[i](x)
        
        y = x[...,-1].transpose(1,2)

        return y


class AnoDAE(nn.Module):
    def __init__(self, opt):
        super(AnoDAE, self).__init__()
        self.num_nodes = opt['num_adj']
        self.num_features = opt['num_feature']
        self.hidden1 = opt['hidden1']
        self.hidden2 = opt['hidden2']
        self.dropout = opt['dropout']

        self.fea_trans = nn.Sequential(nn.Linear(self.num_features, self.hidden1, bias = True), nn.Tanh())
        self.attr_trans1 = nn.Sequential(nn.Linear(self.num_nodes, self.hidden1, bias = True), nn.Tanh())
        self.attr_trans2 = nn.Sequential(nn.Linear(self.hidden1, self.hidden2, bias = True), nn.Tanh())
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)
        self.gat = NodeAttention(self.hidden1, self.hidden2)


    def forward(self, data, time, A):
        '''
        input data:  (B, T, V, L*D)
        input A: adjacent matrix: (V, V)
        '''
        real_data = data[:,-1]
        hidden1 = self.fea_trans(real_data)
        embedding_n = self.gat(hidden1, A)  # (B, V, H)
        
        hidden2 = self.attr_trans1(real_data.transpose(1,2))
        embedding_a = self.attr_trans2(hidden2) # (B, D, H)
        
        embedding_n, embedding_a = self.dropout1(embedding_n), self.dropout2(embedding_a)

        rec_A = torch.matmul(embedding_n, embedding_n.transpose(1,2))
        rec_real = torch.matmul(embedding_n, embedding_a.transpose(1,2))
        
        return rec_A, rec_real


class AnoDAE1(nn.Module):
    def __init__(self, opt):
        super(AnoDAE1, self).__init__()
        self.num_nodes = opt['num_adj']
        self.num_features = opt['num_feature']
        self.num_layers = opt['num_layers']
        self.hidden1 = opt['hidden1']
        self.hidden2 = opt['hidden2']
        self.dropout = opt['dropout']
        self.device = opt['device']
        self.emb = opt['emb']

        if opt['act'] == 'relu':
            self.act = nn.ReLU()
        elif opt['act'] == 'tanh':
            self.act = nn.Tanh()

        if self.emb =='rnn':
            self.fea_trans = nn.GRU(input_size=self.num_features, 
                                    hidden_size=self.hidden1, 
                                    num_layers=self.num_layers, 
                                    batch_first=True )
            self.attr_trans = nn.GRU(input_size=self.num_nodes, 
                                hidden_size=self.hidden2, 
                                num_layers=self.num_layers, 
                                batch_first=True)
        elif self.emb == 'cnn':
            self.fea_trans = TCN(self.num_features, self.hidden1)  
            self.attr_trans = TCN(self.num_nodes, self.hidden1)

        
        self.linear1 = nn.Sequential(nn.Linear(self.hidden1, self.hidden1), self.act)
        self.linear2 = nn.Sequential(nn.Linear(self.hidden2, self.hidden2), self.act)
        
        self.gat = NodeAttention(self.hidden1, self.hidden2)

        self.linear3 = nn.Linear(self.hidden2, self.num_features)

        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)


    def forward(self, data, time, A):
        '''
        input data:  (B, T, V, D)
        input A: adjacent matrix: (V, V)
        '''
        batch_size = data.shape[0]
        
        if self.emb == 'rnn':
            x1 = torch.cat(torch.split(data, 1, dim=2), dim=0).squeeze(2) # (B*V, T, D)
            h1 = torch.zeros(self.num_layers, x1.shape[0], self.hidden1).to(self.device) # (L, B, H)

            z_fea, x_fea = self.fea_trans(x1, h1) # (num_layer, B*V, H)

            x_fea = x_fea[-1].reshape(batch_size, -1, x_fea.shape[-1])  #(B, V, H)
            x_fea = self.linear1(x_fea)
            x_fea = self.gat(x_fea, A)  # (B, V, H) 

            x2 = torch.cat(torch.split(data, 1, dim=-1), dim=0).squeeze(-1) # (B*D, T, V)
            _, x_attr = self.attr_trans(x2) # (num_layer, B*D, H)
            x_attr = x_attr[-1].reshape(batch_size, -1, x_attr.shape[-1])  #(B, D, H)
            x_attr = self.linear2(x_attr)
        elif self.emb == 'cnn':
            x1 = data
            x_fea = self.fea_trans(x1) # (B, V, D)
            x_fea = self.linear1(x_fea)
            x_fea = self.gat(x_fea, A)  # (B, V, H) 

            x2 = x1.transpose(2,3)   #(B, T, D, V)
            x_attr = self.attr_trans(x2) # (num_layer, B*D, H)
            x_attr = self.linear2(x_attr)
    
        ### (B, V, L*D )

        embedding_n = self.dropout1(x_fea)
        embedding_a = self.dropout2(x_attr)

        rec_A = torch.matmul(embedding_n, embedding_n.transpose(1,2))
        rec_real = torch.matmul(embedding_n, embedding_a.transpose(1,2))
        # rec_real = self.linear3(embedding_n)
        
        return rec_A, rec_real

class AnoDAE2(nn.Module):
    def __init__(self, opt):
        super(AnoDAE2, self).__init__()
        self.num_nodes = opt['num_adj']
        self.num_features = opt['num_feature']
        self.num_layers = opt['num_layers']
        self.hidden1 = opt['hidden1']
        self.hidden2 = opt['hidden2']
        self.dropout = opt['dropout']
        self.device = opt['device']
        self.emb = opt['emb']
        
        if opt['act'] == 'relu':
            self.act = nn.ReLU()
        elif opt['act'] == 'tanh':
            self.act = nn.Tanh()
        
        if self.emb == 'rnn':
            self.fea_trans = nn.GRU(input_size=self.num_features, 
                                    hidden_size=self.hidden1, 
                                    num_layers=self.num_layers, 
                                    batch_first=True)
        elif self.emb == 'cnn':
            self.fea_trans = TCN(self.num_features, self.hidden1)  

        self.linear1 = nn.Sequential(nn.Linear(self.hidden1, self.hidden1), self.act)
        self.linear2 = nn.Sequential(nn.Linear(self.hidden2, self.hidden2), self.act)
        
        self.gat = NodeAttention(self.hidden1, self.hidden2)

        self.linear3 = nn.Linear(self.hidden2, self.num_features)

        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)


    def forward(self, data, time, A):
        '''
        input data:  (B, T, V, D)
        input A: adjacent matrix: (V, V)
        '''
        batch_size = data.shape[0]

        if self.emb == 'rnn':
            x1 = torch.cat(torch.split(data, 1, dim=2), dim=0).squeeze(2) # (B*V, T, D)
            h1 = torch.zeros(self.num_layers, x1.shape[0], self.hidden1).to(self.device) # (L, B, H)

            _, x_fea = self.fea_trans(x1, h1) # (num_layer, B*V, H)

            x_fea = x_fea[-1].reshape(batch_size, -1, x_fea.shape[-1])  #(B, V, H)
            x_fea = self.linear1(x_fea)
            x_fea = self.gat(x_fea, A)  # (B, V, H) 
        elif self.emb == 'cnn':
            x1 = data
            x_fea = self.fea_trans(x1) # (B, V, D)
            x_fea = self.linear1(x_fea)
            x_fea = self.gat(x_fea, A)  # (B, V, H) 
        ### (B, V, H)

        embedding_n = self.dropout1(x_fea)
        rec_A = torch.matmul(embedding_n, embedding_n.transpose(1,2))

        rec_real = self.linear3(embedding_n)
        
        return rec_A, rec_real

class AnoDAE3(nn.Module):
    def __init__(self, opt):
        super(AnoDAE3, self).__init__()
        self.num_nodes = opt['num_adj']
        self.num_features = opt['num_feature']
        self.num_layers = opt['num_layers']
        self.hidden1 = opt['hidden1']
        self.hidden2 = opt['hidden2']
        self.dropout = opt['dropout']
        self.device = opt['device']
        self.emb = opt['emb']
        
        if opt['act'] == 'relu':
            self.act = nn.ReLU()
        elif opt['act'] == 'tanh':
            self.act = nn.Tanh()
        
        if self.emb == 'rnn':
            self.fea_trans = nn.GRU(input_size=self.num_features, 
                                    hidden_size=self.hidden1, 
                                    num_layers=self.num_layers, 
                                    batch_first=True)
        elif self.emb == 'cnn':
            self.fea_trans = TCN(self.num_features, self.hidden1)  
        
        self.linear1 = nn.Sequential(nn.Linear(self.hidden1, self.hidden1), nn.Tanh())
        
        self.linear2 = nn.Sequential(nn.Linear(self.hidden1 + 31, self.hidden2), nn.Tanh())
        
        self.gat = NodeAttention(self.hidden1, self.hidden2)

        self.linear3 = nn.Linear(self.hidden2, self.num_features)

        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)


    def forward(self, data, time, A):
        '''
        input data:  (B, T, V, D)
        input A: adjacent matrix: (V, V)
        input time:  (B, d)
        '''
        batch_size = data.shape[0]

        if self.emb == 'rnn':
            x1 = torch.cat(torch.split(data, 1, dim=2), dim=0).squeeze(2) # (B*V, T, D)
            h1 = torch.zeros(self.num_layers, x1.shape[0], self.hidden1).to(self.device) # (L, B, H)

            _, x_fea = self.fea_trans(x1, h1) # (num_layer, B*V, H)

            x_fea = x_fea[-1].reshape(batch_size, -1, x_fea.shape[-1])  #(B, V, H)
            x_fea = self.linear1(x_fea)
            x_fea = self.gat(x_fea, A)  # (B, V, H) 

        elif self.emb == 'cnn':
            x1 = data
            x_fea = self.fea_trans(x1) # (B, V, D)
            x_fea = self.linear1(x_fea)
            x_fea = self.gat(x_fea, A)  # (B, V, H) 
        

        t_fea = time.unsqueeze(1).repeat(1,x_fea.shape[1],1)
        fea = torch.cat((x_fea, t_fea), axis=-1)  # (B, V, H + d)
        
        fea = self.linear2(fea)  # (B, V, H2)

        ### (B, V, H)

        embedding_n = self.dropout1(fea)
        rec_A = torch.matmul(embedding_n, embedding_n.transpose(1,2))

        rec_real = self.linear3(embedding_n)
        
        return rec_A, rec_real

class AnoDAE4(nn.Module):
    def __init__(self, opt):
        super(AnoDAE4, self).__init__()
        self.num_nodes = opt['num_adj']
        self.num_features = opt['num_feature']
        self.num_layers = opt['num_layers']
        self.hidden1 = opt['hidden1']
        self.hidden2 = opt['hidden2']
        self.dropout = opt['dropout']
        self.device = opt['device']
        # self.type = opt['type']
        self.emb = opt['emb']
        
        if opt['act'] == 'relu':
            self.act = nn.ReLU()
        elif opt['act'] == 'tanh':
            self.act = nn.Tanh()

        if self.emb == 'rnn':
            self.fea_trans = nn.GRU(input_size=self.num_features, 
                                    hidden_size=self.hidden1, 
                                    num_layers=self.num_layers, 
                                    batch_first=True )
        elif self.emb == 'cnn':
            self.fea_trans = TCN(self.num_features, self.hidden1)  
        
        self.linear1 = nn.Sequential(nn.Linear(self.hidden1, self.hidden1), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(self.hidden1 + 31, self.hidden2), nn.ReLU())
        
        self.gat = NodeAttention(self.hidden1, self.hidden2)

        self.linear3 = nn.Linear(self.hidden2, self.num_features)

        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)


    def forward(self, data, time, A):
        '''
        input data:  (B, T, V, D)
        input A: adjacent matrix: (V, V)
        input time:  (B, d)
        '''
        batch_size = data.shape[0]

        if self.emb == 'rnn':
            x1 = torch.cat(torch.split(data, 1, dim=2), dim=0).squeeze(2) # (B*V, T, D)
            h1 = torch.zeros(self.num_layers, x1.shape[0], self.hidden1).to(self.device) # (L, B, H)


            x_seq, x_fea = self.fea_trans(x1, h1) # (B*V, T, H), (num_layer, B*V, H)
            
            x_his = x_seq[:, -2, :]
            x_fea = x_fea[-1].reshape(batch_size, -1, x_fea.shape[-1])  #(B, V, H)
            x_his = x_his.reshape(batch_size, -1, x_fea.shape[-1])

        elif self.emb == 'cnn':
            x1 = data
            x_fea = self.fea_trans(x1) # (B, V, D)
            x_his = self.fea_trans(x1[:,:-1])
        
        
        x_fea = self.linear1(x_fea)
        x_fea = self.gat(x_fea, A)  # (B, V, H) 
        t_fea = time.unsqueeze(1).repeat(1,x_fea.shape[1],1)
        fea = torch.cat((x_fea, t_fea), axis=-1)  # (B, V, H + d)
        fea = self.linear2(fea)  # (B, V, H2)

        ## his representation
        his_fea = self.linear1(x_his)
        his_fea = self.gat(his_fea, A)  # (B, V, H) 
        his_fea = torch.cat((his_fea, t_fea), axis=-1)  # (B, V, H + d)
        his_fea = self.linear2(his_fea)  # (B, V, H2)

        ### (B, V, H)

        embedding_n = self.dropout1(fea)
        rec_A = torch.matmul(embedding_n, embedding_n.transpose(1,2))
        rec_real = self.linear3(embedding_n)
        
        embedding_n_his = self.dropout1(his_fea)
        rec_A_his = torch.matmul(embedding_n_his, embedding_n_his.transpose(1,2))
        
        
        return rec_A, rec_real, rec_A_his

class dominant(nn.Module):
    def __init__(self, opt):
        super(dominant, self).__init__()
        self.num_nodes = opt['num_adj']
        self.num_features = opt['num_feature']
        self.hidden1 = opt['hidden1']
        self.hidden2 = opt['hidden2']
        self.dropout = opt['dropout']

        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)
        # self.gat = NodeAttention(self.hidden1, self.hidden2)
        self.encoder = nn.ModuleList()
        self.encoder.append(GCN(self.num_features, self.hidden1))
        # for i in range(1):
        #     self.encoder.append(GCN(self.hidden1, self.hidden1))
        
        self.decoder = nn.ModuleList()
        # for i in range(1):
        #     self.decoder.append(GCN(self.hidden1, self.hidden1))
        self.decoder.append(GCN(self.hidden1, self.num_features))
        self.act= nn.Tanh()
        # self.norm = nn.BatchNorm1d(self.hidden1)
        self.gra_act = nn.Sigmoid()

    def forward(self, data, time, A):
        '''
        input data:  (B, T, V, L*D)
        input A: adjacent matrix: (V, V)
        '''
        x = data[:,-1]
        for layer in self.encoder:
            x = layer(x, A)  #(B,N,D)
            x = self.act(x)
            # x = self.norm(x)

        embedding_n = x  # (B, N, D)
        rec_A = torch.matmul(embedding_n, embedding_n.transpose(1,2))
        # rec_A = self.gra_act(rec_A)
        
        # embedding_n, embedding_a = self.dropout1(embedding_n), self.dropout2(embedding_a)
        for layer in self.decoder:
            x = layer(x, A)
            x = self.act(x)
            # x = self.norm(x)

        rec_real = x
        
        return rec_A, rec_real