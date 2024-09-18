import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from tqdm.auto import tqdm


class Generator(nn.Module):
    
    def __init__(self, feature_size, hidden_size, latent_size, num_layers, num_nodes=196, replicate_latent=False):
        super(Generator, self).__init__()
        self.num_layers = num_layers
        self.fc = nn.Linear(latent_size, num_nodes * latent_size)

        self.conv_layers = nn.ModuleList([GATConv(latent_size, hidden_size)])
        for _ in range(self.num_layers - 1):
                self.conv_layers.append(GATConv(hidden_size, hidden_size))
        self.conv_layers.append(GATConv(hidden_size, feature_size))

        self.latent_size = latent_size
        self.num_nodes = num_nodes
        self.replicate_latent = replicate_latent

    def forward(self, z, edge_index):
        
        if self.replicate_latent:
            x = z.unsqueeze(1).expand(-1, self.num_nodes, -1)
            x = x.reshape(z.size(0) * self.num_nodes, self.latent_size)
        else:
            x = F.relu(self.fc(z))
            x = x.view(z.size(0) * self.num_nodes, self.latent_size)

        for i, layer in enumerate(self.conv_layers):
            x = layer(x, edge_index)
            
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
            
        return x
    
    
class Encoder(nn.Module):
    
    def __init__(self, feature_size, hidden_size, latent_size, num_layers, dropout_p=0.2):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        
        self.conv_layers = nn.ModuleList([GATConv(feature_size, hidden_size)])
        for _ in range(self.num_layers - 1):
            self.conv_layers.append(GATConv(hidden_size, hidden_size))

        self.fc = nn.Linear(hidden_size, latent_size)
        self.dropout_p = dropout_p
    
    def forward(self, x, edge_index, batch):
        
        for layer in self.conv_layers:
            x = F.relu(layer(x, edge_index))

        x = F.dropout(x, self.dropout_p, training=self.training)
        x = global_mean_pool(x, batch)
        z = self.fc(x)

        return z
    
    
class Discriminator(nn.Module):
    
    def __init__(self, feature_size, hidden_size, latent_size, num_layers, dropout_p=0.2):
        super(Discriminator, self).__init__()
        self.num_layers = num_layers
        
        self.layers_x = nn.ModuleList([GATConv(feature_size, hidden_size)])
        for _ in range(self.num_layers - 1):
            self.layers_x.append(GATConv(hidden_size, hidden_size))

        self.layers_z = nn.ModuleList([nn.Linear(latent_size, hidden_size)])
        for _ in range(self.num_layers - 1):
                self.layers_z.append(nn.Linear(hidden_size, hidden_size))
  
        self.layers_xz = nn.ModuleList([nn.Linear(2 * hidden_size, hidden_size)])
        for _ in range(self.num_layers - 1):
                self.layers_xz.append(nn.Linear(hidden_size, hidden_size))
        self.layers_xz.append(nn.Linear(hidden_size, 1))
        
        self.dropout_p = dropout_p
        
    def inf_x(self, x, edge_index, batch):
        
        for layer in self.layers_x:
            x = F.relu(layer(x, edge_index))
        
        x = F.dropout(x, self.dropout_p, training=self.training)
        x = global_mean_pool(x, batch)

        return x
        
    def inf_z(self, z):
        
        for layer in self.layers_z:
            z = F.relu(layer(z))
        
        return z
    
    def inf_xz(self, xz):
        
        for i, layer in enumerate(self.layers_xz):
            if i < len(self.layers_xz) - 1:
                xz = F.relu(layer(xz))
            else:
                pred = layer(xz)

        return pred
    
    def forward(self, x, z, edge_index, batch):
        x = self.inf_x(x, edge_index, batch)
        z = self.inf_z(z)
        xz = torch.cat((x, z), dim=1)
        pred = self.inf_xz(xz)
        return pred


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


class Model:
    
    def __init__(self, args, device='cpu'):
        self.args = args
        self.device = device
        self.build_model()
    
    def build_model(self):
        self.netG = Generator(self.args.feature_size, 
                              self.args.hidden_size, 
                              self.args.latent_size, 
                              self.args.num_layers,
                              self.args.num_nodes, 
                              self.args.replicate_latent).to(self.device)        
        self.netE = Encoder(self.args.feature_size, 
                            self.args.hidden_size, 
                            self.args.latent_size, 
                            self.args.num_layers,
                            self.args.dropout_p).to(self.device)
        self.netD = Discriminator(self.args.feature_size, 
                                  self.args.hidden_size, 
                                  self.args.latent_size, 
                                  self.args.num_layers, 
                                  self.args.dropout_p).to(self.device)

        self.netG = self.netG.apply(weights_init)
        self.netE = self.netE.apply(weights_init)
        self.netD = self.netD.apply(weights_init)
        
    def train(self, train_loader):

        optimizerGE = optim.Adam(list(self.netG.parameters()) + 
                                 list(self.netE.parameters()), 
                                 lr = self.args.lr_GE, betas=(0.5, 0.999))
        optimizerD = optim.Adam(self.netD.parameters(), 
                                lr = self.args.lr_D, betas=(0.5, 0.999))        
        criterion = nn.BCEWithLogitsLoss()
        loss_history = {'D':[], 'GE':[]}

        for epoch in range(self.args.num_epochs):
            D_losses = []; GE_losses = []

            for data in tqdm(train_loader, desc=f'Training... Epoch {epoch}'):
                data = data.to(self.device)
                x, edge_index, batch = data.x, data.edge_index, data.batch

                # Update Discriminator
                for _ in range(self.args.repeats_D):
                    optimizerD.zero_grad()
                    mean_iteration_D_loss = 0

                    z = torch.randn((len(data), self.args.latent_size), device=self.device)
                    x_gen = self.netG(z, edge_index)
                    z_gen = self.netE(x, edge_index, batch)
                    D_fake_pred = self.netD(x_gen.detach(), z, edge_index, batch)
                    D_real_pred = self.netD(x, z_gen.detach(), edge_index, batch)

                    D_loss = criterion(D_real_pred, torch.ones_like(D_real_pred)) + criterion(D_fake_pred, torch.zeros_like(D_fake_pred))

                    D_loss.backward()
                    optimizerD.step()

                    mean_iteration_D_loss += D_loss.item() / self.args.repeats_D

                # Update Generator and Encoder
                for _ in range(self.args.repeats_GE):
                    optimizerGE.zero_grad()
                    mean_iteration_GE_loss = 0

                    z = torch.randn((len(data), self.args.latent_size), device=self.device)
                    x_gen = self.netG(z, edge_index)
                    z_gen = self.netE(x, edge_index, batch)
                    D_fake_pred = self.netD(x_gen, z, edge_index, batch)
                    D_real_pred = self.netD(x, z_gen, edge_index, batch)
                    x_recon = self.netG(self.netE(x, edge_index, batch), edge_index)
                    z_recon = self.netE(self.netG(z, edge_index), edge_index, batch)

                    consistency_loss = F.l1_loss(x_recon, x, reduction='sum') + F.l1_loss(z_recon, z, reduction='sum')
                    GE_loss = (criterion(D_real_pred, torch.zeros_like(D_real_pred)) + criterion(D_fake_pred, torch.ones_like(D_fake_pred))) + self.args.cc_weight * consistency_loss

                    GE_loss.backward()
                    optimizerGE.step()

                    mean_iteration_GE_loss += GE_loss.item() / self.args.repeats_GE

                D_losses.append(mean_iteration_D_loss)
                GE_losses.append(mean_iteration_GE_loss)

            loss_history['D'].append(np.mean(D_losses))
            loss_history['GE'].append(np.mean(GE_losses))

            if self.args.verbose:
                print(f'''Epoch {epoch} -- Discrimiantors Loss: {loss_history['D'][-1]:.3f}, Generator-Encoder Loss: {loss_history['GE'][-1]:.3f}''') 

        return loss_history

    def test(self, test_data, model_save_path, error_weights=torch.tensor([1., 1., 1.])):
        
        if self.args.pretrained:
            self.load_weights(model_save_path)

        self.netG.eval(); self.netE.eval(); self.netD.eval()
        errors = []; true_speeds = []; recon_speeds = []
        
        for data in test_data:
            data = data.to(self.device)
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            x_recon = self.netG(self.netE(x, edge_index, batch), edge_index)
            error = torch.mean(error_weights * ((x_recon.detach() - x) ** 2), dim=1)
            
            errors.append(error)
            true_speeds.append(x[:, 1])
            recon_speeds.append(x_recon[:, 1].detach())
        
        return torch.stack(errors).numpy(), torch.stack(true_speeds).numpy(), torch.stack(recon_speeds).numpy()
    
    def save_weights(self, model_save_path):
        state_dict_netG = self.netG.state_dict()
        state_dict_netE = self.netE.state_dict()
        state_dict_netD = self.netD.state_dict()
        torch.save({'Generator': state_dict_netG,
                    'Encoder': state_dict_netE,
                    'Discriminator': state_dict_netD}, model_save_path + '/model_parameters.pth')

    def load_weights(self, model_save_path):
        state_dict = torch.load(model_save_path + '/model_parameters.pth')
        self.netG.load_state_dict(state_dict['Generator'])
        self.netE.load_state_dict(state_dict['Encoder'])
        self.netD.load_state_dict(state_dict['Discriminator'])

