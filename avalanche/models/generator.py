################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 03-03-2022                                                             #
# Author: Florian Mies                                                         #
# Website: https://github.com/travela                                          #
################################################################################

"""

File to place any kind of generative models 
and their respective helper functions.

"""

from abc import abstractmethod
from typing import Union
from matplotlib import transforms
import torch
import torch.nn as nn
from torchvision import transforms
from avalanche.models.utils import MLP, Flatten
from avalanche.models.base_model import BaseModel


class Generator(BaseModel):
    """
    A base abstract class for generators
    """

    @abstractmethod
    def generate(self, batch_size=None, condition=None):
        """
        Lets the generator sample random samples.
        Output is either a single sample or, if provided,
        a batch of samples of size "batch_size"

        :param batch_size: Number of samples to generate
        :param condition: Possible condition for a condotional generator
                          (e.g. a class label)
        """


###########################
# VARIATIONAL AUTOENCODER #
###########################


class VAEMLPEncoder(nn.Module):
    """
    Encoder part of the VAE, computer the latent represenations of the input.

    :param shape: Shape of the input to the network: (channels, height, width)
    :param latent_dim: Dimension of last hidden layer
    """

    def __init__(self, shape=None, latent_dim=128):
        super(VAEMLPEncoder, self).__init__()
        # flattened_size = torch.Size(shape).numel()
        flattened_size = (16*32*8) + 3
        print(flattened_size)
        self.encode = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=flattened_size, out_features=400),
            nn.BatchNorm1d(400),
            nn.LeakyReLU(),
            MLP([400, latent_dim]),
        )

    def forward(self, x, y=None):
        x = self.encode(x)
        return x


class VAEMLPDecoder(nn.Module):
    """
    Decoder part of the VAE. Reverses Encoder.

    :param shape: Shape of output: (channels, height, width).
    :param nhid: Dimension of input.
    """

    def __init__(self, shape, nhid=16):
        super(VAEMLPDecoder, self).__init__()
        flattened_size = torch.Size(shape).numel()

        self.shape = shape
        # self.decode = nn.Sequential(
        #     MLP([nhid, 64, 128, 256, flattened_size], last_activation=False),
        #     nn.Sigmoid(),
        # )
        self.decode = nn.Sequential(
            MLP([nhid, 64, 128, 256, flattened_size], last_activation=False),
            nn.Identity(),
        )
        self.invTrans = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])

    def forward(self, z, y=None):
        if y is None:
            # return self.invTrans(self.decode(z).view(-1, *self.shape))
            return self.decode(z).view(-1, *self.shape)

        else:
            # return self.invTrans(
            #     self.decode(torch.cat((z, y), dim=1)).view(-1, *self.shape)
            # )
            return self.decode(torch.cat((z, y), dim=1)).view(-1, *self.shape)
            

class MlpVAE_RAADL(Generator, nn.Module):
    """
    Variational autoencoder module:
    fully-connected and suited for any input shape and type.

    The encoder only computes the latent represenations
    and we have then two possible output heads:
    One for the usual output distribution and one for classification.
    The latter is an extension the conventional VAE and incorporates
    a classifier into the network.
    More details can be found in: https://arxiv.org/abs/1809.10635
    """

    def __init__(
        # self, shape, nhid=16, n_classes=10, device: Union[str, torch.device] = "cpu"
        self, shape, nhid=16, output_dim=1, device: Union[str, torch.device] = "cpu"
    ):
        """
        :param shape: Shape of each input sample
        :param nhid: Dimension of latent space of Encoder.
        :param n_classes: Number of classes -
                        defines classification head's dimension
        """
        super(MlpVAE_RAADL, self).__init__()
        self.dim = nhid
        if device is None:
            device = "cpu"

        self.device = torch.device(device)
        self.encoder = VAEMLPEncoder(shape, latent_dim=128)
        self.calc_mean = MLP([128, nhid], last_activation=False)
        self.calc_logvar = MLP([128, nhid], last_activation=False)
        # self.classification = MLP([128, n_classes], last_activation=False)
        self.regression_head = MLP([128, output_dim], last_activation=False)
        self.flightparams_head = nn.Sequential(
            nn.Linear(nhid, 64),  # Example layer
            nn.ReLU(),
            nn.Linear(64, 3)  # Assuming 3 flight parameters to generate
        )
        self.decoder = VAEMLPDecoder(shape, nhid)

    def get_features(self, x):
        """
        Get features for encoder part given input x
        """
        return self.encoder(x)

    def generate(self, batch_size=None):
        """
        Generate random samples.
        Output is either a single sample if batch_size=None,
        else it is a batch of samples of size "batch_size".
        """
        z = (
            torch.randn((batch_size, self.dim)).to(self.device)
            if batch_size
            else torch.randn((1, self.dim)).to(self.device)
        )
        
        res = self.decoder(z)
        generated_flightparams = self.generate_flightparams(z)

        if not batch_size:
            res = res.squeeze(0)
            generated_flightparams = generated_flightparams.squeeze(0)
        
        return res, generated_flightparams
    
    def generate_flightparams(self, z):
        """
        Generate flight parameters from latent variables.

        Args:
            z (Tensor): Latent variables.

        Returns:
            Tensor: The generated flight parameters.
        """
        return self.flightparams_head(z)

    def sampling(self, mean, logvar):
        """
        VAE 'reparametrization trick'
        """
        eps = torch.randn(mean.shape).to(self.device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, voxel_tns5, flightparams_tns2):
        """
        Forward.
        """
        flightparams_tns2 = self.norm_flightparams(flightparams_tns2)  # Normalize flight parameters

        # Concatenate flight parameters with encoder output
        combined_features = torch.cat((encoder_output, flightparams_tns2), dim=1)
        
        represntations = self.encoder(voxel_tns5)
        mean, logvar = self.calc_mean(represntations), self.calc_logvar(represntations)
        z = self.sampling(mean, logvar)
        
        reconstructed_voxel_data = self.decoder(z)
        generated_flightparams = self.generate_flightparams(z)
        # res = self.generate()
                
        return reconstructed_voxel_data, generated_flightparams, mean, logvar

class MlpVAE(Generator, nn.Module):
    """
    Variational autoencoder module:
    fully-connected and suited for any input shape and type.

    The encoder only computes the latent represenations
    and we have then two possible output heads:
    One for the usual output distribution and one for classification.
    The latter is an extension the conventional VAE and incorporates
    a classifier into the network.
    More details can be found in: https://arxiv.org/abs/1809.10635
    """

    def __init__(
        self, shape, nhid=16, n_classes=10, device: Union[str, torch.device] = "cpu"
    ):
        """
        :param shape: Shape of each input sample
        :param nhid: Dimension of latent space of Encoder.
        :param n_classes: Number of classes -
                        defines classification head's dimension
        """
        super(MlpVAE, self).__init__()
        self.dim = nhid
        if device is None:
            device = "cpu"

        self.device = torch.device(device)
        self.encoder = VAEMLPEncoder(shape, latent_dim=128)
        self.calc_mean = MLP([128, nhid], last_activation=False)
        self.calc_logvar = MLP([128, nhid], last_activation=False)
        self.classification = MLP([128, n_classes], last_activation=False)
        self.decoder = VAEMLPDecoder(shape, nhid)

    def get_features(self, x):
        """
        Get features for encoder part given input x
        """
        return self.encoder(x)

    def generate(self, batch_size=None):
        """
        Generate random samples.
        Output is either a single sample if batch_size=None,
        else it is a batch of samples of size "batch_size".
        """
        z = (
            torch.randn((batch_size, self.dim)).to(self.device)
            if batch_size
            else torch.randn((1, self.dim)).to(self.device)
        )
        res = self.decoder(z)
        if not batch_size:
            res = res.squeeze(0)
        return res

    def sampling(self, mean, logvar):
        """
        VAE 'reparametrization trick'
        """
        eps = torch.randn(mean.shape).to(self.device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x):
        """
        Forward.
        """
        represntations = self.encoder(x)
        mean, logvar = self.calc_mean(represntations), self.calc_logvar(represntations)
        z = self.sampling(mean, logvar)
        return self.decoder(z), mean, logvar


# Loss functions
BCE_loss = nn.BCELoss(reduction="sum")
MSE_loss = nn.MSELoss(reduction="sum")
CE_loss = nn.CrossEntropyLoss()


def VAE_loss(X, forward_output):
    """
    Loss function of a VAE using mean squared error for reconstruction loss.
    This is the criterion for VAE training loop.

    :param X: Original input batch.
    :param forward_output: Return value of a VAE.forward() call.
                Triplet consisting of (X_hat, mean. logvar), ie.
                (Reconstructed input after subsequent Encoder and Decoder,
                mean of the VAE output distribution,
                logvar of the VAE output distribution)
    """
    # X_hat, mean, logvar = forward_output
    X_hat, mean, logvar = forward_output
    reconstruction_loss = MSE_loss(X_hat, X)
    # regression_loss = MSE_loss(regression_output, y)
    KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean**2)
    return reconstruction_loss + KL_divergence

def VAE_loss_RAADL(voxel_tns5, flightparams_tns2, forward_output):
    """
    Loss function of a VAE using mean squared error for reconstruction loss.
    This is the criterion for VAE training loop.

    :param X: Original input batch.
    :param forward_output: Return value of a VAE.forward() call.
                Triplet consisting of (X_hat, mean. logvar), ie.
                (Reconstructed input after subsequent Encoder and Decoder,
                mean of the VAE output distribution,
                logvar of the VAE output distribution)
    """
    # X_hat, mean, logvar = forward_output
    X_hat, fltparam_hat, mean, logvar = forward_output
    print('Xhat shape', X_hat.shape)
    print('voxel shape', voxel_tns5.shape)

    reconstruction_loss = MSE_loss(X_hat, voxel_tns5)
    fltparam_loss = MSE_loss(fltparam_hat, flightparams_tns2)
    
    # regression_loss = MSE_loss(regression_output, y)
    KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean**2)
    return reconstruction_loss + KL_divergence


__all__ = ["MlpVAE", "MlpVAE_RAADL", "VAE_loss", "VAE_loss_RAADL"]