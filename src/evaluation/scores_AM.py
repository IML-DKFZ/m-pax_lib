import torch
from torch import nn
import numpy as np
import shap

"""
Creates the attribution scores computing objects based on the original images
and the latent representations as inputs. Every object has attached methods for 
IG, EG, Kernel SHAP and Deep SHAP based attribution computation.
"""

class Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        _, _, x = self.model(x)
        return x

class scores_AM_Original:
    def __init__(self, model, datamodule, method, out_dim=10, n=5):
        self.model = model
        self.model.eval()
        self.n = n
        self.out_dim = out_dim

        self.method = method

        self.datamodule = datamodule

    def deep_shap(self):
        with torch.no_grad():
            iter_obj = iter(self.datamodule)
            images_test, _ = iter_obj.next()
            images_train, _ = iter_obj.next()

        if self.model.__class__.__name__ == "betaTCVAE_ResNet":
            self.model = Wrapper(self.model)

        exp = shap.DeepExplainer(self.model,
                                 data=images_train  
                                )

        deep_shap_values = exp.shap_values(images_test[:self.n], check_additivity=True)

        return deep_shap_values, images_test[:self.n]

    def expgrad_shap(self):
        with torch.no_grad():
            iter_obj = iter(self.datamodule)
            images_test, _ = iter_obj.next()
            images_train, _ = iter_obj.next()

            height = images_train[0].shape[1]
            width = images_train[0].shape[2]

            if self.method == "IG":
                images_train = torch.zeros((1, 1, height, width))

        if self.model.__class__.__name__ == "betaTCVAE_ResNet":
            self.model = Wrapper(self.model)

        exp = shap.GradientExplainer(self.model,
                                     data=images_train 
                                    )

        expgrad_shap_values = exp.shap_values(images_test[:self.n])

        return expgrad_shap_values, images_test[:self.n]

    def kernel_shap(self):
        with torch.no_grad():
            iter_obj = iter(self.datamodule)
            images_test, _ = iter_obj.next()
            images_train, _ = iter_obj.next()


        if self.model.__class__.__name__ == "betaTCVAE_ResNet":
            self.model = Wrapper(self.model)

        exp = shap.KernelExplainer(self.kernel_model,
                                   data=images_train.detach().numpy()
                                )

        kernel_shap_values = exp.shap_values(images_test[:self.n].detach().numpy(), 
                                            check_additivity=True)

        return kernel_shap_values, images_test[:self.n]

    def kernel_model(self, x):
        x = torch.from_numpy(x)
        x = self.model(x)
        x = x.detach().numpy()
        return x

    def compute(self):
        if self.method == "DeepSHAP":
            values, images_test = self.deep_shap()
        elif self.method == "KernelSHAP":
            values, images_test = self.kernel_shap()
        else:
            values, images_test = self.expgrad_shap()

        return values, images_test

class scores_AM_Latent:
    def __init__(self, model, encoder, datamodule, method):
        self.model = model
        self.model.eval()

        self.encoder = encoder
        self.encoder.eval()


        self.method = method

        self.datamodule = datamodule

    def expgrad_shap(self):
        with torch.no_grad():
            iter_obj = iter(self.datamodule)
            images_test, labels_test = iter_obj.next()
            images_train, _ = iter_obj.next()

            height = images_train[0].shape[1]
            width = images_train[0].shape[2]

            if self.method == "IG":
                images_train = torch.zeros((1, 1, height, width))

            encoding_train, _ = self.encoder.encoder(images_train)
            encoding_test, _ = self.encoder.encoder(images_test)
            
        exp = shap.GradientExplainer(self.model, 
                                     data = encoding_train
                                    )
        
        expgrad_shap_values = exp.shap_values(encoding_test)
        return exp, expgrad_shap_values, encoding_test.numpy().astype('float32'), labels_test

    def deep_shap(self):
        with torch.no_grad():
            iter_obj = iter(self.datamodule)
            images_test, labels_test = iter_obj.next()
            images_train, _ = iter_obj.next()

            encoding_train, _ = self.encoder.encoder(images_train)
            encoding_test, _ = self.encoder.encoder(images_test)

        exp = shap.DeepExplainer(self.model,
                                 data=encoding_train
                                )

        deep_shap_values = exp.shap_values(encoding_test, check_additivity=True)
        return exp, deep_shap_values, encoding_test.numpy().astype('float32'), labels_test

    def kernel_shap(self):
        with torch.no_grad():
            iter_obj = iter(self.datamodule)
            images_test, labels_test = iter_obj.next()
            images_train, _ = iter_obj.next()

            encoding_train, _ = self.encoder.encoder(images_train)
            encoding_test, _ = self.encoder.encoder(images_test)

        exp = shap.KernelExplainer(self.kernel_model,
                                   data=encoding_train.detach().numpy()
                                  )

        kernel_shap_values = exp.shap_values(encoding_test.detach().numpy(), 
                                            check_additivity=True)

        return exp, kernel_shap_values, encoding_test.numpy().astype('float32'), labels_test

    def kernel_model(self, x):
        x = torch.from_numpy(x)
        x = self.model(x)
        x = x.detach().numpy()
        return x

    def compute(self):
        if self.method == "DeepSHAP":
            exp, values, encoding, labels_test = self.deep_shap()
        elif self.method == "KernelSHAP":
            exp, values, encoding, labels_test = self.kernel_shap()
        else:
            exp, values, encoding, labels_test = self.expgrad_shap()

        return exp, values, encoding, labels_test