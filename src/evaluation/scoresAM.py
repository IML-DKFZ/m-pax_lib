import torch
import numpy as np
import shap

"""
Creates the attribution scores computing objects based on the original images
and the latent representations as inputs. Every object has attached methods for 
IG, EG, Kernel SHAP and Deep SHAP based attribution computation.
"""

class scores_AM_Original:
    def __init__(self, model, datamodule, type, method, out_dim=10, n=5):
        self.model = model
        self.model.eval()
        self.n = n
        self.out_dim = out_dim

        self.method = method

        self.datamodule = datamodule
        self.type = type

    def deep_shap(self):
        with torch.no_grad():
            iter_obj = iter(self.datamodule)
            images_test, labels_test = iter_obj.next()
            images_train, labels_train = iter_obj.next()

            height = images_train[0].shape[1]
            width = images_train[0].shape[2]

            if self.type == "betaVAE" or self.type == "betaTCVAE":
                images_train = images_train.view(-1, height * width)
                images_test = images_test.view(-1, height * width)

        exp = shap.DeepExplainer(self.model,
                                 data=images_train  
                                )

        deep_shap_values = exp.shap_values(images_test[:self.n], check_additivity=True)

        if self.type == 'betaVAE' or self.type == 'betaTCVAE':
            deep_shap_values = np.asarray(deep_shap_values).reshape(
                                    self.out_dim, self.n, height, width)
            images_test = images_test.view(-1, 1, height, width)

        return deep_shap_values, images_test[:self.n]

    def expgrad_shap(self):
        with torch.no_grad():
            iter_obj = iter(self.datamodule)
            images_test, labels_test = iter_obj.next()
            images_train, labels_train = iter_obj.next()

            height = images_train[0].shape[1]
            width = images_train[0].shape[2]

            if self.method == "IG":
                images_train = torch.zeros((1, 1, height, width))

            if self.type == "betaVAE" or self.type == "betaTCVAE":
                images_train = images_train.view(-1, height * width)
                images_test = images_test.view(-1, height * width)

        exp = shap.GradientExplainer(self.model,
                                     data=images_train 
                                    )

        expgrad_shap_values = exp.shap_values(images_test[:self.n])

        if self.type == 'betaVAE' or self.type == 'betaTCVAE':
            expgrad_shap_values = np.asarray(expgrad_shap_values).reshape(
                                        self.out_dim, self.n, height, width)
            images_test = images_test.view(-1, 1, height, width)

        return expgrad_shap_values, images_test[:self.n]

    def kernel_shap(self):
        with torch.no_grad():
            iter_obj = iter(self.datamodule)
            images_test, labels_test = iter_obj.next()
            images_train, labels_train = iter_obj.next()

            height = images_train[0].shape[1]
            width = images_train[0].shape[2]

            if self.type == "betaVAE" or self.type == "betaTCVAE":
                images_train = images_train.view(-1, height * width)
                images_test = images_test.view(-1, height * width)

        exp = shap.KernelExplainer(self.kernel_model,
                                   data=images_train.detach().numpy()
                                )

        kernel_shap_values = exp.shap_values(images_test[:self.n].detach().numpy(), 
                                            check_additivity=True)

        if self.type == 'betaVAE' or self.type == 'betaTCVAE':
            kernel_shap_values = np.asarray(kernel_shap_values).reshape(
                                        self.out_dim, self.n, height, width)
            images_test = images_test.view(-1, 1, height, width)

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
    def __init__(self, model, encoder, datamodule, method, type):
        self.model = model
        self.model.eval()

        self.encoder = encoder
        self.encoder.eval()


        self.method = method

        self.datamodule = datamodule
        self.type = type

    def expgrad_shap(self):
        with torch.no_grad():
            iter_obj = iter(self.datamodule)
            images_test, labels_test = iter_obj.next()
            images_train, labels_train = iter_obj.next()

            height = images_train[0].shape[1]
            width = images_train[0].shape[2]

            if self.method == "IG":
                images_train = torch.zeros((1, 1, height, width))

            if self.type == "betaVAE" or self.type == "betaTCVAE":
                images_train = images_train.view(-1, height * width)
                images_test = images_test.view(-1, height * width)

            encoder = self.encoder

            encoding_train, _ = encoder.encode(images_train)
            encoding_test, _ = encoder.encode(images_test)
            
        exp = shap.GradientExplainer(self.model, 
                                     data = encoding_train
                                    )
        
        expgrad_shap_values = exp.shap_values(encoding_test)
        return exp, expgrad_shap_values, encoding_test.numpy().astype('float32'), labels_test

    def deep_shap(self):
        with torch.no_grad():
            iter_obj = iter(self.datamodule)
            images_test, labels_test = iter_obj.next()
            images_train, labels_train = iter_obj.next()

            height = images_train[0].shape[1]
            width = images_train[0].shape[2]

            if self.type == "betaVAE" or self.type == "betaTCVAE":
                images_train = images_train.view(-1, height * width)
                images_test = images_test.view(-1, height * width)

            encoder = self.encoder

            encoding_train, _ = encoder.encode(images_train)
            encoding_test, _ = encoder.encode(images_test)

        exp = shap.DeepExplainer(self.model,
                                 data=encoding_train
                                )

        deep_shap_values = exp.shap_values(encoding_test, check_additivity=True)
        return exp, deep_shap_values, encoding_test.numpy().astype('float32'), labels_test

    def kernel_shap(self):
        with torch.no_grad():
            iter_obj = iter(self.datamodule)
            images_test, labels_test = iter_obj.next()
            images_train, labels_train = iter_obj.next()

            height = images_train[0].shape[1]
            width = images_train[0].shape[2]

            if self.type == "betaVAE" or self.type == "betaTCVAE":
                images_train = images_train.view(-1, height * width)
                images_test = images_test.view(-1, height * width)

            encoder = self.encoder

            encoding_train, _ = encoder.encode(images_train)
            encoding_test, _ = encoder.encode(images_test)

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
        print("\nVisualizing Attribution of LSF into Predictions... ")

        if self.method == "DeepSHAP":
            exp, values, encoding, labels_test = self.deep_shap()
        elif self.method == "KernelSHAP":
            exp, values, encoding, labels_test = self.kernel_shap()
        else:
            exp, values, encoding, labels_test = self.expgrad_shap()

        return exp, values, encoding, labels_test