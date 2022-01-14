import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import shap
from captum.attr import NoiseTunnel, GradientShap, Occlusion
from captum.attr import visualization as vis
from matplotlib.colors import LinearSegmentedColormap
from torch import nn
from PIL import Image

heat_cmap = LinearSegmentedColormap.from_list(
    "heatmap attribution", [(0, "#008BFB"), (0.5, "#ffffff"), (1, "#FF0051")], N=256
)

abs_cmap = LinearSegmentedColormap.from_list(
    "absolute attribution", [(0, "#ffffff"), (1, "#8F37BB")], N=256
)


class AttributionOriginalY:
    """Computes and visualizes the attribution of the original image into the prediction.
    The object contains two methods:
        - attribution
        - visualization
    Whereby the visualization method calls the attribution method.

    """

    def __init__(
        self, baseline, dataloader, dataset, head, index, kernel_size, output_dir
    ):
        """Called upon initialization. Selects label names based on dataset name.

        Parameters
        ----------
        baseline : float
            Baseline value for occlusion maps.
        dataloader : torch.utils.data.Dataloader
            Provides an iterable dataloader over the given dataset.
        dataset : str
            Name of dataset.
        head : src.models.head_mlp.MLP
            Head for downstream task prediction.
        index : int
            Index of image for attribution visualization.
        kernel_size : int
            Quadratic kernel size for occlusion maps.
        output_dir : str
            Folder to output images.
        """
        self.baseline = baseline
        self.dataloader = dataloader
        self.head = head
        self.index = index
        self.kernel_size = kernel_size
        self.output_dir = output_dir

        if dataset == "MNISTDataModule":
            self.labels_name = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif dataset == "DiagVibSixDataModule":
            self.labels_name = [0, 2, 5]
        elif dataset == "ISICDataModule":
            self.labels_name = [
                "MEL",
                "NV",
                "BCC",
                "AK",
                "BKL",
                "DF",
                "VASC",
                "SCC",
                "UNK",
            ]
        else:
            self.labels_name = ["CNV", "DME", "Drusen", "Normal"]

    def attribution(self):
        """Computes occlusion and expected gradient based attribution for image selected via "index".
        Is called within the visualization method.

        Returns
        -------
        torch.Tensor, torch.Tensor
            Returns attribution maps.
        """
        with torch.no_grad():
            self.images, self.labels = next(iter(self.dataloader))

            channels = self.images.shape[1]

        gradient_shap = GradientShap(self.head)

        attributions_gs = gradient_shap.attribute(
            self.images[self.index].unsqueeze(0),
            n_samples=200,
            stdevs=0.0001,
            baselines=self.images,
            target=self.labels[self.index],
        )

        occlusion = Occlusion(self.head)

        attributions_occ = occlusion.attribute(
            self.images[self.index].unsqueeze(0),
            strides=(3, 5, 5),
            target=self.labels[self.index],
            sliding_window_shapes=(channels, self.kernel_size, self.kernel_size),
            baselines=self.baseline,
        )

        return attributions_gs, attributions_occ

    def visualization(self):
        """Computes and saves graphics for via "index" selected image into "output_dir".
        Also calls attribution computation.
        """

        attributions_gs, attributions_occ = self.attribution()

        original_image = np.transpose(
            (self.images[self.index].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0)
        )

        fig, axis = vis.visualize_image_attr_multiple(
            np.transpose(attributions_gs.squeeze(0).cpu().detach().numpy(), (1, 2, 0)),
            original_image,
            ["original_image", "heat_map", "masked_image"],
            ["all", "absolute_value", "absolute_value"],
            cmap=abs_cmap,
            show_colorbar=False,
            use_pyplot=False,
            fig_size=(20, 10),
        )

        axis[0].imshow(original_image, cmap="gray")

        output = self.head(self.images[self.index].unsqueeze(0)) * 100
        pred_label = self.labels_name[output.argmax(1)]
        true_label = self.labels_name[self.labels[self.index]]
        prob_true = (
            output[0][self.labels[self.index]].detach().numpy().round(2).astype("str")
        )
        prob_pred = (
            output[0][output.argmax(1)].detach().numpy()[0].round(2).astype("str")
        )

        managed_fig = plt.figure()
        canvas_manager = managed_fig.canvas.manager
        canvas_manager.canvas.figure = fig
        fig.set_canvas(canvas_manager.canvas)
        fig._original_dpi = 200

        axis[0].set_title("Original Image \n", fontsize=18)
        axis[1].set_title(
            "Abs. Attribution into Ground Truth Label: "
            + str(true_label)
            + " ("
            + str(prob_true)
            + "%)"
            + "\n Predicted Label: "
            + str(pred_label)
            + " ("
            + str(prob_pred)
            + "%) \n",
            fontsize=18,
        )
        axis[2].set_title("Masked Image \n", fontsize=18)

        plt.savefig(self.output_dir + "image1.png")

        fig, axis = vis.visualize_image_attr_multiple(
            np.transpose(attributions_occ.squeeze(0).cpu().detach().numpy(), (1, 2, 0)),
            original_image,
            ["blended_heat_map", "blended_heat_map", "blended_heat_map"],
            ["all", "all", "all"],
            show_colorbar=False,
            outlier_perc=2,
            use_pyplot=False,
            fig_size=(20, 10),
            cmap=heat_cmap,
        )

        managed_fig = plt.figure()
        canvas_manager = managed_fig.canvas.manager
        canvas_manager.canvas.figure = fig
        fig.set_canvas(canvas_manager.canvas)
        fig._original_dpi = 200

        axis[1].set_title("Occlusion Map for Ground Truth \n", fontsize=18)

        plt.savefig(self.output_dir + "image2.png")

        image1 = Image.open(self.output_dir + "image1.png")
        image2 = Image.open(self.output_dir + "image2.png")
        width, height = image2.size

        left = (width - 1200) / 2
        top = (height - 2000) / 2
        right = (width + 1350) / 2
        bottom = (height + 2000) / 2

        image2 = image2.crop((left, top, right, bottom))
        image1_size = image1.size
        image2_size = image2.size
        new_image = Image.new(
            "RGB", (image1_size[0] + image2_size[0], image1_size[1]), (250, 250, 250)
        )
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (image1_size[0], 0))
        new_image.save(self.output_dir + "attribution_original_y.jpg", "JPEG")

        os.remove(self.output_dir + "image1.png")
        os.remove(self.output_dir + "image2.png")


class AttributionLatentY:
    """Computes and visualizes the attribution of the latent representation into the prediction.
    The object contains two methods:
        - attribution
        - visualization
    Whereby the visualization method calls the attribution method.

    """

    def __init__(self, dataloader, dataset, encoder, head, index, output_dir):
        """Called upon initialization. Selects label names based on dataset name.

        Parameters
        ----------
        dataloader : torch.utils.data.Dataloader
            Provides an iterable dataloader over the given dataset.
        dataset : str
            Name of dataset.
        encoder : src.models.tcvae_conv.betaTCVAE_Conv or src.models.tcvae_resnet.betaTCVAE_ResNet
            beta-TCVAE trained encoder, encoding the (disentangled) latent representations.
        head : src.models.head_mlp.MLP
            Head for downstream task prediction.
        index : int
            Index of image for attribution visualization.
        output_dir : str
            Folder to output images.
        """
        self.dataloader = dataloader
        self.encoder = encoder
        self.head = head
        self.index = index
        self.output_dir = output_dir

        if dataset == "MNISTDataModule":
            self.labels_name = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif dataset == "DiagVibSixDataModule":
            self.labels_name = [0, 2, 5]
        elif dataset == "ISICDataModule":
            self.labels_name = [
                "MEL",
                "NV",
                "BCC",
                "AK",
                "BKL",
                "DF",
                "VASC",
                "SCC",
                "UNK",
            ]
        else:
            self.labels_name = ["CNV", "DME", "Drusen", "Normal"]

    def attribution(self):
        """Computes expected gradient based attribution for the latent representation
        of the image selected via "index". Is called within the visualization method.

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor
            Returns attribution map, latent representation, and respective label.
        """
        with torch.no_grad():
            images, labels = next(iter(self.dataloader))
            rand_img_dist, _ = next(iter(self.dataloader))

            images = images[self.index :]
            labels = labels[self.index :]

            encoding, _ = self.encoder.encoder(images)
            encoding_dist, _ = self.encoder.encoder(rand_img_dist)

        exp = shap.GradientExplainer(self.head, data=encoding_dist)

        attributions_gs = exp.shap_values(encoding)
        return attributions_gs, encoding, labels

    def visualization(self):
        """Computes and saves graphics for the via "index" selected representation into "output_dir".
        Also calls attribution computation.
        """

        attributions_gs, encoding, labels = self.attribution()

        plt.figure(figsize=(9, 5), dpi=200)
        shap.summary_plot(
            attributions_gs,
            encoding,
            plot_type="bar",
            color=plt.cm.tab10,
            class_names=self.labels_name,
            show=False,
        )
        plt.savefig(self.output_dir + "attribution_latent_y_global.png")

        fig = plt.figure(figsize=(5, 4), dpi=200)

        shap.multioutput_decision_plot(
            np.zeros((1, len(self.labels_name))).tolist()[0],
            attributions_gs,
            highlight=labels[0],
            legend_labels=self.labels_name,
            legend_location="lower right",
            show=False,
            auto_size_plot=False,
            row_index=0,
            link="logit",
        )

        plt.tight_layout()

        plt.savefig(self.output_dir + "attribution_latent_y_local.png")


class Wrapper(nn.Module):
    """Module wrapping an encoder to only return the latent representation sample from the forward pass,
    ignoring the reconstruction and latent distribution parameters.
    """

    def __init__(self, encoder):
        """Called upon initialization.

        Parameters
        ----------
        encoder : src.models.tcvae_conv.betaTCVAE_Conv or src.models.tcvae_resnet.betaTCVAE_ResNet
            beta-TCVAE trained encoder, encoding the (disentangled) latent representations.
        """
        super().__init__()
        self.encoder = encoder

    def forward(self, image):
        """Returns latent representation sample.

        Parameters
        ----------
        image : torch.Tensor
            Input image.

        Returns
        -------
        torch.Tensor
            Latent representation sample.
        """
        _, _, latent_sample = self.encoder(image)
        return latent_sample


class AttributionOriginalLatent:
    """Computes and visualizes the attribution of the original image into the latent representation.
    The object contains two methods:
        - attribution
        - visualization
    Whereby the visualization method calls the attribution method.

    """

    def __init__(
        self, encoder, dataloader, index, latent_dim, output_dir, baseline, kernel_size
    ):
        self.baseline = baseline
        self.dataloader = dataloader
        self.encoder = Wrapper(encoder)
        self.index = index
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.output_dir = output_dir

    def attribution(self):
        """Computes occlusion and expected gradient based attribution for image selected via "index".
        Is called within the visualization methode.

        Returns
        -------
        torch.Tensor, torch.Tensor
            Returns attribution maps.
        """

        with torch.no_grad():
            self.images, self.labels = next(iter(self.dataloader))

            channels = self.images.shape[1]

        gradient_shap = GradientShap(self.encoder)

        attributions_gs = []

        for i in range(0, self.latent_dim):
            attributions_gs.append(
                gradient_shap.attribute(
                    self.images[self.index].unsqueeze(0),
                    n_samples=200,
                    stdevs=0.0001,
                    baselines=self.images,
                    target=i,
                )
            )

        occlusion = Occlusion(self.encoder)

        attributions_occ = []

        for i in range(0, self.latent_dim):
            attributions_occ.append(
                occlusion.attribute(
                    self.images[self.index].unsqueeze(0),
                    strides=(3, 5, 5),
                    target=i,
                    sliding_window_shapes=(
                        channels,
                        self.kernel_size,
                        self.kernel_size,
                    ),
                    baselines=self.baseline,
                )
            )

        return attributions_gs, attributions_occ

    def visualization(self):
        """Computes and saves graphics for the via "index" selected image into "output_dir".
        The subplots are saved seperatly in "output_dir" before they are combined. Also calls attribution computation.
        """

        attributions_gs, attributions_occ = self.attribution()

        original_image = np.transpose(
            (self.images[self.index].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0)
        )

        for i in range(0, self.latent_dim):
            fig, axis = vis.visualize_image_attr(
                np.transpose(
                    attributions_gs[i].squeeze(0).cpu().detach().numpy(), (1, 2, 0)
                ),
                original_image,
                "heat_map",
                "absolute_value",
                cmap=abs_cmap,
                show_colorbar=False,
                use_pyplot=False,
                fig_size=(20, 10),
            )

            managed_fig = plt.figure()
            canvas_manager = managed_fig.canvas.manager
            canvas_manager.canvas.figure = fig
            fig.set_canvas(canvas_manager.canvas)
            fig._original_dpi = 200

            axis.set_title("Latent Feature " + str(i) + "\n ", fontsize=24)

            plt.savefig(self.output_dir + "image" + str(i) + ".png")

        for i in range(0, self.latent_dim):
            fig, axis = vis.visualize_image_attr(
                np.transpose(
                    attributions_occ[i].squeeze(0).cpu().detach().numpy(), (1, 2, 0)
                ),
                original_image,
                "blended_heat_map",
                "all",
                show_colorbar=False,
                outlier_perc=2,
                use_pyplot=False,
                fig_size=(20, 10),
                cmap=heat_cmap,
            )

            managed_fig = plt.figure()
            canvas_manager = managed_fig.canvas.manager
            canvas_manager.canvas.figure = fig
            fig.set_canvas(canvas_manager.canvas)
            fig._original_dpi = 200

            plt.savefig(self.output_dir + "image" + str(i + 10) + ".png")

        left = (4000 - 1600) / 2
        top = (2000 - 2000) / 2
        right = (4000 + 1700) / 2
        bottom = (2000 + 1700) / 2

        plots = []

        for i in range(0, self.latent_dim):
            image = Image.open(self.output_dir + "image" + str(i) + ".png")
            image = image.crop((left, top, right, bottom))
            plots.append(image)

        top = (2000 - 1650) / 2

        for i in range(self.latent_dim, self.latent_dim * 2):
            image = Image.open(self.output_dir + "image" + str(i) + ".png")
            image = image.crop((left, top, right, bottom))
            plots.append(image)

        width, height = plots[0].size

        new_image = Image.new("RGB", (width * 10, height * 2), (250, 250, 250))

        for i in range(0, self.latent_dim * 2):
            if i < 10:
                new_image.paste(plots[i], (i * width, 0))
            else:
                new_image.paste(plots[i], ((i - 10) * width, height))

        new_image.save(self.output_dir + "attribution_original_latent.jpg", "JPEG")

        for i in range(0, self.latent_dim * 2):
            os.remove(self.output_dir + "image" + str(i) + ".png")
