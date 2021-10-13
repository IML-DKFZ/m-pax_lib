import numpy as np
import matplotlib.pyplot as plt
import shap

"""
Creates two objects visualizing the attribution from the original input images
and the LSFs. Attribution based on the LSFs is visualized locally and globally (total).
All visualizations are saved into the /images folder.
"""


class vis_AM_Original:
    def __init__(self, shap_values, test_images):
        self.shap_values = shap_values
        self.test_images = test_images

    def prep(self, shap_values, test_images):
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
        return shap_numpy, test_numpy

    def visualise(self):
        shap_values, test_values = self.prep(self.shap_values, self.test_images)
        plt.figure(figsize=(20, 14), dpi=200)
        shap.image_plot(shap_values, test_values, show=False)


class vis_AM_Latent:
    def __init__(self, shap_values, explainer, encoding_test, labels_test, output_dir):
        self.shap_values = shap_values
        self.encoding_test = encoding_test
        self.labels_test = labels_test
        self.exp = explainer
        self.n = len(np.unique(labels_test.numpy()))
        self.output_dir = output_dir

        if self.n == 10:
            self.labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif self.n == 4:
            self.labels = ["CNV", "DME", "Drusen", "Normal"]
        else:
            self.labels = ["square", "ellipse", "heart"]

    def visualise(self):

        plt.figure(figsize=(24, 20), dpi=200)
        shap.summary_plot(
            self.shap_values,
            self.encoding_test,
            plot_type="bar",
            color=plt.cm.tab10,
            class_names=self.labels,
            show=False,
        )
        plt.savefig(self.output_dir + "total_attribution_latent_features.png")

        fig, axes = plt.subplots(nrows=1, ncols=4)
        plt.figure(figsize=(30, 10), dpi=200)
        axes = axes.ravel()

        for i in range(0, 4, 1):
            plt.subplot(1, 4, i + 1)
            shap.multioutput_decision_plot(
                np.zeros((1, self.n)).tolist()[0],
                self.shap_values,
                highlight=self.labels_test[i],
                legend_labels=self.labels,
                legend_location="lower left",
                show=False,
                auto_size_plot=False,
                row_index=i,
            )

        plt.savefig(self.output_dir + "local_attribution_latent_features.png")
