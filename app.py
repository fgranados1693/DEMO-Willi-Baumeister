import matplotlib.pyplot as plt
import numpy as np
from shiny import ui, render, App
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


t = np.linspace(0, 2 * np.pi, 1024)
data2d = np.sin(t)[:, np.newaxis] * np.cos(t)[np.newaxis, :]

app_ui = ui.page_fixed(
    ui.tags.head(
        ui.tags.link(rel="stylesheet", href="https://bootswatch.com/4/sandstone/bootstrap.min.css")
    ),
    ui.row(
        ui.column(4,ui.input_radio_buttons("image_choice", "Choose an Image",
            choices={"A.png": "A", "B.png": "B", "C.png": "C"}, selected="A")),
        ui.column(4, ui.output_plot("original_image_plot")),   # Each plot in its own column
        ui.column(4, ui.output_plot("color_points_plot"))
    ),
    ui.row(
        ui.input_slider("obs", "Row:", min=0., max=1., value=0.5),
        ui.output_plot("frequency_spectrum_plot")),
    ui.row(
        ui.h2("Playing with colormaps"),
        ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_radio_buttons("channel", "Channel (rgb)",
                choices={"0": "R", "1": "G", "2": "B"}
            ),
            ui.input_slider("range", "Color range", 0, 255, value=(-1, 1), step=0.05),
        ),
        ui.panel_main(
            ui.output_plot("plot")
        )
    ))
)

#app_ui = ui.page_fixed(
#    ui.output_plot("original_image_plot"),
#    ui.output_plot("downsampled_image_plot"),
#    ui.output_plot("color_points_plot")
#)

def server(input, output, session):
    @output
    @render.plot
    def original_image_plot():
        file_name = f"./{input.image_choice()}"
        image = cv2.imread(file_name)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.title("Original Image")
        return plt.gcf()  # Return the current figure

    @output
    @render.plot
    def downsampled_image_plot():
        file_name = f"./{input.image_choice()}"
        image = cv2.imread(file_name)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        scale_percent = 25  # percentage of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        downsampled_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        downsampled_image_rgb = cv2.cvtColor(downsampled_image, cv2.COLOR_BGR2RGB)

        plt.imshow(downsampled_image)
        plt.axis('off')
        plt.title("Downsampled Image")
        return plt.gcf()

    @output
    @render.plot
    def color_points_plot():

        file_name = f"./{input.image_choice()}"
        image = cv2.imread(file_name)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        scale_percent = 25  # percentage of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        downsampled_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        downsampled_image_rgb = cv2.cvtColor(downsampled_image, cv2.COLOR_BGR2RGB)

        # ... [PCA and scatter plot code]
        # Step 2: Reshape the image to a 2D array of pixels
        pixels = downsampled_image_rgb.reshape((-1, 3))
        
        # Optional: Reduce color dimensions for visualization
        pca = PCA(n_components=2)
        #pixels_reduced = pca.fit_transform(pixels)
        pixels_reduced = pca.fit_transform(pixels)
        plt.scatter(pixels_reduced[:, 0], pixels_reduced[:, 1], color=pixels / 255)
        plt.title("Color Points (Reduced to 2D)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        return plt.gcf()

    @output
    @render.plot
    def frequency_spectrum_plot():
        # Apply Fourier Transform
        file_name = f"./{input.image_choice()}"
        image = cv2.imread(file_name)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        f = np.fft.fft2(image_rgb)
        fshift = np.fft.fftshift(f)
        #magnitude_spectrum = 20 * np.log(np.abs(fshift))

        center_row = int(input.obs() * fshift.shape[0])
        line_frequency_data = fshift[center_row, :]

        # Calculate the magnitude
        magnitude_spectrum = np.abs(line_frequency_data)
        
        # Plot the frequency spectrum along the line
        plt.plot(magnitude_spectrum)
        plt.ylim(1, 1e5)
        plt.yscale("log")
        plt.title("Frequency Spectrum along a Line")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")

        return plt.gcf()

    @output
    @render.plot
    def plot():
        file_name = f"./{input.image_choice()}"
        image = cv2.imread(file_name)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots()
        print(input.channel())
        img = image_rgb[:, :, int(input.channel())]
        print(np.min(image_rgb), np.min(image_rgb), image_rgb.shape)
        im = ax.imshow(img, vmin=input.range()[0], vmax=input.range()[1])
        fig.colorbar(im, ax=ax)
        return fig



app = App(app_ui, server)
