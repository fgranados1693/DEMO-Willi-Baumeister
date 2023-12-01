import cv2
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from shiny import ui, render, App
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import random_projection
from pathlib import Path
import shinyswatch
from PIL import Image

#Refs
css_path = Path(__file__).parent / "www" / "styles.css"
image_path = Path(__file__).parent / "www"
image_black_meta = str(image_path / "mblack_meta.png")
image_bluxao = str(image_path / "mbluxao-1955.png")
image_mit_rotem_kreis = str(image_path / "mit_rotem_kreis.png")

print(image_black_meta)
#Variables 
n_grid = 10

app_ui = ui.page_fluid(
    shinyswatch.theme.minty(), 
    ui.include_css(css_path),
    ui.tags.h2("Willi Baumeister data visualizer", class_="app-heading"),   
    ui.row(
        {"class": "input_container"},
        ui.column(6,
            ui.input_radio_buttons("image_choice", "Choose an Image",
            choices = {
            "mblack_meta.png": "Black metamorphosis",
            "mbluxao-1955.png": "Bluxao",
            "mit_rotem_kreis.png": "Mit rotem Kreis"}, selected="mblack_meta.png"),
        ),   
        ui.column(6,
            ui.tags.h4("Upload an image from device"),
            ui.input_file("upload_image", "", multiple=False)           
        )         
    ),
    ui.tags.div(
         {"class": "card"},
        ui.row(
            ui.column(2,
                ui.row(
                    ui.input_slider("xrange", "X range:", min=1, max=n_grid, step=1, value=0.),
                ),
                ui.row(
                    ui.input_slider("yrange", "Y range:", min=1, max=n_grid, step=1, value=0.),
                ),
            ),        
            ui.column(3,
                ui.output_plot("original_image_plot_with_section"),
            ),
            ui.column(3,
                ui.output_plot("downsampled_image_plot"),
            ),
            ui.column(3,
                ui.output_plot("color_points_plot"),
            ),
        ),
    ),
    ui.tags.div(
         {"class": "card"},
        ui.row(
            ui.column(2,        
                ui.input_slider("obs", "Row:", min=0., max=1., value=0.5),
            ),
            ui.column(3,
                ui.output_plot("frequency_spectrum_plot"),
            ),
            ui.column(3,
                ui.output_plot("pixel_intensity_histogram"),
            ),        
        ),  
    ),   
    ui.tags.div(  
        {"class": "card"},
        ui.row(
            ui.column(2,         
                ui.input_slider("threshold", "Threshold:", min=0., max=1., value=0.5),
            ),
            ui.column(3,
                ui.output_plot("plot_reds"),
            ),
            ui.column(3,
                ui.output_plot("plot_greens"),
            ),   
            ui.column(3,
                ui.output_plot("plot_blues"),
            ),                   
        ),
    ),
)

def server(input, output, session):

 #   @output
 #   @render.plot
 #   def original_image_plot():
 #       image_rgb = load_image_rgb(input.image_choice())
 #       plt.imshow(image_rgb)
 #       plt.axis('off')
 #       plt.title("Original Image")
 #       return plt.gcf()

    @output
    @render.plot
    def original_image_plot():
        image_rgb = load_image_rgb(input.image_choice())
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.title("Original Image")
        return plt.gcf()

    @output
    @render.plot
    def original_image_plot_with_section():
        image_rgb, x_range, y_range = load_and_slice_image(input)
        mark_section_on_image(image_rgb, x_range, y_range)
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.title("Original Image with Section")
        return plt.gcf()

    @output
    @render.plot
    def downsampled_image_plot():
        image_rgb, x_range, y_range = load_and_slice_image(input)
        downsampled_image = image_rgb[y_range[0]:y_range[1], x_range[0]:x_range[1]]
        plt.imshow(downsampled_image)
        plt.axis('off')
        plt.title("Downsampled Image")
        return plt.gcf()

    @output
    @render.plot
    def color_points_plot():
        image_rgb, x_range, y_range = load_and_slice_image(input)
        downsampled_image = image_rgb[y_range[0]:y_range[1], x_range[0]:x_range[1]]
        pixels = downsampled_image.reshape((-1, 3))
        pca = PCA(n_components=2)
        pixels_reduced = pca.fit_transform(pixels)

        # Create figure and axes
        fig, ax = plt.subplots()

        # Scatter plot
        scatter = ax.scatter(pixels_reduced[:, 0], pixels_reduced[:, 1], color=pixels / 255)

        # Set aspect of the plot to be equal to make it square
        X_size = np.max(pixels_reduced[:,0]) - np.min(pixels_reduced[:,0])
        Y_size = np.max(pixels_reduced[:,1]) - np.min(pixels_reduced[:,1])        
        ax.set_aspect(aspect=X_size/Y_size)

        # Set titles and labels
        ax.set_title("Color Points (Reduced to 2D)")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")

        # Return the figure object
        return fig


    # @output
    # @render.plot
    # def color_points_plot():
    #     image_rgb, x_range, y_range = load_and_slice_image(input)
    #     downsampled_image = image_rgb[y_range[0]:y_range[1], x_range[0]:x_range[1]]
    #     pixels = downsampled_image.reshape((-1, 3))
    #     pca = PCA(n_components=2)
    #     pixels_reduced = pca.fit_transform(pixels)
    #     plt.scatter(pixels_reduced[:, 0], pixels_reduced[:, 1], color=pixels / 255)
    #     plt.title("Color Points (Reduced to 2D)")
    #     plt.xlabel("PCA Component 1")
    #     plt.ylabel("PCA Component 2")
    #     return plt.gcf()

    @output
    @render.plot
    def frequency_spectrum_plot():
        image_rgb, x_range, y_range = load_and_slice_image(input)
        downsampled_image = image_rgb[y_range[0]:y_range[1], x_range[0]:x_range[1]]

        f = np.fft.fft2(downsampled_image)
        fshift = np.fft.fftshift(f)
        center_row = int(input.obs() * fshift.shape[0])
        line_frequency_data = fshift[center_row, :]
        magnitude_spectrum = np.abs(line_frequency_data)
        
        plt.plot(magnitude_spectrum)
        plt.ylim(1, 1e5)
        plt.yscale("log")
        plt.title("Frequency Spectrum along a Line")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")

        return plt.gcf()

    @output
    @render.plot
    def pixel_intensity_histogram():
        image_rgb, x_range, y_range = load_and_slice_image(input)
        downsampled_image = image_rgb[y_range[0]:y_range[1], x_range[0]:x_range[1]]

        # Convert to grayscale for intensity histogram
        grayscale_image = cv2.cvtColor(downsampled_image, cv2.COLOR_RGB2GRAY)
        
        # Calculate histogram
        hist = cv2.calcHist([grayscale_image], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.max()  # Normalize the histogram

        # Create bins for the histogram
        bins = np.arange(257)

        # Plot the histogram as a bar chart
        plt.bar(bins[:-1], hist, width=1, color='blue')
        plt.xlim([0, 256])
        plt.title("Pixel Intensity Histogram")
        plt.yscale("log")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Normalized Frequency")
        return plt.gcf()

    @output
    @render.plot
    def plot_reds():
        image_rgb, x_range, y_range = load_and_slice_image(input)
        red_channel, green_channel, blue_channel = cv2.split(image_rgb)
        threshold = input.threshold()*60
        red_mask = (red_channel > green_channel + threshold) & (red_channel > blue_channel + threshold)
        white_image = np.ones_like(image_rgb) * 255
        result_image = np.where(red_mask[:, :, None], image_rgb, white_image)
        plt.imshow(result_image)
        plt.title('Red Channel after Thresholding', color = '#333333')
        plt.axis('off')
        return plt.gcf()

    @output
    @render.plot
    def plot_greens():
        image_rgb, x_range, y_range = load_and_slice_image(input)
        red_channel, green_channel, blue_channel = cv2.split(image_rgb)
        threshold = input.threshold() * 5
        green_mask = (green_channel > red_channel + threshold) & (green_channel > blue_channel + threshold)
        white_image = np.ones_like(image_rgb) * 255
        result_image = np.where(green_mask[:, :, None], image_rgb, white_image)
        plt.imshow(result_image)
        plt.title('Green Channel after Thresholding', color = '#333333')
        plt.axis('off')
        return plt.gcf()

    @output
    @render.plot
    def plot_blues():
        image_rgb, x_range, y_range = load_and_slice_image(input)
        red_channel, green_channel, blue_channel = cv2.split(image_rgb)
        threshold = input.threshold() * 30
        blue_mask = (blue_channel > red_channel + threshold) & (blue_channel > green_channel + threshold)
        white_image = np.ones_like(image_rgb) * 255
        result_image = np.where(blue_mask[:, :, None], image_rgb, white_image)
        plt.imshow(result_image)
        plt.title('Blue Channel after Thresholding', color = '#333333')
        plt.axis('off')
        return plt.gcf()

def apply_threshold(image_rgb, channel_index, threshold):
    channel = image_rgb[:, :, channel_index]
    return np.where(channel > threshold, 1, 0)

def load_image_rgb(filename):
    image = cv2.imread(f"./{filename}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def load_and_slice_image(input):
    image_choice = input.image_choice()
    if image_choice in ["mblack_meta.png", "mbluxao-1955.png", "mit_rotem_kreis.png"]:
        image_rgb = load_image_rgb(image_choice)
    elif input.upload_new_image:
        uploaded_image = input.upload_new_image
        if uploaded_image is not None:
             image_rgb = load_image_rgb(uploaded_image)
    else:
        # Default to A.png if no valid choice or uploaded image
        image_rgb = load_image_rgb("mblack_meta.png")

    img_length, img_width = image_rgb.shape[:2]
    x_range, y_range = calculate_slice_range(input.xrange(), input.yrange(), img_length, img_width)
    return image_rgb, x_range, y_range


def calculate_slice_range(x_input, y_input, img_length, img_width):
    x_range = (int((x_input - 1) / n_grid * img_width), int(x_input / n_grid * img_width))
    y_range = (int((y_input - 1) / n_grid * img_length), int(y_input / n_grid * img_length))
    return x_range, y_range

def mark_section_on_image(image_rgb, x_range, y_range):
    i_start, i_end = y_range
    j_start, j_end = x_range
    image_rgb[i_start:i_end, j_start-4:j_start+4] = 255
    image_rgb[i_start:i_end, j_end-4:j_end+4] = 255
    image_rgb[i_start-4:i_start+4, j_start:j_end] = 255
    image_rgb[i_end-4:i_end+4, j_start:j_end] = 255


app = App(app_ui, server)
