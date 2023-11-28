import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from shiny import ui, render, App
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

n_grid = 10

app_ui = ui.page_fixed(
    ui.row(
        ui.column(6, ui.input_radio_buttons("image_choice", "Choose an Image",
                                             choices={"A.png": "A", "B.png": "B", "C.png": "C"},
                                             selected="A")),
        ui.column(6, ui.output_plot("original_image_plot")),
    ),
    ui.row(
        ui.row(ui.input_slider("xrange", "X range:", min=1, max=n_grid, step=1, value=0.)),
        ui.row(ui.input_slider("yrange", "Y range:", min=1, max=n_grid, step=1, value=0.)),
        ui.column(4, ui.output_plot("original_image_plot_with_section")),
        ui.column(4, ui.output_plot("downsampled_image_plot")),
        ui.column(4, ui.output_plot("color_points_plot"))
    ),
    ui.row(
        ui.input_slider("obs", "Row:", min=0., max=1., value=0.5),
        ui.output_plot("frequency_spectrum_plot"),
    ),       
    ui.row(
        ui.input_slider("threshold", "threshold:", min=0., max=1., value=0.5),
        ui.column(4, ui.output_plot("red_channel_plot")),
        ui.column(4, ui.output_plot("green_channel_plot")),   
        ui.column(4, ui.output_plot("blue_channel_plot"))                   
    )
)    


def server(input, output, session):
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
    def red_channel_plot():
        image_rgb, x_range, y_range = load_and_slice_image(input)
        channel_index = 0  # For green channel
        threshold = 128  # Define your threshold here
        channel = image_rgb[:, :, channel_index]
        binary_channel = np.where(channel > threshold, 1, 0)
        plt.imshow(binary_channel, plt.cm.gray)
        plt.title(f'Channel {channel_index} after thresholding')
        plt.axis('off')
        return plt.gcf()

    @output
    @render.plot
    def color_points_plot():
        image_rgb, x_range, y_range = load_and_slice_image(input)
        downsampled_image = image_rgb[y_range[0]:y_range[1], x_range[0]:x_range[1]]
        pixels = downsampled_image.reshape((-1, 3))
        pca = PCA(n_components=2)
        pixels_reduced = pca.fit_transform(pixels)
        plt.scatter(pixels_reduced[:, 0], pixels_reduced[:, 1], color=pixels / 255)
        plt.title("Color Points (Reduced to 2D)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        return plt.gcf()

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
    def red_channel_plot():
        image_rgb, x_range, y_range = load_and_slice_image(input)
        channel_index = 0  # Red channel
        threshold = input.threshold()*255
        binary_channel = apply_threshold(image_rgb, channel_index, threshold)

        red_cmap = ListedColormap(['white', 'red'])
        plt.imshow(binary_channel, cmap=red_cmap)
        plt.title('Red Channel after Thresholding')
        plt.axis('off')
        return plt.gcf()

    @output
    @render.plot
    def green_channel_plot():
        image_rgb, x_range, y_range = load_and_slice_image(input)
        channel_index = 1  # Green channel
        threshold = input.threshold()*255
        binary_channel = apply_threshold(image_rgb, channel_index, threshold)

        green_cmap = ListedColormap(['white', 'green'])
        plt.imshow(binary_channel, cmap=green_cmap)
        plt.title('Green Channel after Thresholding')
        plt.axis('off')
        return plt.gcf()

    @output
    @render.plot
    def blue_channel_plot():
        image_rgb, x_range, y_range = load_and_slice_image(input)
        channel_index = 2  # Blue channel
        threshold = input.threshold()*255
        binary_channel = apply_threshold(image_rgb, channel_index, threshold)

        blue_cmap = ListedColormap(['white', 'blue'])
        plt.imshow(binary_channel, cmap=blue_cmap)
        plt.title('Blue Channel after Thresholding')
        plt.axis('off')
        return plt.gcf()

def apply_threshold(image_rgb, channel_index, threshold):
    channel = image_rgb[:, :, channel_index]
    return np.where(channel > threshold, 1, 0)


def load_image_rgb(filename):
    image = cv2.imread(f"./{filename}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def load_and_slice_image(input):
    image_rgb = load_image_rgb(input.image_choice())
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

