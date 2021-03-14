import napari
from dask_image.imread import imread

stack = imread("./newdata/images_old/*.jpg")
stack2 = imread("./resultsThreshold/HarDMSEG/reconstructed_haveLabel_test/*.jpg")
stack3 = imread("./results/HarDMSEG/haveLabel_test/*.png")

with napari.gui_qt():
    viewer = napari.view_image(stack, name='Images')
    label_layer = viewer.add_image(stack2, name='Predicted Full Labels',opacity =0.5,visible =False,gamma=100000)
    label_layer2 = viewer.add_image(stack3, name='Predicted Patch Labels',opacity =0.5,visible =False,gamma=100000)