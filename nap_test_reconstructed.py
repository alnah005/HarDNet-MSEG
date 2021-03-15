import napari
from dask_image.imread import imread

stack = imread("./haveLabel_test/images_old/*.jpg")
stack2 = imread("./haveLabel_test/masks_old/*.jpg")
stack3 = imread("./resultsThreshold/HarDMSEG/reconstructed_haveLabel_test/*.jpg")
stack4 = imread("./results/HarDMSEG/haveLabel_test/*.jpg")

with napari.gui_qt():
    viewer = napari.view_image(stack, name='Images')
    label_layer = viewer.add_image(stack2, name='True Labels',opacity =0.5,visible =False,gamma=100000)
    label_layer2 = viewer.add_image(stack3, name='Predicted Patch Labels',opacity =0.5,visible =False,gamma=100000)
    label_layer3 = viewer.add_image(stack4, name='Predicted Full Labels',opacity =0.5,visible =False,gamma=100000)
