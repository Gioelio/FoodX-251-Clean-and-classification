from tensorflow import keras

class Model_util:
    # TODO: il cut layer non è completamente definito
    def __init__(self, model_name, cut_layer):
        if model_name != 'resnet' and model_name != 'vgg' and model_name != 'efficient_net':
            raise ValueError("Variable 'model_name' can assume only those values: 'resnet', 'vgg', 'efficient_net'")

        self.model_name = model_name;
        self.cut_layer = cut_layer;

    def generate_model(self, summary=False):
        model = None

        if self.model_name == 'efficient_net':
            cut_layer_name = 'block7a_se_reduce'
            base_model = keras.applications.EfficientNetB0(include_top=True, weights='imagenet')
            self.preprocess_fun = lambda x: keras.applications.efficientnet.preprocess_input(x)
            if self.cut_layer:
                model = keras.Model(inputs=base_model.input, outputs=base_model.get_layer(cut_layer_name).output)
            
        ## resnet definition
        elif self.model_name == 'resnet':
            cut_layer_name = 'conv4_block36_2_relu'
            base_model = keras.applications.ResNet152V2(
                include_top=True,
                weights="imagenet",
                pooling=None,
            )

            model = keras.Model(inputs=base_model.input, outputs=base_model.get_layer(cut_layer_name).output)

            self.preprocess_fun = lambda x: keras.applications.resnet_v2.preprocess_input(x)
        ## vgg definition
        elif self.model_name == 'vgg':

            cut_layer_name = 'fc1'
            base_model = keras.applications.VGG16(
                include_top=True,
                weights="imagenet",
                pooling=None,
            )

            model = keras.Model(inputs=base_model.input, outputs=base_model.get_layer(cut_layer_name).output)
            self.preprocess_fun = lambda x: keras.applications.vgg16.preprocess_input(x)

        if summary:
            model.summary()

        return model