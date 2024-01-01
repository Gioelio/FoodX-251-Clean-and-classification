from tensorflow import keras


def build_feature_extractor(base_builder, cut_layer):
    feature_extractor = base_builder()
    feature_extractor = keras.Model(inputs=feature_extractor.input,
                                    outputs=feature_extractor.get_layer(cut_layer).output)
    feature_extractor = keras.Sequential([feature_extractor, keras.layers.Flatten()])
    return feature_extractor
