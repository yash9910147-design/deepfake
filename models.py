from __future__ import annotations


def build_model(pretrained: bool = True):
    import tensorflow as tf

    keras = tf.keras
    backbone = keras.applications.Xception(
        weights="imagenet" if pretrained else None,
        include_top=False,
        input_shape=(224, 224, 3),
    )
    backbone.trainable = False
    x = keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(backbone.output)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="real_probability")(x)
    model = keras.Model(backbone.input, outputs, name="xception_deepfake")
    return model, backbone


def is_compatible_xception_model(model) -> bool:
    if (
        tuple(model.input_shape[1:]) != (224, 224, 3)
        or tuple(model.output_shape[1:]) != (1,)
        or not any(layer.name == "block14_sepconv2_act" for layer in model.layers)
    ):
        return False
    activation = getattr(model.layers[-1], "activation", None)
    return getattr(activation, "__name__", None) == "sigmoid"


def compile_model(model, learning_rate: float) -> None:
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
