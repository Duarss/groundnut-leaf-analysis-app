# app/ml/classification/model_loader.py
from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.regularizers import l2
from app.core.config import Config

_model = None

# Harus sama dengan DENSE_L2 di skrip training kamu :contentReference[oaicite:3]{index=3}
DENSE_L2 = 1e-4


def build_classification_model():
    """
    Rebuild arsitektur EfficientNet-B4 yang sama seperti di exe_classification_model.py,
    kemudian nanti diisi weights hasil training.
    """
    img_size = (Config.IMG_HEIGHT, Config.IMG_WIDTH)

    # Tidak perlu download ImageNet di inference, cukup weights=None
    base = EfficientNetB4(
        weights=None,
        include_top=False,
        input_shape=(img_size[0], img_size[1], 3),
    )

    x = base.output
    x = GlobalAveragePooling2D(name="gap")(x)
    x = Dropout(0.5, name="drop1")(x)
    x = Dense(
        256,
        activation="relu",
        kernel_regularizer=l2(DENSE_L2),
        name="dense",
    )(x)
    x = Dropout(0.4, name="drop2")(x)
    out = Dense(
        Config.NUM_CLASSES,
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(x)

    model = Model(inputs=base.input, outputs=out, name="EffNetB4_Classifier")
    return model


def get_classification_model():
    """
    Lazy-load:
    - Bangun arsitektur
    - Load weights dari BEST_WEIGHTS_PATH
    - Cache di memori untuk request berikutnya
    """
    global _model
    if _model is None:
        model = build_classification_model()
        model.load_weights(Config.BEST_WEIGHTS_PATH)
        _model = model
    return _model
