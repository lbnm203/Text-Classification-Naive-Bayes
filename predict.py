import numpy as np
import preprocessing as pr


def predict(text, model, dictionary, le):
    processed_text = pr.preprocess_text(text)
    features = pr.create_features(processed_text, dictionary)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    prediction_cls = le.inverse_transform(prediction)[0]
    return prediction_cls
