import pickle

import numpy as np
from flask import jsonify
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from api.preparation import extract_filenames
from api.preprocess import get_shrinked_img


def create_logreg_model():
    """ロジスティック回帰の学習済みモデルを生成"""
    digits = load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    logreg = LogisticRegression(max_iter=2000)
    logreg_model = logreg.fit(X_train, y_train)
    return logreg_model


def evaluate_probs(request) -> tuple:
    """テストデータを利用してロジスティック回帰の学習済みモデルのアウトプットを評価"""
    file_id = request.json["file_id"]
    filenames = extract_filenames(file_id)
    img_test = get_shrinked_img(filenames)

    model = create_logreg_model()

    X_true = [int(filename[:1]) for filename in filenames]
    X_true = np.array(X_true)

    predicted_result = model.predict(img_test).tolist()
    accuracy = model.score(img_test, X_true).tolist()
    observed_result = X_true.tolist()

    return jsonify(
        {
            "results": {
                "file_id": file_id,
                "observed_result": observed_result,
                "predicted_result": predicted_result,
                "accuracy": accuracy,
            }
        },
        201,
    )
