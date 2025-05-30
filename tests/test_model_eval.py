import pytest

from src.config import ConfigurationManager
from src.models import MultipleChoiceLLM

config = ConfigurationManager(config_file_path="config.yaml")
model_configs = config.get_model_configuration()

test_questions = [
    ("Which one is the capital of Canada? A.Paris B.Ottawa C.Istanbul D.Toronto", "B"),
    ("Which planet is known as the Red Planet? A.Mars B.Earth C.Jupiter D.Venus", "A"),
    (
        "Which language is primarily spoken in Brazil? A.Spanish B.French C.Portuguese D.English",
        "C",
    ),
    ("Which one is the largest mammal? A.Lion B.Human C.Blue whale D.Elephant", "C"),
]


@pytest.fixture(scope="module")
def model():
    return MultipleChoiceLLM(
        model_name=model_configs.model_name,
        allowed_choices=model_configs.allowed_choices,
    )


@pytest.mark.parametrize("question, expected_answer", test_questions)
def test_model_predictions(model, question, expected_answer):
    prediction = model.predict(question)
    assert prediction.strip().upper() == expected_answer.upper(), (
        f"Prediction '{prediction}' does not match expected '{expected_answer}' for question: {question}"
    )
