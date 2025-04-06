# Copyright 2024 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterator
from pytest import fixture
import logging
from parlant_qna.app import QuestionFilter, Question, TagBasedQuestionFilter


@fixture
def question_filter() -> Iterator[QuestionFilter]:
    logger = logging.getLogger("test_filter")
    logger.setLevel(logging.DEBUG)
    question_filter = TagBasedQuestionFilter(logger=logger)
    yield question_filter


async def test_that_question_tagged_with_customTagX_returned_when_selected_tag_is_customTagX(
    question_filter: QuestionFilter,
) -> None:

    question_a = Question(
        id="2e9834",
        variants=["Who plays George on Seinfeld"],
        answer="Jason Alexander",
        tags=["Seinfeld"],
    )

    question_b = Question(
        id="2e9835",
        variants=["Which actor played Richard Hendricks on Silicon Valley?"],
        answer="Thomas Middleditch",
        tags=["Silicon Valley"],
    )

    question_dict = {"2e9834": question_a, "2e9835": question_b}

    filtered_questions = question_filter.filter_questions(question_dict, ["Seinfeld"])

    assert len(filtered_questions) == 1

    question = list(filtered_questions.values())[0]
    assert question.variants[0] == "Who plays George on Seinfeld"
    assert question.answer == "Jason Alexander"
    assert question.tags == ["Seinfeld"]


async def test_that_question_tagged_with_customTagX_are_not_returned_when_selected_tag_is_not_customTagX(
    question_filter: QuestionFilter,
) -> None:

    question_a = Question(
        id="2e9834",
        variants=["Who plays George on Seinfeld"],
        answer="Jason Alexander",
        tags=["Seinfeld"],
    )

    question_b = Question(
        id="2e9835",
        variants=["Which actor played Richard Hendricks on Silicon Valley?"],
        answer="Thomas Middleditch",
        tags=["Silicon Valley"],
    )

    question_dict = {"2e9834": question_a, "2e9835": question_b}

    filtered_questions = question_filter.filter_questions(
        questions=question_dict, tags=["Silicon Valley"]
    )

    assert "Seinfeld" not in list(filtered_questions.values())[0].tags


async def test_that_no_questions_are_returned_when_no_questions_exist_with_selected_tag(
    question_filter: QuestionFilter,
) -> None:

    question_a = Question(
        id="2e9834",
        variants=["Who plays George on Seinfeld"],
        answer="Jason Alexander",
        tags=["Seinfeld"],
    )

    question_b = Question(
        id="2e9835",
        variants=["Which actor played Richard Hendricks on Silicon Valley?"],
        answer="Thomas Middleditch",
        tags=["Silicon Valley"],
    )

    question_dict = {"2e9834": question_a, "2e9835": question_b}

    filtered_questions = question_filter.filter_questions(
        questions=question_dict, tags=["House MD"]
    )

    assert len(filtered_questions) == 0


async def test_that_untagged_questions_are_returned_when_no_tags_are_selected(
    question_filter: QuestionFilter,
) -> None:

    question_a = Question(
        id="2e9834",
        variants=["Who plays George on Seinfeld"],
        answer="Jason Alexander",
        tags=["Seinfeld"],
    )

    question_b = Question(
        id="2e9835",
        variants=["Who played Richard Hendricks on Silicon Valley?"],
        answer="Thomas Middleditch",
        tags=["Silicon Valley"],
    )

    question_c = Question(
        id="2e9836",
        variants=["Which actor played Dr. House on House MD?"],
        answer="Hugh Laurie",
        tags=[],
    )

    question_dict = {"2e9834": question_a, "2e9835": question_b, "2e9836": question_c}

    filtered_questions = question_filter.filter_questions(
        questions=question_dict, tags=[]
    )

    assert len(filtered_questions) == 1

    question = list(filtered_questions.values())[0]
    assert question.variants[0] == "Which actor played Dr. House on House MD?"
    assert question.answer == "Hugh Laurie"
