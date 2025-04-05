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

import asyncio
from collections.abc import AsyncIterator
import os
from pathlib import Path
from pytest import fixture
from parlant_qna.app import App, Question, create_transient_app, parse_md_file
from tests.test_utilities import nlp_test


@fixture
async def parlant_questions() -> list[Question]:
    dir = Path("example_qna")
    questions: list[Question] = []

    for file in os.listdir(dir):
        questions.append(await parse_md_file(dir / file))

    return questions


@fixture
async def app() -> AsyncIterator[App]:
    async with create_transient_app() as app:
        yield app


@fixture
async def parlant_qna(app: App, parlant_questions: list[Question]) -> App:
    for q in parlant_questions:
        await app.create_question(q.variants, q.answer)

    return app


async def test_that_question_tagged_with_customTagX_returned_when_selected_tag_is_customTagX(
    app: App,
) -> None:
    await app.create_question(
        variants=["Who plays George on Seinfeld"],
        answer="Jason Alexander",
        tags=["Seinfeld"],
    )

    await app.create_question(
        variants=["Which actor played Richard Hendricks on Silicon Valley?"],
        answer="Thomas Middleditch",
        tags=["Silicon Valley"],
    )

    filtered_questions = app._question_filter.filter_questions(
        app._questions, ["Seinfeld"]
    )

    assert len(filtered_questions) == 1

    question = list(filtered_questions.values())[0]
    assert question.variants[0] == "Who plays George on Seinfeld"
    assert question.answer == "Jason Alexander"
    assert question.tags == ["Seinfeld"]


async def test_that_question_tagged_with_customTagX_are_not_returned_when_selected_tag_is_not_customTagX(
    app: App,
) -> None:
    await app.create_question(
        variants=["Who plays George on Seinfeld"],
        answer="Jason Alexander",
        tags=["Seinfeld"],
    )

    await app.create_question(
        variants=["Which actor played Richard Hendricks on Silicon Valley?"],
        answer="Thomas Middleditch",
        tags=["Silicon Valley"],
    )

    filtered_questions = app._question_filter.filter_questions(
        questions=app._questions, tags=["Silicon Valley"]
    )

    assert "Seinfeld" not in list(filtered_questions.values())[0].tags


async def test_that_no_questions_are_returned_when_no_questions_exist_with_selected_tag(
    app: App,
) -> None:
    await app.create_question(
        variants=["Who plays George on Seinfeld"],
        answer="Jason Alexander",
        tags=["Seinfeld"],
    )

    await app.create_question(
        variants=["Which actor played Richard Hendricks on Silicon Valley?"],
        answer="Thomas Middleditch",
        tags=["Silicon Valley"],
    )

    filtered_questions = app._question_filter.filter_questions(
        questions=app._questions, tags=["House MD"]
    )

    assert len(filtered_questions) == 0


async def test_that_untagged_questions_are_returned_when_no_tags_are_selected(
    app: App,
) -> None:
    await app.create_question(
        variants=["Who plays George on Seinfeld"],
        answer="Jason Alexander",
        tags=["Seinfeld"],
    )

    await app.create_question(
        variants=["Which actor played Richard Hendricks on Silicon Valley?"],
        answer="Thomas Middleditch",
        tags=["Silicon Valley"],
    )

    await app.create_question(
        variants=["Which actor played Dr. House on House MD?"],
        answer="Hugh Laurie",
    )

    filtered_questions = app._question_filter.filter_questions(
        questions=app._questions, tags=[]
    )

    assert len(filtered_questions) == 1

    question = list(filtered_questions.values())[0]
    assert question.variants[0] == "Which actor played Dr. House on House MD?"
    assert question.answer == "Hugh Laurie"
