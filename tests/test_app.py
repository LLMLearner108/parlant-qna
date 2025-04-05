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
from parlant_qna.server import GLOBAL_TAG


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


async def test_parlant_qna(parlant_qna: App) -> None:
    answer = await parlant_qna.ask_question(
        "How do you compare guidelines to systems with flows and intents?"
    )

    assert answer.grade != "no-answer"


# ASK: Now everywhere I am assuming that we will be providing the GLOBAL_TAG when we see that the user has not asked to filter by tags, isn't it?
async def test_that_a_question_thats_not_added_cannot_be_answered(app: App) -> None:
    await app.create_question(
        variants=["Who plays George on Seinfeld"],
        answer="Jason Alexander",
        tags=[GLOBAL_TAG],
    )

    answer = await app.ask_question("what is 1+1", tags=[GLOBAL_TAG])

    assert answer.grade == "no-answer"
    assert not answer.content


# Renamed the test from added to answered
# Interesting test case now after the creation of the global tag ASK: If we should change the filter_questions function
async def test_that_a_question_can_be_answered_based_on_background_info(
    app: App,
) -> None:
    await app.create_question(
        variants=["Who plays George on Seinfeld"],
        answer="Jason Alexander played in Seinfeld from 1989 (he was 29 years old) until the end of the show in 1998 (when we was 38 years old)",
    )

    question = "How old was Jason Alexandar when Seinfeld ended?"
    answer = await app.ask_question(question)

    assert answer.grade == "full"
    assert answer.content
    assert await nlp_test(
        context=f"Question: {question} ;; Answer: {answer.content}",
        condition="The answer to the question is 38",
    )


async def test_report(parlant_qna: App) -> None:
    report_id = await parlant_qna.create_report(sample_percentage=10)

    while (report := await parlant_qna.read_report(report_id)).status == "running":
        await asyncio.sleep(5)
        print(len(report.samples))

    print(report)


async def test_that_a_question_can_be_answered_when_context_exists_in_a_tagged_question_and_that_tag_is_selected(
    app: App,
) -> None:
    
    question = "Who plays George on Seinfeld"
    answer_string = "Jason Alexander"

    await app.create_question(
        variants=[question],
        answer=answer_string,
        tags=["Seinfeld"],
    )

    answer = await app.ask_question("Who plays George on Seinfeld", tags=["Seinfeld"])

    assert answer.grade == "full"
    assert answer.content
    assert await nlp_test(
        context=f"Question: {question} ;; Answer: {answer.content}",
        condition="The answer to the question is Jason Alexander",
    )

async def test_that_a_question_cannot_be_answered_when_context_exists_in_a_tagged_question_and_that_tag_is_not_selected(
    app: App,
) -> None:
    
    question = "Who plays George on Seinfeld"
    answer_string = "Jason Alexander"

    await app.create_question(
        variants=[question],
        answer=answer_string,
        tags=["Seinfeld"],
    )

    question = "Who plays Dr. House on House MD"
    answer_string = "Hugh Laurie"

    await app.create_question(
        variants=[question],
        answer=answer_string,
        tags=["House"],
    )

    answer = await app.ask_question("Who plays George on Seinfeld", tags=["House"])

    assert answer.grade == "no-answer"
    assert not answer.content

