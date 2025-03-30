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
async def test_that_a_question_can_be_answered_based_on_background_info(app: App) -> None:
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

async def test_that_no_questions_are_retrieved_when_tags_list_is_None(
    app: App,
) -> None:
    await app.create_question(
        variants=["What is a Banana?"], answer="Banana is a fruit", tags=[GLOBAL_TAG]
    )

    await app.create_question(
        variants=["How does Aamras taste?"],
        answer="Aamras is an extremely sweet and yummy desert",
        tags=["Mango"],
    )

    question = "Explain the taste of Aamras"
    questions_used = app._filter_questions(tags=None)
    answer = await app.ask_question(question, tags=None)

    assert len(questions_used) == 0
    qns = list(questions_used.values())
    qn_variants = sorted([x.variants[0] for x in qns])
    assert len(qn_variants) == 0

    assert answer.grade == "no-answer"
    assert not answer.content

async def test_that_no_questions_are_retrieved_when_tags_list_is_empty_list(
    app: App,
) -> None:
    await app.create_question(
        variants=["What is a Banana?"], answer="Banana is a fruit", tags=[GLOBAL_TAG]
    )

    await app.create_question(
        variants=["How does Aamras taste?"],
        answer="Aamras is an extremely sweet and yummy desert",
        tags=["Mango"],
    )

    question = "Explain the taste of Aamras"
    questions_used = app._filter_questions(tags=[])
    answer = await app.ask_question(question, tags=[])

    assert len(questions_used) == 0
    qns = list(questions_used.values())
    qn_variants = sorted([x.variants[0] for x in qns])
    assert len(qn_variants) == 0

    assert answer.grade == "no-answer"
    assert not answer.content

async def test_that_only_global_questions_are_in_context_when_tags_list_has_global_tag_and_query_cannot_be_answered_using_global_questions(
    app: App,
) -> None:
    await app.create_question(
        variants=["What is a Banana?"], answer="Banana is a fruit", tags=[GLOBAL_TAG]
    )

    await app.create_question(
        variants=["How does Aamras taste?"],
        answer="Aamras is an extremely sweet and yummy desert",
        tags=["Mango"],
    )

    question = "Explain the taste of Aamras"
    questions_used = app._filter_questions(tags=[GLOBAL_TAG])
    answer = await app.ask_question(question, tags=[GLOBAL_TAG])

    assert len(questions_used) == 1
    qns = list(questions_used.values())
    qn_variants = sorted([x.variants[0] for x in qns])
    assert qn_variants == ["What is a Banana?"]

    assert answer.grade == "no-answer"
    assert not answer.content

async def test_that_only_global_questions_are_in_context_when_tags_list_has_global_tag_and_query_can_be_answered_using_global_questions(
    app: App,
) -> None:
    await app.create_question(
        variants=["What is a Banana?"], answer="Banana is a fruit", tags=[GLOBAL_TAG]
    )

    await app.create_question(
        variants=["How does Aamras taste?"],
        answer="Aamras is an extremely sweet and yummy desert",
        tags=["Mango"],
    )

    question = "Tell me about bananas"
    questions_used = app._filter_questions(tags=[GLOBAL_TAG])
    answer = await app.ask_question(question, tags=[GLOBAL_TAG])

    assert len(questions_used) == 1
    qns = list(questions_used.values())
    qn_variants = sorted([x.variants[0] for x in qns])
    assert qn_variants == ["What is a Banana?"]

    assert answer.grade == "full"
    assert answer.content
    assert await nlp_test(
        context=f"Question: {question} ;; Answer: {answer.content}",
        condition="The answer to that banana is a fruit",
    )


async def test_that_empty_context_exception_is_raised_when_a_non_existent_tag_is_provided(
    app: App,
) -> None:
    await app.create_question(
        variants=["What is a Banana?"],
        answer="Banana is a fruit",
        tags=[GLOBAL_TAG],
    )

    await app.create_question(
        variants=["How does Aamras taste?"],
        answer="Aamras is an extremely sweet and yummy desert",
        tags=["Mango"],
    )

    question = "Explain the taste of Aamras"
    questions_extracted = app._filter_questions(tags=["Orange"])
    answer = await app.ask_question(question, tags=["Orange"])

    assert len(questions_extracted) == 0

    assert answer.evaluation == "None of the provided tags are present in the database"
    assert answer.grade == "no-answer"
    assert not answer.content


async def test_that_non_zero_questions_are_retreived_and_query_must_be_answered_with_retrieved_questions_when_existing_non_global_tag_is_provided(
    app: App,
) -> None:
    await app.create_question(
        variants=["What is a Banana?"],
        answer="Banana is a fruit",
        tags=[GLOBAL_TAG],
    )

    await app.create_question(
        variants=["How does Aamras taste?"],
        answer="Aamras is an extremely sweet and yummy desert",
        tags=["Mango"],
    )

    question = "Explain the taste of Aamras"
    questions_used = app._filter_questions(tags=["Mango"])
    answer = await app.ask_question(question, tags=["Mango"])

    assert len(questions_used) == 1
    qn = list(questions_used.values())[0]
    assert qn.variants[0] == "How does Aamras taste?"

    assert answer.grade == "full"
    assert answer.content
    assert await nlp_test(
        context=f"Question: {question} ;; Answer: {answer.content}",
        condition="The answer to the question is that Aamras is a sweet desert",
    )


async def test_that_non_zero_questions_are_retreived_and_query_must_not_be_answered_with_retrieved_questions_when_existing_non_global_tag_is_provided(
    app: App,
) -> None:
    await app.create_question(
        variants=["What is a Banana?"],
        answer="Banana is a fruit",
        tags=[GLOBAL_TAG],
    )

    await app.create_question(
        variants=["How does Aamras taste?"],
        answer="Aamras is an extremely sweet and yummy desert",
        tags=["Mango"],
    )

    question = "What is the main ingredient of Aamras?"
    questions_used = app._filter_questions(tags=["Mango"])
    answer = await app.ask_question(question, tags=["Mango"])

    assert len(questions_used) == 1
    qn = list(questions_used.values())[0]
    assert qn.variants[0] == "How does Aamras taste?"

    assert answer.grade == "no-answer"
    assert not answer.content
