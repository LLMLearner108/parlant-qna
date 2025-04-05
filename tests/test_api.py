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

from collections.abc import AsyncIterator
import httpx
from pytest import fixture

from parlant_qna.app import create_transient_app
from parlant_qna.server import create_server, GLOBAL_TAG
from tests.test_utilities import nlp_test


@fixture
async def api() -> AsyncIterator[httpx.AsyncClient]:
    port = 8899

    async with create_transient_app() as app:
        async with create_server(port=port, qna_app=app) as server:
            print("server started")
            async with httpx.AsyncClient(
                base_url=f"http://localhost:{port}",
                follow_redirects=True,
                timeout=60,
            ) as client:
                yield client
            await server.shutdown()


async def test_crud(api: httpx.AsyncClient) -> None:
    response = await api.post(
        "/questions",
        json={
            "variants": ["Who is Bubble Joe"],
            "answer": "He is the king of Bubbleland",
        },
    )

    question_id = response.json()["question_id"]

    response = await api.get(f"/questions/{question_id}")

    question = response.json()

    assert len(question["variants"]) == 1
    assert question["variants"][0] == "Who is Bubble Joe"
    assert question["answer"] == "He is the king of Bubbleland"

    await api.patch(
        f"/questions/{question_id}",
        json={
            "variants": ["Who is Bubble Joseph"],
            "answer": "He is the prince of Bubbleland",
        },
    )

    response = await api.get(f"/questions/{question_id}")

    question = response.json()

    assert len(question["variants"]) == 1
    assert question["variants"][0] == "Who is Bubble Joseph"
    assert question["answer"] == "He is the prince of Bubbleland"

    response = await api.get("/questions")

    questions = response.json()

    # MODIFIED: Earlier assertion was question in questions which failed because the response schema for get a single question vs get all questions is different
    assert len(questions) == 1
    all_question_ids = [q["id"] for q in questions]
    assert question["id"] in all_question_ids

    await api.delete(f"/questions/{question_id}")

    response = await api.get("/questions")

    questions = response.json()

    assert len(questions) == 0


async def test_that_a_question_is_given_a_global_tag_when_no_tags_are_provided(
    api: httpx.AsyncClient,
) -> None:
    response = await api.post(
        "/questions",
        json={
            "variants": ["What is Banana"],
            "answer": "Banana is a fruit",
        },
    )
    print(response.json())
    question_id = response.json()["question_id"]
    response = await api.get(f"/questions/{question_id}")
    question = response.json()
    assert question["tags"] == [GLOBAL_TAG]
    assert len(question["tags"]) == 1


async def test_that_a_question_can_be_answered_when_only_global_questions_have_context_and_selected_tag_is_None(
    api: httpx.AsyncClient,
) -> None:
    # Create a question without a tag: i.e. a global question
    _ = await api.post(
        "/questions",
        json={
            "variants": ["What is Banana"],
            "answer": "Banana is a fruit",
        },
    )

    # Create another question with a custom tag
    _ = await api.post(
        "/questions",
        json={
            "variants": ["How does a banana taste?"],
            "answer": "Bananas are yummy! They taste sweet and delicious.",
            "tags": ["Banana"],
        },
    )

    # Ask a question that can be answered using the global question
    question = "Can you tell what a banana is?"
    response = await api.post(
        "/answers",
        json={"query": question},
    )

    assert await nlp_test(
        context=f"Question: {question} ;; Answer: {response.json()['answer']}",
        condition="The answer to the question is banana is a fruit",
    )

    assert response.json()["answer"]


async def test_that_a_question_can_be_answered_when_only_global_questions_have_context_and_selected_tag_is_empty_list(
    api: httpx.AsyncClient,
) -> None:
    # Create a question without a tag: i.e. a global question
    _ = await api.post(
        "/questions",
        json={
            "variants": ["What is Banana"],
            "answer": "Banana is a fruit",
        },
    )

    # Create another question with a custom tag
    _ = await api.post(
        "/questions",
        json={
            "variants": ["How does a banana taste?"],
            "answer": "Bananas are yummy! They taste sweet and delicious.",
            "tags": ["Banana"],
        },
    )

    # Ask a question that can be answered using the global question
    question = "Can you tell what a banana is?"
    response = await api.post(
        "/answers",
        json={"query": question, "tags": []},
    )

    assert await nlp_test(
        context=f"Question: {question} ;; Answer: {response.json()['answer']}",
        condition="The answer to the question is banana is a fruit",
    )

    assert response.json()["answer"]


async def test_that_a_question_can_be_answered_when_only_global_questions_have_context_and_selected_tag_is_a_custom_tag(
    api: httpx.AsyncClient,
) -> None:
    # Create a question without a tag: i.e. a global question
    _ = await api.post(
        "/questions",
        json={
            "variants": ["What is Banana"],
            "answer": "Banana is a fruit",
        },
    )

    # Create another question with a custom tag
    _ = await api.post(
        "/questions",
        json={
            "variants": ["How does a banana taste?"],
            "answer": "Bananas are yummy! They taste sweet and delicious.",
            "tags": ["Banana"],
        },
    )

    # Ask a question that can be answered using the global question
    question = "Can you tell what a banana is?"
    response = await api.post(
        "/answers",
        json={"query": question, "tags": ["Banana"]},
    )

    assert await nlp_test(
        context=f"Question: {question} ;; Answer: {response.json()['answer']}",
        condition="The answer to the question is banana is a fruit",
    )

    assert response.json()["answer"]


async def test_that_a_question_can_be_answered_when_only_global_questions_have_context_and_selected_tag_is_a_custom_tag(
    api: httpx.AsyncClient,
) -> None:

    # Create an untagged question
    _ = await api.post(
        "/questions",
        json={
            "variants": ["How does a banana taste?"],
            "answer": "Bananas are yummy! They taste sweet and delicious.",
        },
    )

    # Create a question without one custom tag
    _ = await api.post(
        "/questions",
        json={
            "variants": ["What is Banana"],
            "answer": "Banana is a fruit",
            "tags": ["Banana"],
        },
    )

    # Ask a question that can be answered using the global question
    question = "Can you tell what a banana is?"
    response = await api.post(
        "/answers",
        json={"query": question, "tags": ["Banana"]},
    )

    assert await nlp_test(
        context=f"Question: {question} ;; Answer: {response.json()['answer']}",
        condition="The answer to the question is banana is a fruit",
    )

    assert response.json()["answer"]


async def test_that_a_question_can_be_answered_when_global_questions_do_not_have_context_but_custom_question_has_context_and_selected_tag_is_a_custom_tag(
    api: httpx.AsyncClient,
) -> None:
    # Create an untagged question
    _ = await api.post(
        "/questions",
        json={
            "variants": ["How does a banana taste?"],
            "answer": "Bananas are yummy! They taste sweet and delicious.",
        },
    )

    # Create a question without one custom tag
    _ = await api.post(
        "/questions",
        json={
            "variants": ["What is Banana"],
            "answer": "Banana is a fruit",
            "tags": ["Banana"],
        },
    )

    # Ask a question that can be answered using the global question
    question = "Can you tell what a banana is?"
    response = await api.post(
        "/answers",
        json={"query": question, "tags": ["Banana"]},
    )

    assert response.json()["answer"]
    assert await nlp_test(
        context=f"Question: {question} ;; Answer: {response.json()['answer']}",
        condition="The answer to the question is banana is a fruit",
    )


async def test_that_a_question_cannot_be_answered_when_only_questions_tagged_customTagX_have_context_and_selected_tag_is_customTagY(
    api: httpx.AsyncClient,
) -> None:

    # Create an untagged question
    _ = await api.post(
        "/questions",
        json={
            "variants": ["What is Banana Republic?"],
            "answer": "Banana Republic is a clothing brand.",
        },
    )

    # Create a question without one custom tag
    _ = await api.post(
        "/questions",
        json={
            "variants": ["What is Banana"],
            "answer": "Banana is a fruit",
            "tags": ["Banana"],
        },
    )

    # Create another question with a custom tag
    _ = await api.post(
        "/questions",
        json={
            "variants": ["How does a banana taste?"],
            "answer": "Bananas are yummy! They taste sweet and delicious.",
            "tags": ["Cooking"],
        },
    )

    # Ask a question that can be answered using the global question
    question = "Can you tell what a banana is?"
    response = await api.post(
        "/answers",
        json={"query": question, "tags": ["Cooking"]},
    )

    assert not response.json()["answer"]


async def test_that_a_question_cannot_be_answered_when_global_questions_do_not_have_context_and_selected_tag_is_empty_list(
    api: httpx.AsyncClient,
) -> None:
    # Create an untagged question
    _ = await api.post(
        "/questions",
        json={
            "variants": ["How does a banana taste?"],
            "answer": "Bananas are yummy! They taste sweet and delicious.",
        },
    )

    # Create a question without one custom tag
    _ = await api.post(
        "/questions",
        json={
            "variants": ["What is Banana"],
            "answer": "Banana is a fruit",
            "tags": ["Banana"],
        },
    )

    # Ask a question that can be answered using the global question
    question = "Can you tell what a banana is?"
    response = await api.post(
        "/answers",
        json={"query": question, "tags": []},
    )

    assert not response.json()["answer"]


async def test_with_find_answer_tool_that_a_question_can_be_answered_when_context_exists_in_a_tagged_question_and_that_tag_is_selected_(
    api: httpx.AsyncClient,
) -> None:
    _ = await api.post(
        "/questions",
        json={
            "variants": ["How does a banana taste?"],
            "answer": "Bananas are yummy! They taste sweet and delicious.",
        },
    )

    # Create a question without one custom tag
    _ = await api.post(
        "/questions",
        json={
            "variants": ["What is Banana"],
            "answer": "Banana is a fruit",
            "tags": ["Banana"],
        },
    )

    # Ask a question that can be answered using the global question
    question = "Can you tell what a banana is?"
    response = await api.post(
        "/tools/find_answer/calls",
        json={
            "agent_id": "<test-agent>",
            "session_id": "<test-session>",
            "customer_id": "<test-customer>",
            "arguments": {
                "query": question,
                "tags": ["Banana"],
            },
        },
    )

    answer = response.json()["data"]
    assert answer
    assert await nlp_test(
        context=f"Question: {question} ;; Answer: {answer}",
        condition="The answer to the question is banana is a fruit",
    )


async def test_with_find_answer_tool_that_a_question_cannot_be_answered_when_context_exists_in_a_tagged_question_and_that_tag_is_not_selected_(
    api: httpx.AsyncClient,
) -> None:
    _ = await api.post(
        "/questions",
        json={
            "variants": ["How does a banana taste?"],
            "answer": "Bananas are yummy! They taste sweet and delicious.",
        },
    )

    # Create a question without one custom tag
    _ = await api.post(
        "/questions",
        json={
            "variants": ["What is Banana"],
            "answer": "Banana is a fruit",
            "tags": ["Banana"],
        },
    )

    # Ask a question that can be answered using the global question
    question = "Can you tell what a banana is?"
    response = await api.post(
        "/tools/find_answer/calls",
        json={
            "agent_id": "<test-agent>",
            "session_id": "<test-session>",
            "customer_id": "<test-customer>",
            "arguments": {
                "query": question,
                "tags": [],
            },
        },
    )

    answer = response.json()["data"]
    assert not answer
