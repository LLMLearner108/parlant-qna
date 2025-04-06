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
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from abc import ABC, abstractmethod

# ADDED: Additional field import to initialize the Question document with an empty list
from dataclasses import field
from pathlib import Path
from typing import Any, Literal, Sequence, TypeAlias, cast, Optional, List, Dict

import aiofiles
from parlant.adapters.db.json_file import JSONFileDocumentDatabase
from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.adapters.nlp.openai_service import OpenAIService
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.common import DefaultBaseModel, Version, generate_id
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.loggers import FileLogger, Logger, LogLevel, StdoutLogger
from parlant.core.nlp.generation import GenerationInfo
from parlant.core.nlp.service import NLPService
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import BaseDocument, DocumentDatabase
from typing_extensions import Self


# MODIFIED: Added a field called tags to permit manual filtering of context
@dataclass(frozen=True)
class Question:
    id: str
    variants: list[str]
    answer: str
    tags: list[str] = field(default_factory=list)


async def parse_md_file(file: Path) -> Question:
    async with aiofiles.open(file) as f:
        variants: list[str] = []

        lines = await f.readlines()

        content_start = 0

        for line in lines:
            if not line.strip():
                content_start += 1
                continue
            if not line.startswith("# "):
                break

            variants.append(line[1:].strip())
            content_start += 1

        content_lines = lines[content_start:]
        content = "".join(content_lines).strip()

    return Question(
        id="",
        variants=variants,
        answer=content,
    )


AnswerGrade: TypeAlias = Literal["partial", "full", "no-answer"]


# MODIFIED: Added a field called tags to permit manual filtering of context
class _QuestionDocument(BaseDocument):
    variants: list[str]
    answer: str
    tags: list[str] = field(default_factory=list)


class _RelevantQuotes(DefaultBaseModel):
    question_id: str
    quotes: list[str]


class _AnswerSchema(DefaultBaseModel):
    user_questions: list[str]
    relevant_question_variants: Optional[list[str]] = None
    full_answer_can_be_found_in_background_info: Optional[str] = None
    partial_answer_can_be_found_in_background_info: Optional[str] = None
    insights_on_what_could_be_a_legitimate_answer: Optional[str] = None
    collected_relevant_quotes_from_background_info: Optional[list[_RelevantQuotes]] = (
        None
    )
    concise_and_minimal_synthesized_answer_based_solely_on_relevant_quotes__draft: (
        Optional[str]
    ) = (None)
    critique: Optional[str] = None
    brief_explanation_of_what_needs_to_change_in_order_to_stay_within_the_boundaries_of_collected_quotes: Optional[
        str
    ] = None
    could_use_better_markdown: Optional[bool] = None
    concise_and_minimal_synthesized_answer_based_solely_on_relevant_quotes__revised: (
        Optional[str]
    ) = (None)
    extracted_entities_found_in_background_info_and_referred_to_by_answer: Optional[
        list[str]
    ] = None
    question_answered_in_full: bool
    question_answered_partially: bool
    question_not_answered_at_all: bool


class _TestSchema(DefaultBaseModel):
    insights_and_evaluation_on_the_generated_answer_compared_to_the_original: str
    does_the_generated_answer_contain_hallucinations: bool
    detected_hallucination_explanation: Optional[str] = None
    does_the_generated_answer_contain_any_facts_that_are_not_given_in_the_original_question_and_answer: (
        bool
    )
    does_the_generated_answer_provide_a_full_answer: bool
    does_the_generated_answer_provide_a_partial_answer: bool


class _VariantSchema(DefaultBaseModel):
    insights_on_different_scenarios_and_angles_from_which_to_ask_the_same_question: str
    variant_1: str
    variant_2: str
    variant_3: str


@dataclass(frozen=True)
class Reference:
    question_id: str
    quotes: list[str]


@dataclass(frozen=True)
class Answer:
    content: Optional[str]
    grade: AnswerGrade
    generation_info: Optional[GenerationInfo]
    evaluation: str
    references: list[Reference]
    extracted_entities: list[str]


@dataclass(frozen=True)
class ReportSample:
    question_id: str
    variant: str
    answer: Answer
    evaluation: str
    references_check_out: bool
    hallucination: Optional[str]

    @property
    def score(self) -> float:
        if self.answer.grade == "no-answer":
            return 0

        if self.hallucination:
            return 0

        if self.answer.grade == "partial":
            if self.references_check_out:
                return 0.5
            else:
                return 0.1

        if not self.references_check_out:
            # We want to severely penalize for this, but still
            # give some credit for getting a full answer right.
            return 0.25

        return 1

    @property
    def true_positive(self) -> bool:
        return self.answer.grade != "no-answer" and not self.hallucination

    @property
    def true_negative(self) -> bool:
        return False  # There shouldn't be any in our test

    @property
    def false_positive(self) -> bool:
        if self.answer.grade != "no-answer":
            return self.hallucination is not None
        return False

    @property
    def false_negative(self) -> bool:
        return self.answer.grade == "no-answer"


@dataclass
class Report:
    expected_samples: int
    status: Literal["running", "failed", "completed"]
    samples: list[ReportSample]

    @property
    def true_positives(self) -> float:
        return len([s for s in self.samples if s.true_positive])

    @property
    def true_negatives(self) -> int:
        return len([s for s in self.samples if s.true_negative])

    @property
    def false_positives(self) -> int:
        return len([s for s in self.samples if s.false_positive])

    @property
    def false_negatives(self) -> int:
        return len([s for s in self.samples if s.false_negative])

    @property
    def precision(self) -> float:
        return (self.true_positives) / max(
            1, (self.true_positives + self.false_positives)
        )

    @property
    def recall(self) -> float:
        return (self.true_positives) / max(
            1, (self.true_positives + self.false_negatives)
        )

    @property
    def accuracy(self) -> float:
        return (self.true_positives + self.true_negatives) / max(1, len(self.samples))

    @property
    def f1(self) -> float:
        return (
            2
            * (self.precision * self.recall)
            / max(0.00001, (self.precision + self.recall))
        )


class QNABackgroundTaskService(BackgroundTaskService):
    # Inherited to change the class name (which is referred to from logs) for cleaner logs
    pass



# ADDED: Abstracted the filtering of questions based on tags to a separate class
class QuestionFilter(ABC):
    @abstractmethod
    def filter_questions(
        self, questions: Dict[str, Question], tags: Optional[list[str]] = None
    ) -> dict[str, Question]:
        ...

# Question Filter implementation can be extended to include more sophisticated filtering logic like when we introduce auto tags etc.
class TagBasedQuestionFilter(QuestionFilter):
    def __init__(self, logger: Logger):
        self.logger = logger

    def filter_questions(
        self, questions: Dict[str, Question], tags: Optional[list[str]] = None
    ) -> Dict[str, Question]:
        """
        Filter questions based on provided tags.

        Args:
            questions: Dictionary of question_id to Question objects
            tags: Optional list of tags to filter by. If None or empty, returns untagged questions.

        Returns:
            Dictionary of filtered questions
        """
        filtered_questions = {}

        self.logger.debug(
            f"\033[33m[QuestionFilter.filter_questions] Tags received for filtering: {tags}\033[0m"
        )

        # Fallback: If there are no tags provided for filtering, then use questions that do not have any tags
        if (tags is None) or len(tags) == 0:
            self.logger.debug(
                f"\033[33m[QuestionFilter.filter_questions] Using questions that are untagged\033[0m"
            )

            # Filter the questions which are untagged
            for question_id, question in questions.items():
                if len(question.tags) == 0:
                    filtered_questions[question_id] = question

        # If tags are actually provided for filtering the questions
        else:
            self.logger.debug(
                f"\033[33m[QuestionFilter.filter_questions] Received tags as a non-empty list. Filtering the questions based on tags\033[0m"
            )

            for question_id, question in questions.items():
                question_tags = question.tags
                common_tags = set(question_tags) & set(tags)
                if len(common_tags) > 0:
                    filtered_questions[question_id] = question

        self.logger.debug(
            f"\033[33m[QuestionFilter.filter_questions] Before filtering: {len(questions)} | After filtering: {len(filtered_questions)}\033[0m"
        )

        # Print the questions along with their variants to the logger
        qns = "\n\n".join(["\n".join(q.variants) for q in filtered_questions.values()])
        self.logger.debug(f"\033[33mThese are the filtered questions:\n{qns}\033[0m")

        return filtered_questions


class App:
    VERSION = Version.String("0.1.0")

    def __init__(
        self,
        database: DocumentDatabase,
        service: NLPService,
        logger: Logger,
        question_filter: QuestionFilter | None = None,
    ):
        self._db = database
        self._service = service
        self.logger = logger
        self._question_filter = question_filter or TagBasedQuestionFilter(logger)

        self._questions: dict[str, Question] = {}
        self._reports: dict[str, Report] = {}
        self._task_service = QNABackgroundTaskService(logger)
        self._report_lock = asyncio.Lock()

    async def identity_loader(self, doc: BaseDocument) -> _QuestionDocument:
        return cast(_QuestionDocument, doc)

    async def __aenter__(self, *args: Any, **kwargs: Any) -> Self:
        self._collection = await self._db.get_or_create_collection(
            "questions",
            schema=_QuestionDocument,
            document_loader=self.identity_loader,
        )

        self._generator = await self._service.get_schematic_generator(_AnswerSchema)
        self._test_generator = await self._service.get_schematic_generator(_TestSchema)
        self._variant_generator = await self._service.get_schematic_generator(
            _VariantSchema
        )

        persisted_questions = await self._collection.find({})

        for q in persisted_questions:
            assert "id" in q

            self._questions[q["id"]] = Question(
                id=q["id"], variants=q["variants"], answer=q["answer"], tags=q["tags"]
            )

        self._task_service = await self._task_service.__aenter__()

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> bool:
        return await self._task_service.__aexit__(exc_type, exc_value, traceback)

    # MODIFIED: Additional argument called tags to allow the user to pass a set of tags and the ask question method will only use the questions tagged with these set of tags as relevant context
    async def ask_question(
        self, question: str, tags: Optional[list[str]] = None
    ) -> Answer:
        self.logger.debug(
            f'\033[33m[ask_question] Looking up the answer for this user provided question: \n"{question}"\033[0m'
        )

        # If none of the provided tags is in the list of tags, return and do not give any answer. If some of the tags are present, then return without any answer
        all_tags = await self.list_tags()
        if tags:
            if len(set(tags) & set(all_tags)) == 0:
                return Answer(
                    content=None,
                    evaluation="None of the provided tags are present in the database",
                    grade="no-answer",
                    generation_info=None,
                    references=[],
                    extracted_entities=[],
                )

            absent_tags = set(tags) - set(all_tags)
            self.logger.info(
                f"\033[33m[ask_question] Out of the requested tags, we could not find any questions related to these tags: {absent_tags}. Hence skipping them.\033[0m"
            )

        # MODIFIED: _format_background_info now receives an argument called tags
        background_info = self._format_background_info(tags)

        prompt = f"""\
You are a RAG agent who has exactly one job: to answer the user's question
based ONLY on the background information provided here in-context.

Note that there are cases when data is provided in the answer in a way that's
implied by the question for that answer. For example, if the question is
"What is Blabooshka" and the answer provided is "It's a banana", then
you can infer that "A Blabooshka is a banana".
In this way, the question variants themselves are directly connected to
their answers. Also, often, the answer is to be considered an explicit and
direct continuation of one of the question variants, as if continuing the idea or sentence.
This is only true within a particular question, its variants, and its answer.
It does not apply cross-questions (i.e. the answer to one question is never
a direct continuation of a different question).

IMPORTANT: Try your best to answer the question *fully*, based on the background information provided.

Always attempt to provide the answer in a clean Markdown format,
separating your answer into multiple lines where applicable, for readability,
using Markdown elements like bold text, lists, and tables where applicable.
However, avoid headings, to make your responses more conversational.

Finally, note that, for review and improvement purposes, it's important to capture the quotes
on which you base your answer, as well as any entity you've made reference to.
Examples of entities are (but not limited to): pronouns, products, companies, domain-specific concepts, etc.

You must produce the following report.
- What is the user asking? Is it one question? Is it different ones? Rephrase the user's input approrpiately to better articulate this.
- What question variants from the provided background information contain the answer to each of the user's queries?
- Determine if any or all of the user's queries can be answered (fully or at least partially) solely based on and using the background information provided here exclusively.
- Try to reason about what could be a satisfying answer to the user. Use your generated insight to seek out the most relevant quotes from the background information, making sure to stay exclusively within the bounds of the background information provided here.
- Draft an answer. Make sure it's nicely formatted with Markdown.
- Critique your initial answer to ensure it meets all of the required standards you are given here.
- Draft a final, satisfactory answer, that stays within the strict bounds and standards you are given here.

Produce a JSON object according to the following schema: ###
{{
    "user_questions": [ QUERY_1, ..., QUERY_N ],
    "relevant_question_variants": [ VARIANT_1, ..., VARIANT_N ],
    "full_answer_can_be_found_in_background_info": <"BRIEF EXPLANATION OF WHETHER AND WHY">,
    "partial_answer_can_be_found_in_background_info": <"BRIEF EXPLANATION OF WHETHER AND WHY">,
    "insights_on_what_could_be_a_legitimate_answer": <"YOUR BRIEF INSIGHTS AS TO WHAT COULD BE A LEGITIMATE ANSWER">,
    "collected_relevant_quotes_from_background_info": [
        {{
            "question_id": QUESTION_ID,
            "quotes": [ QUOTE_1, ..., QUOTE_N ]
        }},
        ...
    ],
    "concise_and_minimal_synthesized_answer_based_solely_on_relevant_quotes__draft": <"PRODUCE AN ANSWER HERE EXCLUSIVELY AND ONLY BASED ON THE COLLECTED QUOTES, WITHOUT ADDING ANYTHING ELSE">
    "critique": <"EXPLAIN IF ANY PART OF THE DRAFT IS UNBASED/UNGROUNDED IN BACKGROUND INFO">,
    "brief_explanation_of_what_needs_to_change_in_order_to_stay_within_the_boundaries_of_collected_quotes": <"BRIEF EXPLANATION OF WHAT NEEDS TO CHANGE TO MITIGATE FACTUAL ISSUES">,
    "could_use_better_markdown": <BOOL>,
    "concise_and_minimal_synthesized_answer_based_solely_on_relevant_quotes__revised": <"PRODUCE AN ANSWER HERE EXCLUSIVELY AND ONLY BASED ON THE COLLECTED QUOTES, WITHOUT ADDING ANYTHING ELSE">
    "extracted_entities_found_in_background_info_and_referred_to_by_answer": [ ENTITY_1, ..., ENTITY_N ],
    "question_answered_in_full": <BOOL>,
    "question_answered_partially": <BOOL>,
    "question_not_answered_at_all": <BOOL>
}}
###


Please note that in case you couldn't find any answer (neither full nor partial) within the background information provided here, meaning, you couldn't find specific quotes in the background info, then this is the format you should follow in such a case — note specifically how some of the fields in this case are left as null : ###
{{
    "user_questions": [ QUERY_1, ..., QUERY_N ],
    "relevant_question_variants": [],
    "full_answer_can_be_found_in_background_info": null,
    "partial_answer_can_be_found_in_background_info": null,
    "insights_on_what_could_be_a_legitimate_answer": <"YOUR BRIEF INSIGHTS AS TO WHAT COULD HAVE BEEN A LEGITIMATE ANSWER">,
    "collected_relevant_quotes_from_background_info": [],
    "concise_and_minimal_synthesized_answer_based_solely_on_relevant_quotes__draft": null,
    "critique": null,
    "brief_explanation_of_what_needs_to_change_in_order_to_stay_within_the_boundaries_of_collected_quotes": "N/A",
    "could_use_better_markdown": null,
    "concise_and_minimal_synthesized_answer_based_solely_on_relevant_quotes__revised": null,
    "extracted_entities_found_in_background_info_and_referred_to_by_answer": [ ENTITY_1, ..., ENTITY_N ],
    "question_answered_in_full": false,
    "question_answered_partially": false,
    "question_not_answered_at_all": true
}}
###

Background Information: ###
{background_info}
###

User Question: ###
{question}
###
"""

        result = await self._generator.generate(
            prompt,
            hints={
                "strict": True,
                "temperature": 0.1,
            },
        )

        self.logger.debug(result.content.model_dump_json(indent=2))

        if (
            (
                not result.content.full_answer_can_be_found_in_background_info
                and not result.content.partial_answer_can_be_found_in_background_info
            )
            or not (
                result.content.question_answered_in_full
                or result.content.question_answered_partially
            )
            or result.content.question_not_answered_at_all
            or not result.content.collected_relevant_quotes_from_background_info
        ):
            self.logger.info("No answer")

            return Answer(
                content=None,
                evaluation=result.content.insights_on_what_could_be_a_legitimate_answer
                or "",
                grade="no-answer",
                generation_info=result.info,
                references=[],
                extracted_entities=[],
            )

        final_answer = None

        if (
            result.content.concise_and_minimal_synthesized_answer_based_solely_on_relevant_quotes__revised
        ):
            final_answer = (
                result.content.concise_and_minimal_synthesized_answer_based_solely_on_relevant_quotes__revised
            )
        elif (
            not result.content.brief_explanation_of_what_needs_to_change_in_order_to_stay_within_the_boundaries_of_collected_quotes
        ):
            final_answer = (
                result.content.concise_and_minimal_synthesized_answer_based_solely_on_relevant_quotes__draft
                or None
            )
        elif (
            result.content.brief_explanation_of_what_needs_to_change_in_order_to_stay_within_the_boundaries_of_collected_quotes
            and result.content.concise_and_minimal_synthesized_answer_based_solely_on_relevant_quotes__draft
        ):
            self.logger.warning(
                "Underlying LLM failed to generate a revised answer; falling back to draft"
            )
            final_answer = (
                result.content.concise_and_minimal_synthesized_answer_based_solely_on_relevant_quotes__draft
            )

        answer = Answer(
            content=final_answer
            and final_answer.encode("utf8")
            .decode("unicode-escape")
            .encode("utf16", "surrogatepass")
            .decode("utf16"),
            evaluation=result.content.insights_on_what_could_be_a_legitimate_answer
            or "",
            grade="full" if result.content.question_answered_in_full else "partial",
            generation_info=result.info,
            references=[
                Reference(
                    question_id=q.question_id,
                    quotes=q.quotes,
                )
                for q in result.content.collected_relevant_quotes_from_background_info
            ],
            extracted_entities=result.content.extracted_entities_found_in_background_info_and_referred_to_by_answer
            or [],
        )

        self.logger.info(
            f'Question: "{question}"; Answer ({answer.grade}): "{answer.content}"'
        )

        return answer

    def _format_background_info(self, tags: Optional[list[str]] = None) -> str:
        filtered_questions = self._question_filter.filter_questions(
            self._questions, tags
        )

        if not filtered_questions:
            return "DATA NOT AVAILABLE"

        return "\n\n".join(
            [
                f"""\
Question #{q.id}[variants={q.variants}][[
Answer: {q.answer}
]]
"""
                for q in filtered_questions.values()
            ]
        )

    # MODIFIED: Added tags as a parameter with default value of empty list
    async def create_question(
        self, variants: list[str], answer: str, tags: Optional[list[str]] = []
    ) -> Question:
        new_id = generate_id()

        await self._collection.insert_one(
            _QuestionDocument(
                id=ObjectId(new_id),
                version=self.VERSION,
                variants=variants,
                answer=answer,
                tags=tags,
            )
        )

        question = Question(id=new_id, variants=variants, answer=answer, tags=tags)

        self._questions[question.id] = question

        return question

    # MODIFIED: Added tags as a parameter with default value of empty list
    async def update_question(
        self,
        question_id: str,
        variants: Optional[list[str]] = None,
        answer: Optional[str] = None,
        tags: Optional[list[str]] = [],
    ) -> Question:
        if question_id not in self._questions:
            raise KeyError()

        await self._collection.update_one(
            {"id": {"$eq": question_id}},
            params=cast(
                _QuestionDocument,
                {
                    **({"variants": variants} if variants else {}),
                    **({"answer": answer} if answer else {}),
                    **({"tags": tags} if tags else {}),
                },
            ),
        )

        question = self._questions[question_id]

        self._questions[question_id] = Question(
            id=question.id,
            variants=variants or question.variants,
            answer=answer or question.answer,
            tags=tags or question.tags,
        )

        return await self.read_question(question_id)

    async def read_question(self, question_id: str) -> Question:
        if question_id not in self._questions:
            raise KeyError()

        return self._questions[question_id]

    async def list_questions(self) -> Sequence[Question]:
        return list(self._questions.values())

    # ADDED: A function to list all the tags associated with the questions that we have in the bank so far
    async def list_tags(self) -> List[str]:
        all_tags = set()
        for question in self._questions.values():
            for tag in question.tags:
                all_tags.add(tag)
        return list(all_tags)

    async def delete_question(self, question_id: str) -> bool:
        if question_id in self._questions:
            del self._questions[question_id]
            await self._collection.delete_one({"id": {"$eq": question_id}})
            return True
        return False

    async def read_report(self, report_id: str) -> Report:
        async with self._report_lock:
            if report_id not in self._reports:
                raise KeyError()

            report = self._reports[report_id]

            return Report(
                expected_samples=report.expected_samples,
                status=report.status,
                samples=list(report.samples),
            )

    async def create_report(self, sample_percentage: int) -> str:
        questions_snapshot = list(self._questions.values())
        pct = len(questions_snapshot) / 100
        sample_count = int(pct * sample_percentage)

        questions = random.sample(questions_snapshot, sample_count)
        number_of_variants_to_test = 3

        report = Report(
            expected_samples=len(questions) * number_of_variants_to_test,
            status="running",
            samples=[],
        )

        async def report_task() -> None:
            try:
                for q in questions:
                    variant_prompt = f"""\
        Your job is to generate 3 variants for a given question - different ways to ask the exact same question.
        Each variant should be asked as if from a different persona in a slightly different situation, in a different styl.
        However, each variant should semantically be exactly the same as the original question - no more, no less.

        Generate a JSON object with the following properties:

        insights_on_different_scenarios_and_angles_from_which_to_ask_the_same_question: "your chain of thought..."
        variant_1: "..."
        variant_2: "..."
        variant_3: "..."

        Here's the question you are given: ###
        {q.variants[0]}
        ###

        And for your reference and context, here's its answer: ###
        {q.answer}
        ###
        """

                    if len(q.variants) < number_of_variants_to_test:
                        self.logger.info(
                            f"[Report] Generating variants for question {q.id}"
                        )

                        variant_result = await self._variant_generator.generate(
                            variant_prompt, hints={"strict": True}
                        )

                        variants = random.sample(
                            q.variants
                            + [
                                variant_result.content.variant_1,
                                variant_result.content.variant_2,
                                variant_result.content.variant_3,
                            ],
                            number_of_variants_to_test,
                        )
                    else:
                        variants = random.sample(q.variants, number_of_variants_to_test)

                    for variant in variants:
                        self.logger.info(
                            f"[Report] Generating answer for variant {variant}"
                        )

                        answer = await self.ask_question(variant)

                        test_prompt = f"""\
        Your job is to aid in creating a Confusion Matrix for a language model that retrieves information based on a query.

        You are given 3 things:
        1. Ground Truth (including original Q&A from the database)
        2. A user query
        3. A generated answer

        IMPORTANT NOTE: Omission, specifically, is NOT considered a hallucination.
        ALSO NOTE: It is okay for an answer to cover more than was was asked, as long as the information
        provided is directly based on the provided ground truth.

        You must evaluate the following conditions, providing a JSON object with the following format:
        insights_and_evaluation_on_the_generated_answer_compared_to_the_original: str
        does_the_generated_answer_contain_hallucinations: bool
        detected_hallucination_explanation: str | None
        does_the_generated_answer_contain_any_facts_that_are_not_given_in_the_original_question_and_answer: bool
        does_the_generated_answer_provide_a_full_answer: bool
        does_the_generated_answer_provide_a_partial_answer: bool

        Here's the Ground Truth: ###
        Question: {q.variants[0]}

        Primary Answer:
        {q.answer}

        Additional Information Provided as Part of Answer:
        [Quotes]:: {[r.quotes for r in answer.references]}
        [Relevant Entities]:: {answer.extracted_entities}
        ###

        Here's the user query: ###
        {variant}
        ###

        Here's the generated answer: ###
        {answer.content}
        ###
        """
                        self.logger.info(
                            f"[Report] Evaluating {variant} with respect to question {q.id}"
                        )

                        result = await self._test_generator.generate(
                            test_prompt, hints={"strict": True}
                        )

                        references_check_out = True

                        for ref in answer.references:
                            if ref.question_id not in self._questions:
                                references_check_out = False
                                break
                            for quote in ref.quotes:
                                if quote not in self._questions[ref.question_id].answer:
                                    references_check_out = False
                                    break

                        async with self._report_lock:
                            report.samples.append(
                                ReportSample(
                                    question_id=q.id,
                                    variant=variant,
                                    answer=answer,
                                    evaluation=result.content.insights_and_evaluation_on_the_generated_answer_compared_to_the_original,
                                    references_check_out=references_check_out,
                                    hallucination=(
                                        result.content.detected_hallucination_explanation
                                        if (
                                            result.content.does_the_generated_answer_contain_hallucinations
                                            or result.content.does_the_generated_answer_contain_any_facts_that_are_not_given_in_the_original_question_and_answer
                                        )
                                        else None
                                    ),
                                )
                            )

            except Exception:
                report.status = "failed"

            async with self._report_lock:
                report.status = "completed"

        report_id = generate_id()

        async with self._report_lock:
            self._reports[report_id] = report
            await self._task_service.start(report_task(), tag=f"Report {report_id}")

        return report_id


@asynccontextmanager
async def create_persistent_app(
    service: NLPService | None = None,
) -> AsyncIterator[App]:
    correlator = ContextualCorrelator()
    logger = FileLogger(
        Path("parlant-qna.log"),
        correlator,
        log_level=LogLevel.DEBUG,
        logger_id="parlant-qna",
    )

    if not service:
        service = OpenAIService(logger)

    async with JSONFileDocumentDatabase(
        logger=logger,
        file_path=Path("parlant-qna-db.json"),
    ) as db:
        with correlator.correlation_scope("parlant-qna"):
            async with App(db, service, logger) as app:
                logger.info("Initialized Parlant Q&A")
                yield app


@asynccontextmanager
async def create_transient_app() -> AsyncIterator[App]:
    correlator = ContextualCorrelator()
    logger = StdoutLogger(correlator, logger_id="parlant-qna")
    service = OpenAIService(logger)

    with correlator.correlation_scope("parlant-qna"):
        async with App(TransientDocumentDatabase(), service, logger) as app:
            yield app
