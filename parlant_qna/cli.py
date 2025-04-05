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
import json
import os
import sys
from dataclasses import dataclass
import time
from typing import cast

import click
import httpx
import rich
from rich import box
from rich.table import Table
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
)

from parlant_qna.app import create_persistent_app
from parlant_qna.server import create_server, GLOBAL_TAG

DEFAULT_PORT = 8807


async def run_server(port: int) -> None:
    async with (
        create_persistent_app() as qna_app,
        create_server(port, qna_app),
    ):
        pass


def main() -> None:
    @dataclass(frozen=True)
    class Config:
        server_address: str
        client: httpx.Client

    def get_client(ctx: click.Context) -> httpx.Client:
        return cast(Config, ctx.obj).client

    def write_error(message: str) -> None:
        rich.print(f"[bold red]error: {message}")

    def die_if_error(response: httpx.Response, message: str | None = None) -> None:
        if response.is_success:
            return

        if message:
            rich.print(
                f"[red]error: {message}: http status code {response.status_code}"
            )
        else:
            rich.print(f"[red]error: http status code {response.status_code}")

        try:
            rich.print(response.json())
        except Exception:
            pass

        sys.exit(1)

    def success(message: str) -> None:
        rich.print(f"[green]{message}")

    @click.group("CLI")
    @click.option(
        "-s",
        "--server",
        type=str,
        help="QnA server address",
        metavar="ADDRESS[:PORT]",
        default=f"http://localhost:{DEFAULT_PORT}",
        show_default=True,
    )
    @click.pass_context
    def cli(ctx: click.Context, server: str) -> None:
        if not ctx.obj:
            ctx.obj = Config(
                server_address=server,
                client=httpx.Client(
                    base_url=server,
                    follow_redirects=True,
                    timeout=60,
                ),
            )

    @cli.command("serve", help="Run as a standalone server")
    @click.option(
        "-p",
        "--port",
        metavar="PORT",
        help="Port number",
        default=DEFAULT_PORT,
        show_default=True,
    )
    def server(port: int) -> None:
        asyncio.run(run_server(port))

    # Modified: The ask method can now get tags optionally and use them to filter relevant context for answering a question
    @cli.command("ask", help="Ask a question")
    @click.argument("question", metavar="QUESTION")
    @click.option(
        "-t",
        "--tags",
        help="Tags associated with the question",
        multiple=True,
        required=False,
    )
    @click.pass_context
    def ask(ctx: click.Context, question: str, tags: tuple[str] = ()) -> None:
        if len(tags) == 0:
            tags = [GLOBAL_TAG]
        else:
            tags = [GLOBAL_TAG] + list(tags)

        response = get_client(ctx).post(
            "/answers", json={"query": question, "tags": tags}
        )

        die_if_error(response, "ask question")

        rich.print(response.json())

    # Modified: There is a default tag which gets added to all the question answer pairs if no tags have been provided for it
    @cli.command("add", help="Add a question/answer pair")
    @click.option(
        "-q",
        "--variant",
        help="Question variant (can be given multiple times for different ways to ask it)",
        multiple=True,
        required=True,
    )
    @click.option(
        "-a",
        "--answer",
        help="Answer text",
        required=True,
    )
    @click.option(
        "-t",
        "--tags",
        help="List of tags associated with the question",
        multiple=True,
        required=False,
    )
    @click.pass_context
    def add(
        ctx: click.Context, variant: tuple[str], answer: str, tags: tuple[str] = ()
    ) -> None:

        # When no tags are provided, use the global tag
        if len(tags) == 0:
            tags = [GLOBAL_TAG]

        response = get_client(ctx).post(
            "/questions",
            json={
                "variants": variant,
                "answer": answer,
                "tags": tags,
            },
        )

        die_if_error(response, "add question")

        question_id = response.json()["question_id"]

        success(f"added question (id: {question_id})")

    @cli.command("update", help="Update a question")
    @click.argument("id", metavar="QUESTION_ID")
    @click.option(
        "-q",
        "--variant",
        help="New question variant (can be given multiple times for different ways to ask it). "
        "NOTE: Providing this will override any existing variants.",
        multiple=True,
    )
    @click.option(
        "-a",
        "--answer",
        help="New answer text",
    )
    @click.option(
        "-t",
        "--tags",
        help="New list of tags" "NOTE: Providing this will override any existing tags.",
        multiple=True,
    )
    @click.pass_context
    def update(
        ctx: click.Context,
        id: str,
        variant: tuple[str],
        answer: str | None,
        tags: tuple[str] | None,
    ) -> None:
        response = get_client(ctx).patch(
            f"/questions/{id}",
            json={
                **({"variants": variant} if variant else {}),
                **({"answer": answer} if answer else {}),
                **({"tags": tags} if tags else {}),
            },
        )

        die_if_error(response, "update question")

        question_id = response.json()["id"]

        success(f"updated question (id: {question_id})")

    @cli.command("list", help="List questions")
    @click.option(
        "-v",
        "--verbose",
        help="Show a lot of information about each answer",
        is_flag=True,
        default=False,
    )
    @click.pass_context
    def list_questions(ctx: click.Context, verbose: bool) -> None:
        response = get_client(ctx).get("/questions", params={"verbose": verbose})

        die_if_error(response, "list questions")

        questions = list(response.json())

        if not questions:
            rich.print("No data available")
            return

        table = Table(box=box.ROUNDED, border_style="bright_green")

        headers = map(str.capitalize, questions[0].keys())

        table.add_column("#", header_style="bright_green", overflow="fold")

        for header in headers:
            table.add_column(header, header_style="bright_green", overflow="fold")

        index = 1

        for q in questions:
            table.add_row(*[str(index)] + list(map(str, q.values())))
            index += 1

        rich.print(table)

    @cli.command("view", help="View a question")
    @click.argument("id", metavar="QUESTION_ID")
    @click.pass_context
    def view(ctx: click.Context, id: str) -> None:
        response = get_client(ctx).get(f"/questions/{id}")

        die_if_error(response, "view question")

        rich.print(response.json())

    @cli.command("remove", help="Remove a question")
    @click.argument("id", metavar="QUESTION_ID")
    @click.pass_context
    def remove(ctx: click.Context, id: str) -> None:
        response = get_client(ctx).delete(f"/questions/{id}")

        die_if_error(response, "delete question")

        success(f"deleted question (id: {id})")

    @cli.command(
        "report",
        help="Test and create a confusion matrix",
    )
    @click.option(
        "-s",
        "--sample-percentage",
        default=10,
        show_default=True,
        help="Sample percentage",
    )
    @click.pass_context
    def report(ctx: click.Context, sample_percentage: int) -> None:
        response = get_client(ctx).post(
            "/reports", json={"sample_percentage": sample_percentage}
        )
        die_if_error(response, "create report")
        report_id = response.json()["report_id"]

        response = get_client(ctx).get(f"/reports/{report_id}")
        die_if_error(response, "read report status")
        report = response.json()["report"]

        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(elapsed_when_finished=True),
        ) as progress:
            task = progress.add_task(
                description=f"Recall: {report['recall']:.3f} | Precision: {report['precision']:.3f}",
                total=report["expected_samples"],
            )

            while True:
                time.sleep(1)

                response = get_client(ctx).get(f"/reports/{report_id}")
                die_if_error(response, "read report status")
                report = response.json()["report"]

                progress.update(
                    task,
                    description=f"Recall: {report['recall']:.3f} | Precision: {report['precision']:.3f}",
                    completed=int(report["completed_samples"]),
                )

                if report["status"] != "running":
                    break

        if report["status"] == "failed":
            write_error("Report generation failed")

        table = Table(box=box.ROUNDED, border_style="bright_green")

        headers = (
            "Accuracy",
            "Recall",
            "Precision",
            "F1",
            "TP",
            "TN",
            "FP",
            "FN",
            "Partial",
        )

        for header in headers:
            table.add_column(header, header_style="bright_green", overflow="fold")

        table.add_row(
            f"{report['accuracy']:.3f}",
            f"{report['recall']:.3f}",
            f"{report['precision']:.3f}",
            f"{report['f1']:.3f}",
            f'{report["matrix"]["tp"]} (Partial: {report["matrix"]["ptp"]})',
            f'{report["matrix"]["tn"]} (Partial: {report["matrix"]["ptn"]})',
            f'{report["matrix"]["fp"]} (Partial: {report["matrix"]["pfp"]})',
            f'{report["matrix"]["fn"]} (Partial: {report["matrix"]["pfn"]})',
        )

        rich.print(table)

        if hallucinations := report["hallucinations"]:
            table = Table(box=box.ROUNDED, border_style="bright_green")

            for header in (
                "Question Id",
                "Query",
                "Generated Answer",
                "Additional References",
                "Extracted Entities",
                "Issue",
            ):
                table.add_column(header, header_style="bright_green", overflow="fold")

            for question_id, item in hallucinations.items():
                table.add_row(
                    str(question_id),
                    str(item["query"]),
                    str(item["answer"]),
                    str(json.dumps(item["references"], indent=2)),
                    str(",".join(item["entities"])),
                    str(item["issue"]),
                )

            rich.print(table)

    @cli.command(
        "help",
        context_settings={"ignore_unknown_options": True},
        help="Show help for a command",
    )
    @click.argument("command", nargs=-1, required=False)
    @click.pass_context
    def help_command(ctx: click.Context, command: tuple[str] | None = None) -> None:
        def transform_and_exec_help(command: str) -> None:
            new_args = [sys.argv[0]] + command.split() + ["--help"]
            os.execvp(sys.executable, [sys.executable] + new_args)

        if not command:
            click.echo(cli.get_help(ctx))
        else:
            transform_and_exec_help(" ".join(command))

    try:
        cli()
    except httpx.NetworkError as e:
        rich.print(f"[red]error: {e} ({type(e).__name__})")
        sys.exit(1)


if __name__ == "__main__":
    main()
