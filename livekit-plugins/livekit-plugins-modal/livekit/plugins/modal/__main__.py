# Copyright 2026 Mahimai Raja J
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

"""python -m livekit.plugins.modal provision stt --model ... [--env ...] [--region ...]"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from . import provision
from .models import DEFAULT_MODEL, STT_MODELS

TOKEN_ID_ENV = "MODAL_PROXY_TOKEN_ID"
TOKEN_SECRET_ENV = "MODAL_PROXY_TOKEN_SECRET"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m livekit.plugins.modal")
    commands = parser.add_subparsers(dest="command", required=True)
    prov = commands.add_parser("provision", help="deploy a model server to Modal, once")
    kinds = prov.add_subparsers(dest="kind", required=True)

    stt = kinds.add_parser("stt", help="speech-to-text server")
    stt.add_argument("--model", default=DEFAULT_MODEL, choices=list(STT_MODELS))
    stt.add_argument("--env", default=None, help="Modal environment (default: your profile's)")
    stt.add_argument("--gpu", default=None, help="GPU type (default: the model's)")
    stt.add_argument("--region", default=None, help="pin containers to a region, e.g. us-east")
    stt.add_argument(
        "--min-containers",
        type=int,
        default=0,
        help="containers kept warm at all times; each one bills while idle (default: 0)",
    )
    stt.add_argument(
        "--scaledown-window",
        type=int,
        default=300,
        help="seconds an idle container stays up after its last request (default: 300)",
    )
    stt.add_argument(
        "--redeploy", action="store_true", help="deploy even if the App already exists"
    )
    stt.add_argument(
        "--no-proxy-token",
        action="store_true",
        help=f"do not create a proxy token when {TOKEN_ID_ENV} is unset",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = provision.ServerConfig(
        model=args.model,
        environment=args.env,
        gpu=args.gpu,
        region=args.region,
        min_containers=args.min_containers,
        scaledown_window=args.scaledown_window,
    )
    result = provision.provision_stt(config, redeploy=args.redeploy)

    low, high = result.warm_cost_per_hour
    cost = f"${low:.2f}/h" if low == high else f"${low:.2f} to ${high:.2f}/h (region pricing)"
    print()
    print(f"app      {result.app_name} (env: {result.environment or 'default'})")
    print(f"status   {'deployed now' if result.created else 'already deployed'}")
    print(f"url      {result.url}")
    print(f"gpu      {result.gpu}")
    print(f"warm     {result.min_containers} container(s), {cost} while idle")
    if result.min_containers == 0:
        print("         scales to zero: the first request after idle waits for a cold start")

    print()
    if os.environ.get(TOKEN_ID_ENV) and os.environ.get(TOKEN_SECRET_ENV):
        print(f"proxy token: using {TOKEN_ID_ENV} from the environment")
    elif args.no_proxy_token:
        print("The endpoint requires a proxy token. Create one with:")
        print("    modal workspace proxy-tokens create")
        print(f"and set {TOKEN_ID_ENV} and {TOKEN_SECRET_ENV} for your agent.")
    else:
        token_id, token_secret = provision.create_proxy_token(args.env)
        print("Created a proxy token. The secret is shown once; set these for your agent:")
        print(f"    export {TOKEN_ID_ENV}={token_id}")
        print(f"    export {TOKEN_SECRET_ENV}={token_secret}")

    print()
    env_arg = f', env="{args.env}"' if args.env else ""
    print(f'Then: stt=modal.STT(model="{args.model}"{env_arg})')
    return 0


if __name__ == "__main__":
    sys.exit(main())
