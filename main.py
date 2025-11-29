import argparse
import logging
import os
import sys
import traceback

logging.getLogger("httpx").setLevel(logging.WARNING)

from ai_steward.steward import run_secure_answer
from ai_steward.llm import AVAILABLE_LLMS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Secure Answer Agent with On-the-fly Embedding and Chain-of-Thought")
    parser.add_argument("--model", type=str, default=os.environ['LLM_MODEL_NAME'], choices=AVAILABLE_LLMS, help="The reasoning model's name. If not specified, use LLM_MODEL_NAME.")
    parser.add_argument("--embed-model", type=str, default=os.environ["EMBEDDING_MODEL_NAME"], help="The embedding model's name. If not specified, use EMBEDDING_MODEL_NAME")
    parser.add_argument("--question", type=str, default=None, help="An user input")
    parser.add_argument("--question_file", type=str, default=None, help="A path to user input's file")

    parser.add_argument("--user-level", type=str, required=True, choices=["L0", "L1", "L2", "L3"])
    parser.add_argument("--kb", type=str, default="examples/kb.json", help="A path to database")
    parser.add_argument("--out", type=str, default="outputs/secure_answer", help="A path to store results")
    parser.add_argument("--lang", type=str, default="English", help="Language")
    parser.add_argument("--secure", dest="allow_upper", action="store_false", default=True, help="Access upper confidential documents")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    parser.add_argument(
        "--refiner-max-depth", type=int, default=8,
        help="Maximum search depth for the refinement beam search. "
            "Controls how many iterative refinement steps the algorithm is allowed to explore."
    )

    parser.add_argument(
        "--refiner-beam-size", type=int, default=6,
        help="Beam width for the refinement search. "
            "Higher values allow exploring more candidate refinements in parallel."
    )

    parser.add_argument(
        "--refiner-max-trial-num", type=int, default=24,
        help="Maximum number of total candidate refinements that can be evaluated across all depths. "
            "Acts as a global budget to prevent excessive search cost."
    )

    parser.add_argument(
        "--refiner-epsilon", type=float, default=0.25,
        help="Stochastic exploration parameter for refinement search. "
            "Higher epsilon increases diversification by sampling lower-ranked candidates."
    )

    parser.add_argument(
        "--refiner-explore-top-k", type=int, default=6,
        help="Number of top-ranked candidate refinements to consider at each step before applying epsilon sampling."
    )

    args = parser.parse_args()

    if args.question is None and args.question_file is None:
        raise ValueError("Either --question or --question_file must be specified.")

    if args.question_file is not None:
        try:
            with open(args.question_file, "r", encoding="utf-8") as f:
                args.question = f.read().strip()
        except Exception as e:
            logging.error(f"Error reading question file: {e}")
            sys.exit(1)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        if args.debug: logging.debug("--- DEBUG MODE ENABLED ---")
    else:
        logging.basicConfig(level=logging.INFO)
    try:
        run_secure_answer(
            model=args.model,
            embed_model=args.embed_model,
            question=args.question, user_level=args.user_level,
            kb_path=args.kb, out_path=args.out, lang=args.lang,
            allow_upper_context=args.allow_upper,
            max_depth=args.refiner_max_depth, beam_size=args.refiner_beam_size, 
            max_trial_num=args.refiner_max_trial_num, epsilon=args.refiner_epsilon, 
            explore_top_k=args.refiner_explore_top_k
        )
    except Exception:
        traceback.print_exc()
        sys.exit(1)
