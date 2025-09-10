import os
import argparse
from qa.agent_namespace import agent_mod
from qa.patches import (
    patch_logging_to_auto_state,
    patch_bandit_ban_rule,
    patch_video_processor_init,
    patch_openrouter_quality_guard,
    patch_enrich_run_iteration_metrics,
)
import eva_p3.logger as _p3a
import eva_p1.analysis_config as _p1a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8188")
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--state-dir", default="/workspace/wan22_system/auto_state")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--openrouter-key")
    parser.add_argument("--use-enhanced-analysis", action="store_true")
    parser.add_argument("--train-improved", action="store_true")
    parser.add_argument("--reference-only", action="store_true", help="Використовувати тільки еталонні параметри з reference_params.json")
    parser.add_argument("--reference-file", type=str, help="Шлях до reference_params.json (опційно)")
    args = parser.parse_args()

    if args.openrouter_key:
        os.environ["OPENROUTER_API_KEY"] = args.openrouter_key

    patch_logging_to_auto_state()
    patch_bandit_ban_rule()
    patch_video_processor_init()
    patch_openrouter_quality_guard()
    patch_enrich_run_iteration_metrics()

    # ВАЖЛИВО: використовуємо пропатчений логер з agent_mod, щоб мати JSONL та лог-директорію
    logger = agent_mod.EnhancedLogger(_p1a.AnalysisConfig())

    agent = agent_mod.EnhancedVideoAgentV4Merged(
        api=args.api,
        base_workflow=args.workflow,
        state_dir=args.state_dir,
        seconds=max(5.0, args.seconds),
        logger=logger,
        openrouter_key=args.openrouter_key,
        reference_only=bool(args.reference_only),
        reference_file=args.reference_file,
    )

    stats = agent.get_stats_v4()
    agent_mod.log.info(f"📊 QA initial stats: {stats}")
    agent.search_v4(iterations=args.iterations)


