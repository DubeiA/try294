# Copied from eva_p1_workflow_and_agent.py (depends on many symbols kept intact)
import os, json, time, shutil, pathlib
from typing import Dict, Any, Optional, List
from eva_env_base import log, SYSTEM_BASE_DIR, GPT_AVAILABLE
from eva_p1.comfy_client import ComfyClient
from eva_p1.video_analyzer import VideoAnalyzer
from eva_p1.workflow import validate_workflow_nodes
from eva_p1.multi_bandit import MultiDimensionalBandit
from eva_p1.openrouter_analyzer import OpenRouterAnalyzer
from eva_p1.knowledge_analyzer import KnowledgeAnalyzer
from eva_p1.prompt_generator import MegaEroticJSONPromptGenerator
import cv2

class EnhancedVideoAgentV4:
    """Enhanced Video Agent v4 with Mega Erotic JSON Prompt Generator Integration + GPT"""

    def __init__(self, api: str, base_workflow: str, state_dir: str = None, seconds: float = 5.0, openrouter_key: str = None):
        if state_dir is None:
            state_dir = f"{SYSTEM_BASE_DIR}/auto_state"

        # Initialize paths
        self.state_dir = state_dir
        self.comfyui_output = "/workspace/ComfyUI/output"
        self.review_dir = f"{SYSTEM_BASE_DIR}/video_reviews"
        self.prompts_dir = f"{SYSTEM_BASE_DIR}/generated_prompts"

        # Create directories
        for directory in [self.state_dir, self.review_dir, self.prompts_dir]:
            os.makedirs(directory, exist_ok=True)

        for subdir in ["pending", "rated", "thumbnails"]:
            os.makedirs(os.path.join(self.review_dir, subdir), exist_ok=True)

        # Initialize components
        self.client = ComfyClient(api)

        # Load workflow
        try:
            with open(base_workflow, "r", encoding="utf-8") as f:
                self.base_wf = json.load(f)
        except FileNotFoundError:
            log.error(f"Workflow file not found: {base_workflow}")
            raise
        except json.JSONDecodeError as e:
            log.error(f"Invalid JSON in workflow file: {e}")
            raise

        # Validate workflow
        if not validate_workflow_nodes(self.base_wf):
            raise ValueError("Workflow validation failed")

        self.seconds = max(5.0, seconds)
        self.analyzer = VideoAnalyzer()

        # Setup knowledge system
        self.knowledge_path = os.path.join(self.state_dir, "knowledge.json")
        self.ratings_path = os.path.join(self.state_dir, "manual_ratings.json") 
        self.queue_path = os.path.join(self.state_dir, "review_queue.json")

        self.knowledge = self._load_knowledge()
        self.manual_ratings = self._load_manual_ratings()
        self.review_queue = self._load_review_queue()

        # Initialize MEGA EROTIC JSON prompt generator with knowledge analyzer
        self.knowledge_analyzer = KnowledgeAnalyzer(self.knowledge)
        self.mega_erotic_generator = MegaEroticJSONPromptGenerator(self.knowledge_analyzer)

        # Initialize multi-dimensional bandit
        bandit_path = os.path.join(self.state_dir, "bandit_state.json")
        self.bandit = MultiDimensionalBandit(bandit_path)

        # 🤖 ДОДАЄМО GPT АНАЛІЗАТОР
        if openrouter_key and GPT_AVAILABLE:
            try:
                self.gpt_analyzer = OpenRouterAnalyzer(openrouter_key)
                log.info("🤖 GPT-powered rating analysis enabled")
            except Exception as e:
                log.error(f"Failed to initialize GPT analyzer: {e}")
                self.gpt_analyzer = None
        else:
            self.gpt_analyzer = None
            if openrouter_key and not GPT_AVAILABLE:
                log.warning("⚠️ OpenRouter key provided but OpenAI library not installed")
            else:
                log.info("⚠️ No OpenRouter key provided, using basic analysis")

        log.info(f"✅ Initialized Enhanced Video Agent v4 with MEGA EROTIC JSON Prompt Generator {'+ GPT' if self.gpt_analyzer else ''}")
        log.info(f"📁 ComfyUI output: {self.comfyui_output}")
        log.info(f"📁 System state: {self.state_dir}")
        log.info(f"🎯 Multi-dimensional bandit ready")
        log.info(f"🔥 MEGA EROTIC intelligent JSON prompt generation enabled")
        log.info(f"⏰ Minimum video duration: {self.seconds}s")

    def _extract_manual_overall(self, rating_data: dict) -> float:
        """Універсальний екстрактор ручних оцінок для різних форматів"""
        r = rating_data.get("rating", {})
        val = r.get("overall_quality") or r.get("overall")

        if val is None:
            val = rating_data.get("user_feedback", {}).get("overall_rating")

        if val is None:
            val = r.get("quality") or r.get("score") or rating_data.get("overall")

        try:
            return float(val)
        except (TypeError, ValueError):
            log.warning(f"Cannot parse manual rating value: {val}, using default 5.0")
            return 5.0

    def _load_knowledge(self) -> Dict[str, Any]:
        """Load knowledge database"""
        if os.path.isfile(self.knowledge_path):
            try:
                with open(self.knowledge_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log.warning(f"Failed to load knowledge: {e}")
        return {"best_score": -1.0, "best_params": {}, "best_combo": None, "history": []}

    def _save_knowledge(self):
        """Save knowledge database with full parameters"""
        with open(self.knowledge_path, "w", encoding="utf-8") as f:
            json.dump(self.knowledge, f, ensure_ascii=False, indent=2)

    def _load_manual_ratings(self) -> Dict[str, Any]:
        """Load manual ratings database"""
        if os.path.isfile(self.ratings_path):
            try:
                with open(self.ratings_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log.warning(f"Failed to load manual ratings: {e}")
        return {}

    def _load_review_queue(self) -> Dict[str, Any]:
        """Load review queue"""
        if os.path.isfile(self.queue_path):
            try:
                with open(self.queue_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log.warning(f"Failed to load review queue: {e}")
        return {"pending": [], "in_review": [], "completed": []}

    def _save_review_queue(self):
        """Save review queue"""
        with open(self.queue_path, "w", encoding="utf-8") as f:
            json.dump(self.review_queue, f, ensure_ascii=False, indent=2)

    def find_generated_video(self, prefix: str) -> Optional[str]:
        """Find generated video in ComfyUI output directory"""
        search_paths = [
            "/workspace/ComfyUI/output",
            "/workspace/ComfyUI/output/video",
            f"{self.comfyui_output}",
            "./workspace/ComfyUI/output",
            "workspace/ComfyUI/output", 
            "./output",
            "output"
        ]

        log.info(f"🔍 Searching for video with prefix: {prefix}")

        for search_dir in search_paths:
            if not os.path.exists(search_dir):
                continue

            try:
                import pathlib, time
                pattern_files = list(pathlib.Path(search_dir).glob(f"*{prefix}*.mp4"))
                if pattern_files:
                    video_path = str(max(pattern_files, key=lambda p: p.stat().st_mtime))
                    log.info(f"✅ Found video with prefix: {video_path}")
                    return video_path

                all_videos = list(pathlib.Path(search_dir).glob("*.mp4"))
                if all_videos:
                    recent_videos = [
                        v for v in all_videos 
                        if time.time() - v.stat().st_mtime < 180
                    ]
                    if recent_videos:
                        video_path = str(max(recent_videos, key=lambda p: p.stat().st_mtime))
                        log.info(f"✅ Found recent video: {video_path}")
                        return video_path

            except Exception as e:
                log.warning(f"Error searching in {search_dir}: {e}")

        return None

    def create_thumbnail(self, video_path: str, video_id: str) -> Optional[str]:
        """Create thumbnail from video middle frame"""
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            mid_frame = frame_count // 2

            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()
            cap.release()

            if ret:
                thumbnail_path = os.path.join(self.review_dir, "thumbnails", f"{video_id}.jpg")
                resized = cv2.resize(frame, (320, 240))
                cv2.imwrite(thumbnail_path, resized)
                return thumbnail_path
        except Exception as e:
            log.error(f"Failed to create thumbnail: {e}")

        return None

    def add_to_review_queue(self, video_path: str, params: Dict[str, Any], auto_metrics: Dict[str, float], combo: List[str]):
        """Add video to manual review queue"""
        import time
        video_id = os.path.basename(video_path).replace('.mp4', '')

        # Create thumbnail
        thumbnail_path = self.create_thumbnail(video_path, video_id)

        # Copy to pending directory
        pending_path = os.path.join(self.review_dir, "pending", f"{video_id}.mp4")
        try:
            shutil.copy2(video_path, pending_path)
        except Exception as e:
            log.warning(f"Failed to copy video to pending: {e}")
            pending_path = video_path

        # Calculate priority based on auto metrics
        score = auto_metrics.get("overall", 0.5)
        if score > 0.8 or score < 0.3:
            priority = 1
        elif score > 0.6:
            priority = 2
        else:
            priority = 3

        # Add to queue
        queue_item = {
            "video_id": video_id,
            "video_path": pending_path,
            "original_path": video_path,
            "thumbnail_path": thumbnail_path,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "auto_metrics": auto_metrics,
            "prompt": params.get("prompt", ""),
            "params": params,
            "combo": combo,
            "priority": priority
        }

        self.review_queue["pending"].append(queue_item)
        self._save_review_queue()

        log.info(f"✚ Added to review queue: {video_id} (priority: {priority})")

