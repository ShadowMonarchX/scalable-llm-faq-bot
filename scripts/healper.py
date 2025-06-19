from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en")

ds = load_dataset("lingshu-medical-mllm/ReasonMed")