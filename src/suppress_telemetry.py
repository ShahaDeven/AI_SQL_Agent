"""
Telemetry Suppression Helper
============================
Import this module FIRST in any file that uses ChromaDB, TensorFlow, or other 
libraries with noisy telemetry/logging.

Usage:
    import src.suppress_telemetry  # Must be first import!
    # ... rest of your imports
"""

import os
import warnings
import logging

# ===== ENVIRONMENT VARIABLES (must be set before library imports) =====
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ["POSTHOG_DISABLED"] = "true"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress tokenizer warnings

# ===== LOGGING SUPPRESSION =====
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('chromadb').setLevel(logging.ERROR)
logging.getLogger('chromadb.telemetry').setLevel(logging.CRITICAL)
logging.getLogger('chromadb.telemetry.product').setLevel(logging.CRITICAL)
logging.getLogger('chromadb.telemetry.product.posthog').setLevel(logging.CRITICAL)
logging.getLogger('posthog').setLevel(logging.CRITICAL)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)

# ===== WARNINGS SUPPRESSION =====
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', module='tensorflow')
warnings.filterwarnings('ignore', module='chromadb')
warnings.filterwarnings('ignore', message='.*torch.classes.*')
warnings.filterwarnings('ignore', message='.*telemetry.*')
warnings.filterwarnings('ignore', message='.*Examining the path.*')


# ===== MONKEY-PATCH CHROMADB TELEMETRY (Nuclear Option) =====
def _disable_chroma_telemetry():
    """Completely disable ChromaDB telemetry by patching the client."""
    try:
        from chromadb.telemetry.product import posthog
        
        class NoOpTelemetry:
            def capture(self, *args, **kwargs):
                pass
            def __getattr__(self, name):
                return lambda *args, **kwargs: None
        
        posthog.Posthog = NoOpTelemetry
    except ImportError:
        pass
    except Exception:
        pass

_disable_chroma_telemetry()

print("Telemetry suppression loaded")