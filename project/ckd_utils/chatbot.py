# ckd_utils/chatbot.py
# ─────────────────────────────────────────────────────────────────────────────
# CKD Medical Assistant Chatbot
#
# Architecture (hybrid approach):
#   1. Offline Knowledge Base search (primary, always available, < 2s)
#   2. Google Gemini API fallback (if GEMINI_API_KEY env var is set)
#   3. OpenAI API fallback       (if OPENAI_API_KEY env var is set)
#   4. Graceful "I don't know" message if neither key is available
#
# Safety:
#   - Always appends a medical disclaimer to LLM-generated responses.
#   - Never suggests diagnoses or prescribes medications.
#   - Keeps responses under 300 words.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import re
import time
import logging
from typing import Optional
from dotenv import load_dotenv

from ckd_utils.knowledge_base import get_knowledge_base

# Ensure we load the .env from the correct root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(BASE_DIR, ".env")
dotenv_loaded = load_dotenv(env_path)

logger = logging.getLogger(__name__)

if dotenv_loaded:
    logger.info(f".env loaded successfully from {env_path}")
else:
    logger.warning(f"Could not load .env file from {env_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MEDICAL_DISCLAIMER = (
    "\n\n---\n"
    "⚠️ *This information is **educational only** and is **not a substitute** for "
    "professional medical advice. Always consult a qualified healthcare provider "
    "for medical decisions.*"
)

# System prompt used when falling back to an LLM API
LLM_SYSTEM_PROMPT = """You are a knowledgeable and empathetic CKD (Chronic Kidney Disease) 
Medical Assistant chatbot embedded in a clinical decision-support tool. Your role is to:

1. Explain CKD stages, medical terms, and test results in simple, clear language.
2. Provide general kidney health information and recommendations.
3. Answer questions about the prediction application.
4. Guide users on lifestyle, diet, and exercise for kidney health.

CRITICAL RULES:
- NEVER diagnose diseases or prescribe medications.
- NEVER claim to replace a doctor or nephrologist.
- ALWAYS recommend consulting a healthcare professional for medical decisions.
- Keep responses under 300 words.
- Use markdown formatting for clarity (bullet points, bold text).
- Be compassionate and reassuring, especially for patients with advanced CKD.
- If asked something outside CKD/kidney health, politely redirect.

You are NOT a general chatbot. Stay focused on kidney health and CKD.
"""

# Suggested quick questions shown in the UI
QUICK_QUESTIONS = [
    "What is CKD?",
    "Explain my prediction",
    "What is eGFR?",
    "What foods should I avoid?",
    "How can I protect my kidneys?",
    "What are the symptoms of CKD?",
    "What is the difference between CKD stages?",
]

# Fallback message when no answer is found and no LLM is configured
FALLBACK_MESSAGE = (
    "I'm sorry, I couldn't find a specific answer to that question in my knowledge base.\n\n"
    "Here are some things I **can** help you with:\n"
    "- Explaining CKD stages and what they mean\n"
    "- Defining medical terms (eGFR, creatinine, BUN, etc.)\n"
    "- Providing kidney health tips and diet recommendations\n"
    "- Explaining your prediction result\n"
    "- Answering common CKD questions\n\n"
    "Try asking one of the quick questions above, or rephrasing your question.\n\n"
    "For specific medical advice, please consult your **nephrologist** or "
    "**primary care physician**."
)

GREETING_MESSAGE = (
    "👋 Hello! I'm your **Chronic Kidney Disease Medical Assistant**.\n\n"
    "I can help you:\n"
    "- 🏥 Understand Chronic Kidney Disease stages and medical terms\n"
    "- 📊 Explain your prediction results\n"
    "- 🥗 Learn about kidney-friendly diet and lifestyle\n"
    "- 💊 Understand medications and tests\n"
    "- 🗺️ Navigate this application\n\n"
    "Use the quick question buttons above or type your question below.\n\n"
    "⚠️ *I provide educational information only — not medical advice.*"
)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Availability Checks (lazy imports with graceful failure)
# ─────────────────────────────────────────────────────────────────────────────

def _get_clean_error_message(exception: Exception) -> str:
    err_str = str(exception)
    if "403" in err_str or "API_KEY_INVALID" in err_str or "invalid" in err_str.lower():
        return "Invalid API Key"
    if "permission" in err_str.lower() or "denied" in err_str.lower():
        return "Permission Denied"
    if "429" in err_str or "quota" in err_str.lower() or "exhausted" in err_str.lower():
        return "Quota Exceeded"
    if "network" in err_str.lower() or "connection" in err_str.lower() or "dns" in err_str.lower():
        return "Network Error"
    return f"Gemini API Error: {err_str}"


def _get_gemini_api_key() -> str:
    """Retrieve GEMINI_API_KEY from environment or Streamlit secrets."""
    # Use os.getenv as requested
    key = os.getenv("GEMINI_API_KEY", "")
    if key:
        return key
    # Fall back to Streamlit secrets
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


def _check_gemini_available() -> bool:
    """Check if Gemini API library is installed."""
    try:
        import google.generativeai  # noqa: F401
        return True
    except ImportError:
        return False


def _check_openai_available() -> bool:
    """Check if OpenAI API is available (env key + library installed)."""
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# LLM Response Functions
# ─────────────────────────────────────────────────────────────────────────────

def _query_gemini(user_message: str, model_name: str, context: Optional[str] = None) -> Optional[str]:
    """
    Query Google Gemini API for a response using google-generativeai.
    """
    try:
        import google.generativeai as genai

        api_key = _get_gemini_api_key()
        if not api_key:
            return None

        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=LLM_SYSTEM_PROMPT,
        )

        # Build the full prompt with optional prediction context
        if context:
            user_prompt = (
                f"[Patient Prediction Context]\n{context}\n\n"
                f"[User Question]\n{user_message}"
            )
        else:
            user_prompt = user_message

        response = model.generate_content(
            user_prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=400,
                temperature=0.3,
            ),
        )

        if response and response.text:
            return _truncate_to_word_limit(response.text, max_words=300)

    except Exception as e:
        logger.warning(f"Gemini API error: {e}")
        raise  # Re-raise so the caller can surface a meaningful error

    return None


def _query_openai(user_message: str, context: Optional[str] = None) -> Optional[str]:
    """
    Query OpenAI API for a response.

    Parameters
    ----------
    user_message : str
        The user's question.
    context : Optional[str]
        Optional prediction context to include in the prompt.

    Returns
    -------
    Optional[str]
        Generated response, or None if the API call fails.
    """
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

        # Build messages list
        messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]

        user_content = user_message
        if context:
            user_content = (
                f"[Patient Prediction Context]\n{context}\n\n"
                f"[User Question]\n{user_message}"
            )

        messages.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=400,
            temperature=0.3,
        )

        if response.choices and response.choices[0].message.content:
            return _truncate_to_word_limit(
                response.choices[0].message.content, max_words=300
            )

    except Exception as e:
        logger.warning(f"OpenAI API error: {e}")

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def _truncate_to_word_limit(text: str, max_words: int = 300) -> str:
    """
    Truncate text to a maximum word count while preserving readability.

    Truncates at the last sentence boundary within the word limit where possible.
    """
    words = text.split()
    if len(words) <= max_words:
        return text

    truncated = " ".join(words[:max_words])
    # Try to end at a sentence boundary
    last_period = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
    if last_period > len(truncated) * 0.6:  # At least 60% of the text
        truncated = truncated[: last_period + 1]

    return truncated + "\n\n*[Response truncated for brevity.]*"


def format_prediction_context(prediction_result: Optional[dict]) -> Optional[str]:
    """
    Convert a prediction result dictionary into a human-readable context string
    for the chatbot.

    Parameters
    ----------
    prediction_result : Optional[dict]
        Dictionary containing prediction output from the CKD model.
        Expected keys: 'stage', 'confidence', 'input_data', 'shap_values'

    Returns
    -------
    Optional[str]
        Formatted context string, or None if no prediction is available.
    """
    if not prediction_result:
        return None

    lines = ["=== Current Patient Prediction ==="]

    stage = prediction_result.get("stage", "Unknown")
    confidence = prediction_result.get("confidence", None)
    lines.append(f"Predicted CKD Stage: {stage}")
    if confidence is not None:
        lines.append(f"Confidence: {confidence:.1f}%")

    # Add key input features if available
    input_data = prediction_result.get("input_data", {})
    if input_data:
        lines.append("\nKey Input Values:")
        key_features = [
            ("eGFR", "eGFR"),
            ("Serum_Creatinine", "Serum Creatinine"),
            ("Blood_Urea_Nitrogen", "BUN"),
            ("Albumin_Creatinine_Ratio", "ACR"),
            ("Systolic_BP", "Systolic BP"),
            ("Diastolic_BP", "Diastolic BP"),
            ("BMI", "BMI"),
            ("Age", "Age"),
        ]
        for key, label in key_features:
            if key in input_data:
                val = input_data[key]
                if isinstance(val, float):
                    lines.append(f"  - {label}: {val:.2f}")
                else:
                    lines.append(f"  - {label}: {val}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main Chatbot Class
# ─────────────────────────────────────────────────────────────────────────────

class CKDChatbot:
    """
    AI-powered CKD Medical Assistant Chatbot.

    Response generation pipeline:
    1. Offline knowledge base search (always tried first, < 2 seconds)
    2. Gemini API fallback (if GEMINI_API_KEY env var is configured)
    3. OpenAI API fallback (if OPENAI_API_KEY env var is configured)
    4. Friendly fallback message (if no answer found)

    The chatbot is stateless — it does not retain conversation history
    internally. The Streamlit app manages session-level history.

    Usage
    -----
    >>> chatbot = CKDChatbot()
    >>> response = chatbot.get_response("What is eGFR?")
    >>> print(response)
    """

    def __init__(self) -> None:
        """Initialise the chatbot, loading the knowledge base singleton."""
        self.kb = get_knowledge_base()
        self.model_name = "gemini-2.5-flash"
        self.gemini_error = None
        self._gemini_available = False

        if not _check_gemini_available():
            self.gemini_error = "google-generativeai library not installed."
            logger.warning(self.gemini_error)
        else:
            api_key = _get_gemini_api_key()
            if not api_key:
                self.gemini_error = "Gemini API Key not configured."
                logger.warning(self.gemini_error)
            else:
                logger.info(f"Loaded Gemini API Key: {'*' * 8} (Length: {len(api_key)})")
                print(f"Loaded Gemini API Key: {'*' * 8} (Length: {len(api_key)})")
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    
                    try:
                        logger.info(f"Testing Gemini connection with {self.model_name}...")
                        model = genai.GenerativeModel(self.model_name)
                        _ = model.generate_content("Hello")
                        self._gemini_available = True
                        logger.info(f"Gemini initialized successfully with {self.model_name}")
                    except Exception as e:
                        err_str = str(e)
                        # If model not found (404), fall back to other models automatically to make the app work
                        if "404" in err_str or "not found" in err_str.lower():
                            logger.warning(f"{self.model_name} not found. Falling back to gemini-2.0-flash...")
                            try:
                                self.model_name = "gemini-2.0-flash"
                                model = genai.GenerativeModel(self.model_name)
                                _ = model.generate_content("Hello")
                                self._gemini_available = True
                                logger.info(f"Gemini initialized successfully with {self.model_name}")
                            except Exception as e2:
                                err_str2 = str(e2)
                                if "404" in err_str2 or "not found" in err_str2.lower():
                                    logger.warning(f"{self.model_name} not found. Falling back to gemini-1.5-flash...")
                                    self.model_name = "gemini-1.5-flash"
                                    model = genai.GenerativeModel(self.model_name)
                                    _ = model.generate_content("Hello")
                                    self._gemini_available = True
                                    logger.info(f"Gemini initialized successfully with {self.model_name}")
                                else:
                                    raise e2
                        else:
                            raise e
                            
                except Exception as e:
                    self.gemini_error = _get_clean_error_message(e)
                    logger.error(f"Gemini initialization or test failed: {self.gemini_error}")

        self._openai_available = _check_openai_available()

    def get_response(
        self,
        user_input: str,
        prediction_context: Optional[dict] = None,
    ) -> tuple[str, str]:
        """
        Generate a response to the user's message.
        """
        start_time = time.time()

        # ── Input validation ──────────────────────────────────────────────
        if not user_input or not user_input.strip():
            return (
                "Please ask me a question about Chronic Kidney Disease "
                "or how to use this application!",
                "fallback",
            )

        user_input = user_input.strip()

        # ── Handle "explain my prediction" intent specially ───────────────
        if self._is_prediction_question(user_input):
            response = self._explain_prediction(prediction_context)
            if response:
                elapsed = time.time() - start_time
                logger.info(f"Prediction explanation generated in {elapsed:.3f}s")
                return response + MEDICAL_DISCLAIMER, "knowledge_base"

        # ── Step 1: Gemini API (primary if configured) ────────────────────
        # Attempt to use Gemini. If it fails due to any reason, catch it and fall back silently.
        if self._gemini_available and not self.gemini_error:
            context_str = format_prediction_context(prediction_context)
            try:
                # Gemini Response
                gemini_response = _query_gemini(user_input, self.model_name, context_str)
                if gemini_response:
                    elapsed = time.time() - start_time
                    logger.info(f"Gemini response in {elapsed:.3f}s")
                    return gemini_response + MEDICAL_DISCLAIMER, "gemini"
            except Exception as e:
                # Error Handling
                err_msg = _get_clean_error_message(e)
                logger.error(f"Gemini query failed ({err_msg}). Falling back to Knowledge Base.")

        # ── Step 2: Knowledge Base Fallback ────────────────────────────────
        kb_answer = self.kb.search(user_input)
        if kb_answer:
            elapsed = time.time() - start_time
            logger.info(f"KB hit in {elapsed:.3f}s for: '{user_input[:50]}'")
            return kb_answer + MEDICAL_DISCLAIMER, "knowledge_base"

        # ── Step 3: Standard Not Found message ─────────────────────────────
        elapsed = time.time() - start_time
        logger.info(f"Fallback response in {elapsed:.3f}s for: '{user_input[:50]}'")
        return "I'm sorry, I couldn't find information related to your question.", "fallback"

    @staticmethod
    def _is_prediction_question(text: str) -> bool:
        """
        Check if the user is asking about their current prediction result.

        Returns True if the message contains keywords suggesting they want
        the prediction explained.
        """
        lower = text.lower()
        prediction_keywords = [
            "my prediction", "my result", "what does my", "explain prediction",
            "interpret result", "what does it mean", "my stage", "my ckd",
            "explain my", "what is my prediction", "what did it predict",
        ]
        return any(kw in lower for kw in prediction_keywords)

    @staticmethod
    def _explain_prediction(prediction_context: Optional[dict]) -> Optional[str]:
        """
        Generate a plain-language explanation of the current prediction.

        Returns None if no prediction context is available (in which case
        the KB search will handle the query).
        """
        if not prediction_context:
            return (
                "No prediction has been made yet in this session. "
                "Please go to the **🔬 Prediction** page, enter the patient's "
                "lab values, and run the assessment first.\n\n"
                "Once a prediction is made, come back here and ask me to "
                "**'Explain my prediction'** — I'll interpret the results for you!"
            )

        stage = prediction_context.get("stage", "Unknown")
        confidence = prediction_context.get("confidence", None)
        input_data = prediction_context.get("input_data", {})

        # Build a personalised explanation
        lines = [f"### 🔍 Prediction Explanation\n"]
        lines.append(f"**Predicted Stage:** {stage}")
        if confidence is not None:
            lines.append(f"**Model Confidence:** {confidence:.1f}%\n")

        # Stage-specific explanation
        stage_explanations = {
            "Healthy Kidney": (
                "The model predicted **Healthy Kidney** — this means the input values "
                "did not show strong markers of CKD. Your lab values are within ranges "
                "typically associated with normal kidney function."
            ),
            "Mild CKD (Stage 1–2)": (
                "The model detected markers consistent with **early-stage CKD**. "
                "Your kidneys may still be functioning at or near normal capacity, "
                "but signs of kidney damage are present (such as elevated protein in urine). "
                "Early action at this stage can significantly slow progression."
            ),
            "Moderate CKD (Stage 3)": (
                "The model detected markers consistent with **moderate CKD (Stage 3)**. "
                "Kidney filtering capacity is moderately reduced (eGFR 30–59). "
                "This stage requires active medical management and regular monitoring."
            ),
            "Severe CKD (Stage 4)": (
                "The model detected markers consistent with **severe CKD (Stage 4)**. "
                "Kidney function is significantly reduced (eGFR 15–29). "
                "This is a critical stage requiring specialist care and planning for "
                "renal replacement therapy."
            ),
            "Kidney Failure (Stage 5)": (
                "The model detected markers consistent with **kidney failure (Stage 5)**. "
                "This represents end-stage renal disease (ESRD) where the kidneys can no "
                "longer adequately filter waste. Dialysis or transplant is typically required."
            ),
        }

        explanation = stage_explanations.get(
            stage,
            "The model assessed the lab values and made a prediction based on patterns "
            "learned from a CKD clinical dataset."
        )
        lines.append(f"\n{explanation}\n")

        # Highlight key values if available
        if input_data:
            lines.append("**Key values that influenced this prediction:**")
            key_features = {
                "eGFR": ("eGFR", "mL/min/1.73m²", "Most critical CKD marker"),
                "Serum_Creatinine": ("Serum Creatinine", "mg/dL", "Kidney waste filtration"),
                "Blood_Urea_Nitrogen": ("BUN", "mg/dL", "Nitrogen waste in blood"),
                "Albumin_Creatinine_Ratio": ("ACR", "mg/g", "Kidney damage marker"),
                "Urine_Albumin": ("Urine Albumin", "mg/L", "Protein leakage"),
            }
            for key, (label, unit, desc) in key_features.items():
                if key in input_data:
                    val = input_data[key]
                    fmt = f"{val:.2f}" if isinstance(val, float) else str(val)
                    lines.append(f"- **{label}:** {fmt} {unit} — {desc}")

        lines.append(
            "\n💡 *For a full explanation with charts, see the **Feature Importance** "
            "section on the Prediction page, or download the PDF report.*"
        )

        return "\n".join(lines)

    def get_quick_questions(self) -> list[str]:
        """Return the list of suggested quick questions for the UI."""
        return QUICK_QUESTIONS

    def get_greeting(self) -> str:
        """Return the initial greeting message shown when the chatbot opens."""
        return GREETING_MESSAGE

    def is_llm_enabled(self) -> bool:
        """Return True if Gemini API is configured and available."""
        return self._gemini_available

    def get_llm_provider(self) -> str:
        """Return a description of the active LLM provider."""
        if self._gemini_available:
            return "Google Gemini"
        return "Not Configured"
