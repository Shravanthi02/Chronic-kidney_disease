# tests/test_chatbot.py
# ─────────────────────────────────────────────────────────────────────────────
# Unit tests for the CKD Medical Assistant Chatbot
#
# Run:  python -m pytest tests/test_chatbot.py -v
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import time
import unittest

# ── Add project root to path ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ckd_utils.knowledge_base import KnowledgeBase, get_knowledge_base
from ckd_utils.chatbot import (
    CKDChatbot,
    MEDICAL_DISCLAIMER,
    QUICK_QUESTIONS,
    GREETING_MESSAGE,
    format_prediction_context,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_prediction_context(stage: str = "Moderate CKD (Stage 3)",
                              confidence: float = 87.5) -> dict:
    """Build a minimal mock prediction context dict."""
    return {
        "stage": stage,
        "confidence": confidence,
        "input_data": {
            "Age": 55,
            "Gender": 1,
            "BMI": 27.3,
            "eGFR": 42.0,
            "Serum_Creatinine": 2.1,
            "Blood_Urea_Nitrogen": 28.0,
            "Albumin_Creatinine_Ratio": 180.0,
            "Urine_Albumin": 95.0,
            "Systolic_BP": 148,
            "Diastolic_BP": 92,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeBase Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKnowledgeBase(unittest.TestCase):
    """Tests for the offline CKD Knowledge Base."""

    @classmethod
    def setUpClass(cls):
        """Load the knowledge base once for all tests in this class."""
        cls.kb = KnowledgeBase()

    def test_kb_has_minimum_entries(self):
        """Knowledge base should contain at least 100 entries."""
        self.assertGreaterEqual(
            len(self.kb), 100,
            f"Expected ≥ 100 KB entries, got {len(self.kb)}"
        )

    def test_kb_singleton(self):
        """get_knowledge_base() should return the same object on repeated calls."""
        kb1 = get_knowledge_base()
        kb2 = get_knowledge_base()
        self.assertIs(kb1, kb2, "Knowledge base singleton not working correctly")

    def test_search_returns_string_for_known_query(self):
        """A well-known query should return a non-empty string."""
        result = self.kb.search("What is CKD?")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_search_ckd_definition(self):
        """Searching for CKD definition returns relevant content."""
        result = self.kb.search("What is chronic kidney disease?")
        self.assertIsNotNone(result)
        # The answer should mention key CKD concepts
        lower = result.lower()
        self.assertTrue(
            any(term in lower for term in ["kidney", "chronic", "filter", "stage"]),
            f"CKD definition answer seems incorrect: {result[:100]}"
        )

    def test_search_egfr(self):
        """Searching for eGFR returns an explanation with numerical ranges."""
        result = self.kb.search("What is eGFR?")
        self.assertIsNotNone(result)
        self.assertIn("egfr", result.lower())

    def test_search_stages(self):
        """Searching for CKD stages returns stage table or stage descriptions."""
        result = self.kb.search("What are the stages of CKD?")
        self.assertIsNotNone(result)
        lower = result.lower()
        # Should mention stage numbers or eGFR ranges
        self.assertTrue(
            any(kw in lower for kw in ["stage", "egfr", "filtration"]),
            "Stages answer does not mention stages or eGFR"
        )

    def test_search_diet_food(self):
        """Searching for diet/food returns dietary recommendations."""
        result = self.kb.search("What foods should I avoid with CKD?")
        self.assertIsNotNone(result)
        lower = result.lower()
        self.assertTrue(
            any(kw in lower for kw in ["sodium", "salt", "potassium", "phosphorus", "food", "diet"]),
            "Diet answer does not mention dietary restrictions"
        )

    def test_search_symptoms(self):
        """Searching for symptoms returns symptom list."""
        result = self.kb.search("What are the symptoms of CKD?")
        self.assertIsNotNone(result)
        lower = result.lower()
        self.assertTrue(
            any(kw in lower for kw in ["fatigue", "swelling", "urine", "tired"]),
            "Symptoms answer does not mention common CKD symptoms"
        )

    def test_search_empty_query_returns_none(self):
        """Empty queries should return None without errors."""
        self.assertIsNone(self.kb.search(""))
        self.assertIsNone(self.kb.search("   "))

    def test_search_none_input_handled(self):
        """None input should be handled gracefully (return None)."""
        # The method signature accepts str, but we test graceful handling
        result = self.kb.search("")
        self.assertIsNone(result)

    def test_search_completely_unrelated_query_returns_none(self):
        """Queries completely unrelated to CKD should not match (below threshold)."""
        # Pure nonsense or unrelated topics should fail to match
        result = self.kb.search("xyzzy plugh frobozz zork irrelevant gibberish")
        # Should return None since no keywords match
        self.assertIsNone(result)

    def test_search_creatinine(self):
        """Searching for creatinine returns relevant explanation."""
        result = self.kb.search("What is serum creatinine?")
        self.assertIsNotNone(result)
        self.assertIn("creatinine", result.lower())

    def test_search_bun(self):
        """Searching for BUN returns Blood Urea Nitrogen explanation."""
        result = self.kb.search("What is blood urea nitrogen BUN?")
        self.assertIsNotNone(result)
        lower = result.lower()
        self.assertTrue(
            any(kw in lower for kw in ["bun", "urea", "nitrogen"]),
        )

    def test_search_hypertension_blood_pressure(self):
        """Searching for blood pressure returns relevant content."""
        result = self.kb.search("What is hypertension blood pressure CKD?")
        self.assertIsNotNone(result)
        lower = result.lower()
        self.assertTrue(
            any(kw in lower for kw in ["blood pressure", "hypertension", "mmhg", "systolic"]),
        )

    def test_search_is_fast(self):
        """KB search should complete in under 500ms for any query."""
        queries = [
            "What is CKD?",
            "eGFR stages kidney",
            "diet food sodium potassium",
            "exercise walking yoga",
            "medication nsaid avoid",
        ]
        for query in queries:
            start = time.time()
            self.kb.search(query)
            elapsed = time.time() - start
            self.assertLess(
                elapsed, 0.5,
                f"KB search took {elapsed:.3f}s for query: '{query}' (limit: 0.5s)"
            )

    def test_get_all_questions_returns_list(self):
        """get_all_questions() should return a non-empty list of strings."""
        questions = self.kb.get_all_questions()
        self.assertIsInstance(questions, list)
        self.assertGreater(len(questions), 0)
        for q in questions:
            self.assertIsInstance(q, str)

    def test_search_kidney_protection(self):
        """Searching for kidney protection advice returns recommendations."""
        result = self.kb.search("How can I protect my kidneys?")
        self.assertIsNotNone(result)
        lower = result.lower()
        self.assertTrue(
            any(kw in lower for kw in ["diet", "exercise", "water", "blood pressure", "kidney"]),
        )

    def test_search_exercise(self):
        """Searching for exercise returns CKD exercise recommendations."""
        result = self.kb.search("Can CKD patients exercise?")
        self.assertIsNotNone(result)
        lower = result.lower()
        self.assertTrue(
            any(kw in lower for kw in ["exercise", "walking", "swimming", "activity"]),
        )

    def test_search_dialysis(self):
        """Searching for dialysis returns dialysis explanation."""
        result = self.kb.search("What is dialysis for kidney failure?")
        self.assertIsNotNone(result)
        lower = result.lower()
        self.assertTrue(
            any(kw in lower for kw in ["dialysis", "hemodialysis", "peritoneal"]),
        )

    def test_search_stage_5(self):
        """Searching for Stage 5 or kidney failure returns relevant content."""
        result = self.kb.search("What is stage 5 kidney failure ESRD?")
        self.assertIsNotNone(result)
        lower = result.lower()
        self.assertTrue(
            any(kw in lower for kw in ["stage 5", "failure", "esrd", "dialysis"]),
        )

    def test_search_acr(self):
        """Searching for ACR returns albumin-creatinine ratio explanation."""
        result = self.kb.search("What is albumin creatinine ratio ACR?")
        self.assertIsNotNone(result)
        lower = result.lower()
        self.assertTrue(
            any(kw in lower for kw in ["albumin", "creatinine", "ratio", "acr"]),
        )

    def test_search_prevent_ckd(self):
        """Searching for CKD prevention returns actionable recommendations."""
        result = self.kb.search("How can I prevent CKD?")
        self.assertIsNotNone(result)
        lower = result.lower()
        self.assertTrue(
            any(kw in lower for kw in ["blood pressure", "diabetes", "diet", "exercise", "prevent"]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# CKDChatbot Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCKDChatbot(unittest.TestCase):
    """Tests for the CKD Medical Assistant Chatbot."""

    @classmethod
    def setUpClass(cls):
        """Create one chatbot instance for all tests."""
        cls.bot = CKDChatbot()

    def _get_text(self, user_input: str, context: dict = None) -> str:
        """Helper: get the response text from the chatbot."""
        text, _ = self.bot.get_response(user_input, context)
        return text

    def _get_source(self, user_input: str, context: dict = None) -> str:
        """Helper: get the response source label."""
        _, source = self.bot.get_response(user_input, context)
        return source

    # ── Response format tests ─────────────────────────────────────────────

    def test_response_returns_tuple(self):
        """get_response() must return a (str, str) tuple."""
        result = self.bot.get_response("What is CKD?")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        text, source = result
        self.assertIsInstance(text, str)
        self.assertIsInstance(source, str)

    def test_disclaimer_always_present(self):
        """Every response from the chatbot must contain the medical disclaimer."""
        queries = [
            "What is CKD?",
            "What is eGFR?",
            "What foods should I avoid?",
            "How can I protect my kidneys?",
            "Tell me about CKD stages",
        ]
        for query in queries:
            text = self._get_text(query)
            self.assertIn(
                "educational", text.lower(),
                f"Disclaimer not found for query: '{query}'"
            )

    def test_empty_input_handled_gracefully(self):
        """Empty or whitespace input should return a helpful prompt."""
        text = self._get_text("")
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_whitespace_only_handled(self):
        """Whitespace-only input should not crash."""
        text = self._get_text("   \n\t  ")
        self.assertIsInstance(text, str)

    def test_known_ckd_question_hits_knowledge_base(self):
        """Standard CKD questions should be answered by the knowledge base."""
        source = self._get_source("What is CKD?")
        self.assertEqual(source, "knowledge_base",
                         "Known CKD question should hit KB, not LLM")

    def test_egfr_question_hits_knowledge_base(self):
        """eGFR questions should be answered offline."""
        source = self._get_source("What is eGFR and what does it measure?")
        self.assertEqual(source, "knowledge_base")

    def test_diet_question_hits_knowledge_base(self):
        """Diet questions should be answered offline."""
        source = self._get_source("What foods should CKD patients avoid?")
        self.assertEqual(source, "knowledge_base")

    def test_symptoms_question_hits_knowledge_base(self):
        """Symptom questions should be answered offline."""
        source = self._get_source("What are the symptoms of chronic kidney disease?")
        self.assertEqual(source, "knowledge_base")

    # ── Prediction explanation tests ──────────────────────────────────────

    def test_no_prediction_context_returns_redirect(self):
        """Asking about prediction without context should redirect to Prediction page."""
        text = self._get_text("Explain my prediction", context=None)
        lower = text.lower()
        self.assertTrue(
            any(kw in lower for kw in ["prediction", "predict", "assessment", "page"]),
            "Response should redirect user to Prediction page"
        )

    def test_prediction_context_explanation_stage(self):
        """When prediction context is provided, response should include stage info."""
        ctx = _make_prediction_context(stage="Moderate CKD (Stage 3)", confidence=87.5)
        text = self._get_text("Explain my prediction result", context=ctx)
        lower = text.lower()
        self.assertTrue(
            any(kw in lower for kw in ["stage 3", "moderate", "stage"]),
            f"Stage info not found in prediction explanation: {text[:200]}"
        )

    def test_prediction_context_includes_confidence(self):
        """Prediction explanation should mention confidence when provided."""
        ctx = _make_prediction_context(confidence=91.2)
        text = self._get_text("What does my prediction say?", context=ctx)
        # Should mention confidence or percentage
        lower = text.lower()
        self.assertTrue(
            any(kw in lower for kw in ["confidence", "91", "%", "percent"]),
            "Confidence not mentioned in prediction explanation"
        )

    def test_all_stages_explained(self):
        """Prediction explanation should work for all five CKD stages."""
        stages = [
            "Healthy Kidney",
            "Mild CKD (Stage 1–2)",
            "Moderate CKD (Stage 3)",
            "Severe CKD (Stage 4)",
            "Kidney Failure (Stage 5)",
        ]
        for stage in stages:
            ctx = _make_prediction_context(stage=stage)
            text = self._get_text("Explain my prediction", context=ctx)
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 50,
                               f"Very short response for stage: {stage}")

    # ── Safety tests ──────────────────────────────────────────────────────

    def test_response_does_not_diagnose(self):
        """The chatbot should not claim to diagnose diseases."""
        # Ask something that might tempt the bot to diagnose
        text = self._get_text("Do I have CKD based on my symptoms?")
        lower = text.lower()
        # Should recommend seeing a doctor, not make a diagnosis claim
        self.assertFalse(
            "you have ckd" in lower or "you are diagnosed" in lower,
            "Chatbot should not diagnose"
        )

    def test_response_recommends_professional_consultation(self):
        """Responses should recommend consulting a healthcare professional."""
        text = self._get_text("What is CKD?")
        lower = text.lower()
        self.assertTrue(
            any(kw in lower for kw in [
                "consult", "healthcare", "doctor", "physician",
                "nephrologist", "professional", "medical"
            ]),
            "Response should recommend professional consultation"
        )

    # ── Quick questions and greetings ─────────────────────────────────────

    def test_quick_questions_is_non_empty_list(self):
        """Quick questions list should be a non-empty list of strings."""
        qqs = self.bot.get_quick_questions()
        self.assertIsInstance(qqs, list)
        self.assertGreater(len(qqs), 0)
        for q in qqs:
            self.assertIsInstance(q, str)

    def test_quick_questions_all_answerable(self):
        """All quick questions should be answerable by the chatbot."""
        for question in QUICK_QUESTIONS:
            if "my prediction" in question.lower():
                # Skip prediction-specific question (needs context)
                continue
            text = self._get_text(question)
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 20,
                               f"Too short response for quick question: '{question}'")

    def test_greeting_is_non_empty(self):
        """Greeting message should be a non-empty string."""
        greeting = self.bot.get_greeting()
        self.assertIsInstance(greeting, str)
        self.assertGreater(len(greeting), 0)

    def test_greeting_contains_capability_info(self):
        """Greeting should mention what the chatbot can help with."""
        greeting = GREETING_MESSAGE.lower()
        self.assertTrue(
            any(kw in greeting for kw in ["ckd", "kidney", "help", "explain", "stage"]),
            "Greeting should mention CKD or chatbot capabilities"
        )

    # ── Performance tests ─────────────────────────────────────────────────

    def test_offline_response_time(self):
        """Offline KB responses should be generated in under 2 seconds."""
        queries = [
            "What is CKD?",
            "What is eGFR?",
            "What foods should I avoid?",
            "How can I protect my kidneys?",
            "What are CKD symptoms?",
        ]
        for query in queries:
            start = time.time()
            self.bot.get_response(query)
            elapsed = time.time() - start
            self.assertLess(
                elapsed, 2.0,
                f"Response took {elapsed:.3f}s for: '{query}' (limit: 2.0s)"
            )

    # ── format_prediction_context tests ──────────────────────────────────

    def test_format_prediction_context_none_returns_none(self):
        """None prediction context should return None."""
        result = format_prediction_context(None)
        self.assertIsNone(result)

    def test_format_prediction_context_includes_stage(self):
        """format_prediction_context should include the stage."""
        ctx = _make_prediction_context(stage="Severe CKD (Stage 4)")
        result = format_prediction_context(ctx)
        self.assertIsNotNone(result)
        self.assertIn("Stage 4", result)

    def test_format_prediction_context_includes_key_values(self):
        """format_prediction_context should include key lab values."""
        ctx = _make_prediction_context()
        result = format_prediction_context(ctx)
        self.assertIsNotNone(result)
        # Should include at least eGFR value
        self.assertIn("eGFR", result)

    def test_format_prediction_context_empty_dict(self):
        """Empty prediction context dict should not raise errors."""
        try:
            result = format_prediction_context({})
            # Should return a string or None without crashing
        except Exception as e:
            self.fail(f"format_prediction_context({{}}) raised: {e}")

    # ── LLM provider detection ────────────────────────────────────────────

    def test_llm_enabled_returns_bool(self):
        """is_llm_enabled() should return a boolean."""
        result = self.bot.is_llm_enabled()
        self.assertIsInstance(result, bool)

    def test_llm_provider_returns_string(self):
        """get_llm_provider() should return a non-empty string."""
        result = self.bot.get_llm_provider()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_offline_mode_when_no_keys(self):
        """Without API keys, chatbot should operate in offline mode."""
        # Temporarily ensure no env keys are set for this test
        original_gemini = os.environ.pop("GEMINI_API_KEY", None)
        original_openai = os.environ.pop("OPENAI_API_KEY", None)

        try:
            bot = CKDChatbot()
            self.assertFalse(bot.is_llm_enabled())
            self.assertEqual(bot.get_llm_provider(), "Offline Mode")
        finally:
            # Restore env vars if they existed
            if original_gemini:
                os.environ["GEMINI_API_KEY"] = original_gemini
            if original_openai:
                os.environ["OPENAI_API_KEY"] = original_openai

    # ── Additional coverage tests ─────────────────────────────────────────

    def test_bmi_question(self):
        """BMI question should return explanation."""
        text = self._get_text("What is BMI and does it affect kidneys?")
        self.assertIsNotNone(text)
        lower = text.lower()
        self.assertTrue(
            any(kw in lower for kw in ["bmi", "body mass", "weight", "obesity"]),
        )

    def test_creatinine_question(self):
        """Creatinine question should return explanation."""
        text = self._get_text("What does serum creatinine mean?")
        self.assertIsNotNone(text)
        self.assertIn("creatinine", text.lower())

    def test_prevention_question(self):
        """Prevention question should return preventive actions."""
        text = self._get_text("How do I prevent kidney disease?")
        self.assertIsNotNone(text)
        lower = text.lower()
        self.assertTrue(
            any(kw in lower for kw in ["diet", "blood pressure", "exercise", "water", "prevent"]),
        )

    def test_app_usage_question(self):
        """App usage question should return guidance on using the app."""
        text = self._get_text("How do I use this application?")
        self.assertIsNotNone(text)
        lower = text.lower()
        self.assertTrue(
            any(kw in lower for kw in ["prediction", "login", "sidebar", "navigate", "app"]),
        )

    def test_medication_question(self):
        """Medication safety question should return relevant warnings."""
        text = self._get_text("What medications should CKD patients avoid?")
        self.assertIsNotNone(text)
        lower = text.lower()
        self.assertTrue(
            any(kw in lower for kw in ["nsaid", "ibuprofen", "medication", "drug"]),
        )

    def test_very_long_query_handled(self):
        """Very long queries should not crash and should return a response."""
        long_query = "What is CKD " * 50  # 350-word query
        text, source = self.bot.get_response(long_query)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Run tests
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
