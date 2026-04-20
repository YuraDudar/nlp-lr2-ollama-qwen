"""
prompts.py — системные промпты и шаблоны для четырёх техник промптинга.

Каждая техника возвращает пару (system_prompt, user_template).
Шаблон user_template содержит плейсхолдер {message}, который подставляется
в run_evaluation.py перед отправкой запроса.

Техники:
    zero_shot       — прямой вопрос без примеров и цепочки рассуждений
    cot             — Chain-of-Thought: пошаговое рассуждение
    few_shot        — несколько размеченных примеров в контексте
    cot_few_shot    — комбинация CoT и few-shot
"""

from typing import NamedTuple


class PromptPair(NamedTuple):
    """Пара системного промпта и пользовательского шаблона."""

    system: str
    user_template: str


# ---------------------------------------------------------------------------
# Zero-shot
# ---------------------------------------------------------------------------

_ZERO_SHOT_SYSTEM = (
    "You are a spam detection system. "
    "Classify SMS messages as spam (1) or ham/legitimate (0). "
    'Always respond with valid JSON only: {"reasoning": "...", "verdict": 0 or 1}'
)

_ZERO_SHOT_USER = (
    "Classify this SMS as spam (1) or ham (0).\n\n"
    "SMS: {message}\n\n"
    'JSON: {{"reasoning": "...", "verdict": 0}}'
)

ZERO_SHOT = PromptPair(system=_ZERO_SHOT_SYSTEM, user_template=_ZERO_SHOT_USER)


# ---------------------------------------------------------------------------
# Chain-of-Thought (CoT)
# ---------------------------------------------------------------------------

_COT_SYSTEM = """You are a spam detection expert. When analyzing an SMS, reason step by step:

Step 1 — Trigger words: look for keywords like "free", "winner", "prize", "urgent", "claim", "click", "call now".
Step 2 — Formatting: check for ALL CAPS, excessive punctuation (!!!), shortcodes, or suspicious URLs.
Step 3 — Intent: does the message ask for personal info, money, or immediate action without context?
Step 4 — Naturalness: does it read like a genuine personal or business message?
Step 5 — Verdict: weigh the evidence and decide (1 = spam, 0 = ham).

Always respond with valid JSON only:
{"reasoning": "your step-by-step analysis", "verdict": 0 or 1}"""

_COT_USER = (
    "Analyze this SMS step by step, then give your verdict.\n\n"
    "SMS: {message}\n\n"
    "JSON:"
)

COT = PromptPair(system=_COT_SYSTEM, user_template=_COT_USER)


# ---------------------------------------------------------------------------
# Few-shot
# ---------------------------------------------------------------------------

_FEW_SHOT_SYSTEM = """You are a spam detection system. Use the labeled examples below to guide your classification.

SPAM examples (verdict: 1):
- "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Call 09061701461 to claim."
  → {"reasoning": "Prize claim + phone shortcode + urgency = spam", "verdict": 1}
- "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry."
  → {"reasoning": "Free prize competition + shortcode = spam", "verdict": 1}
- "URGENT! Your Mobile No 07808726822 was awarded a £2000 Bonus Caller Prize on 02/09/03!"
  → {"reasoning": "Fake award + ALL CAPS urgency + personal number reference = spam", "verdict": 1}
- "Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed & Free entry 2 100 wkly draw txt MUSIC to 87066"
  → {"reasoning": "Multiple prizes + shortcode + truncated language = spam", "verdict": 1}

HAM examples (verdict: 0):
- "Hey, are we still meeting at 3pm today? Let me know if you're running late."
  → {"reasoning": "Normal personal meeting query, no suspicious elements", "verdict": 0}
- "I'll be home by 7. Can you start dinner? Thanks."
  → {"reasoning": "Everyday household message, clearly personal", "verdict": 0}
- "Your appointment is confirmed for Monday 10am at City Clinic. Reply STOP to cancel reminders."
  → {"reasoning": "Legitimate business reminder, context is clear", "verdict": 0}
- "Happy birthday! Hope you have a great day. Miss you!"
  → {"reasoning": "Personal greeting, no spam indicators", "verdict": 0}

Classify the following SMS. Output valid JSON only:
{"reasoning": "brief explanation", "verdict": 0 or 1}"""

_FEW_SHOT_USER = "SMS: {message}\n\nJSON:"

FEW_SHOT = PromptPair(system=_FEW_SHOT_SYSTEM, user_template=_FEW_SHOT_USER)


# ---------------------------------------------------------------------------
# CoT + Few-shot
# ---------------------------------------------------------------------------

_COT_FEW_SHOT_SYSTEM = """You are a spam detection expert. Follow the reasoning steps AND use the examples below.

REASONING STEPS:
1. Trigger words: "free", "winner", "prize", "urgent", "claim", "click", "call now", etc.
2. Formatting: ALL CAPS, excessive "!!!", shortcodes (5-digit numbers), suspicious short URLs.
3. Intent: requests for personal info, money, or immediate unexplained action?
4. Naturalness: does it sound like a real personal or business message?
5. Verdict: 1 for spam, 0 for ham.

EXAMPLES WITH STEP-BY-STEP REASONING:

Example 1 (SPAM):
SMS: "You have won a guaranteed £1000 cash or a £2000 prize. To claim call 09050001! Claim code: UNDO. Valid 12hrs only."
{"reasoning": "Step1: 'won', 'prize', 'claim' — spam triggers. Step2: shortcode 09050001, ALL CAPS urgency. Step3: demands immediate call. Step4: unnatural, no personal context. Verdict: spam.", "verdict": 1}

Example 2 (HAM):
SMS: "Can you pick up the kids from school today? My meeting ran over."
{"reasoning": "Step1: no trigger words. Step2: normal punctuation, no shortcodes. Step3: normal request in personal context. Step4: natural and specific. Verdict: ham.", "verdict": 0}

Example 3 (SPAM):
SMS: "FREE MESSAGE: We tried to reach you re: your mobile phone replacement. CALL 08002988890 or 01223914905."
{"reasoning": "Step1: 'FREE', 'replacement' — bait. Step2: two phone numbers, ALL CAPS. Step3: unsolicited contact attempt. Step4: no personal context. Verdict: spam.", "verdict": 1}

Example 4 (HAM):
SMS: "Reminder: your dentist appointment is tomorrow at 2:30pm. Reply Y to confirm."
{"reasoning": "Step1: no prize/urgency triggers. Step2: normal formatting. Step3: standard appointment reminder. Step4: professional context, credible. Verdict: ham.", "verdict": 0}

Now classify the following SMS using the same reasoning. Output valid JSON only:
{"reasoning": "step-by-step analysis", "verdict": 0 or 1}"""

_COT_FEW_SHOT_USER = "SMS: {message}\n\nJSON:"

COT_FEW_SHOT = PromptPair(system=_COT_FEW_SHOT_SYSTEM, user_template=_COT_FEW_SHOT_USER)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TECHNIQUES: dict[str, PromptPair] = {
    "zero_shot": ZERO_SHOT,
    "cot": COT,
    "few_shot": FEW_SHOT,
    "cot_few_shot": COT_FEW_SHOT,
}


def get_prompt(technique: str, message: str) -> tuple[str, str]:
    """
    Return the (system, user) prompt pair for a given technique and SMS message.

    Args:
        technique: One of 'zero_shot', 'cot', 'few_shot', 'cot_few_shot'.
        message: The SMS text to classify.

    Returns:
        Tuple (system_prompt, user_prompt) ready to be sent to the LLM.

    Raises:
        ValueError: If the technique name is not recognised.
    """
    if technique not in TECHNIQUES:
        raise ValueError(f"Unknown technique '{technique}'. Choose from: {list(TECHNIQUES)}")
    pair = TECHNIQUES[technique]
    user = pair.user_template.format(message=message)
    return pair.system, user
