"""
Social Media Content Agent using CrewAI + Google Gemini.

Generates platform-optimized content (Twitter/X, LinkedIn, Instagram)
from a topic or article URL.

Setup:
    pip install crewai litellm python-dotenv
    Create a .env file next to this script with:
        GEMINI_API_KEY=your_key_here

Usage:
    python agent.py --topic "The rise of AI agents in 2025"
    python agent.py --topic "New product launch: CloudSync Pro v3" --brand "CloudSync"
    python agent.py --topic "..." --mock          # run without an API key, offline demo
"""

import argparse
import os
import re
import textwrap

from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------------------------------------
# Pretty-printing helpers
# ----------------------------------------------------------------------

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
BLUE = "\033[34m"

WIDTH = 64


def rule(char="─", width=WIDTH):
    return char * width


def header(title: str, emoji: str = "") -> str:
    label = f" {emoji} {title} ".strip()
    pad = max(WIDTH - len(label) - 2, 0)
    left = pad // 2
    right = pad - left
    return f"{BOLD}{CYAN}╭{rule('─', left)}{label}{rule('─', right)}╮{RESET}"


def footer() -> str:
    return f"{BOLD}{CYAN}╰{rule('─', WIDTH)}╯{RESET}"


def section(title: str, emoji: str, color: str) -> str:
    return f"\n{BOLD}{color}{emoji}  {title}{RESET}\n{DIM}{rule('·')}{RESET}"


def wrap(text: str, indent: str = "  ") -> str:
    """Wrap text to WIDTH, preserving blank-line paragraph breaks."""
    paragraphs = text.strip().split("\n\n")
    wrapped_paragraphs = [
        textwrap.fill(p.strip(), width=WIDTH - len(indent), initial_indent=indent, subsequent_indent=indent)
        for p in paragraphs
        if p.strip()
    ]
    return "\n\n".join(wrapped_paragraphs)


def box_line(text: str = "") -> str:
    return text


# ----------------------------------------------------------------------
# Mock content generator (no API key needed) — realistic offline demo
# ----------------------------------------------------------------------

def _slugify(topic: str) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", topic.lower())
    return "".join(w.capitalize() for w in words[:3])


def mock_strategy(topic: str, brand: str, platforms: list[str]) -> dict:
    brand_label = brand if brand else "the brand"
    return {
        "message": f"{brand_label} is redefining what's possible with: {topic}",
        "audience": "Tech-forward professionals, builders, and early adopters aged 25-45",
        "hook": "Curiosity + FOMO — 'the shift is already happening, are you keeping up?'",
        "hashtags": [
            f"#{_slugify(topic)}",
            "#Innovation",
            "#TechTrends",
            "#FutureOfWork",
            "#BuildInPublic",
        ],
    }


def mock_posts(topic: str, brand: str, platforms: list[str]) -> dict:
    brand_label = brand if brand else "We"
    posts = {}

    if "twitter" in platforms:
        posts["twitter"] = {
            "tweet_1": f"{topic} isn't a trend. It's the new baseline. "
                       f"Here's what most people are still missing →",
            "tweet_2": f"Everyone's talking about {topic.lower()}. "
                       f"Almost no one is talking about what it actually takes to get there. Let's fix that.",
            "thread_opener": f"I've spent the last few weeks digging into {topic.lower()}. "
                              f"What I found changes how I think about the next 12 months. 🧵👇",
        }

    if "linkedin" in platforms:
        posts["linkedin"] = (
            f"Three years ago, if you'd told me {topic.lower()} would be where it is today, "
            f"I would've been skeptical.\n\n"
            f"{brand_label} has been tracking this shift closely, and one pattern keeps showing up: "
            f"the teams winning aren't the ones with the most resources — they're the ones willing "
            f"to rethink their fundamentals first.\n\n"
            f"That means:\n"
            f"→ Questioning workflows that \"have always worked\"\n"
            f"→ Investing in skills before they're mainstream\n"
            f"→ Treating change as a design problem, not a threat\n\n"
            f"The organizations that adapt now won't just keep up — they'll set the pace for "
            f"everyone else.\n\n"
            f"Where is your team in this shift? Curious to hear how others are approaching it."
        )

    if "instagram" in platforms:
        posts["instagram"] = {
            "caption": (
                f"{topic} is moving faster than most people realize. ⚡\n\n"
                f"Swipe to see what's actually changing — and why it matters for anyone building, "
                f"creating, or leading a team right now.\n\n"
                f"The future doesn't wait for permission. It just shows up. Are you ready for it?"
            ),
            "hashtags": [
                f"#{_slugify(topic)}", "#Innovation", "#TechTrends", "#FutureOfWork",
                "#BuildInPublic", "#StartupLife", "#DigitalTransformation", "#AI",
                "#Growth", "#Leadership", "#Productivity", "#TechNews",
                "#Entrepreneurship", "#NextGen", "#StayAhead",
            ],
        }

    return posts


# ----------------------------------------------------------------------
# Real CrewAI path (requires OPENAI_API_KEY)
# ----------------------------------------------------------------------

def generate_social_content_live(topic: str, brand: str, platforms: list[str]) -> str:
    from crewai import LLM, Agent, Crew, Process, Task

    # CrewAI's LLM class routes through LiteLLM, which understands the
    # "gemini/<model-name>" prefix and reads GEMINI_API_KEY from the
    # environment automatically (no need to pass api_key= explicitly,
    # though you can if you prefer).
    llm = LLM(
        model="gemini/gemini-3.5-flash",
        temperature=0.7,
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    strategist = Agent(
        role="Social Media Strategist",
        goal="Analyze the topic and define the key message, target audience, and tone for each platform",
        backstory="Award-winning social media strategist who has grown 50+ brand accounts to 100k+ followers.",
        llm=llm,
        verbose=False,
    )

    writer = Agent(
        role="Social Media Copywriter",
        goal="Write engaging, platform-optimized content that drives engagement",
        backstory="Viral content creator with expertise in platform-specific formats, hashtags, and hooks.",
        llm=llm,
        verbose=False,
    )

    strategy_task = Task(
        description=f"""Analyze this topic for social media: "{topic}"
Brand: {brand or 'Not specified'}
Platforms: {', '.join(platforms)}
Define: core message, target audience, emotional hook, 5 relevant hashtags.""",
        agent=strategist,
        expected_output="Content strategy: message, audience, hook, and hashtags",
    )

    writing_task = Task(
        description=f"""Write social media posts for: {', '.join(platforms)}
Topic: {topic}. Brand: {brand or 'General'}.

For each platform:
- Twitter/X: 2 tweet variations (under 280 chars each) + thread opener
- LinkedIn: Professional post (150-200 words) with storytelling hook
- Instagram: Caption (100-150 words) + 15 hashtags

Make them platform-native — Twitter punchy, LinkedIn thoughtful, Instagram visual.""",
        agent=writer,
        expected_output="Platform-optimized posts for all requested platforms",
        context=[strategy_task],
    )

    crew = Crew(
        agents=[strategist, writer],
        tasks=[strategy_task, writing_task],
        process=Process.sequential,
        verbose=False,
    )

    return str(crew.kickoff())


# ----------------------------------------------------------------------
# Beautiful rendering
# ----------------------------------------------------------------------

def render(topic: str, brand: str, platforms: list[str], strategy: dict, posts: dict) -> str:
    out = []
    out.append(header("SOCIAL MEDIA CONTENT AGENT", "📱"))
    out.append(f"{BOLD}Topic  {RESET} {topic}")
    out.append(f"{BOLD}Brand  {RESET} {brand or DIM + 'none specified' + RESET}")
    out.append(f"{BOLD}Channels{RESET} {', '.join(p.capitalize() for p in platforms)}")
    out.append(footer())

    out.append(section("STRATEGY BRIEF", "🎯", MAGENTA))
    out.append(f"  {BOLD}Core message:{RESET}")
    out.append(wrap(strategy["message"]))
    out.append(f"\n  {BOLD}Target audience:{RESET}")
    out.append(wrap(strategy["audience"]))
    out.append(f"\n  {BOLD}Emotional hook:{RESET}")
    out.append(wrap(strategy["hook"]))
    out.append(f"\n  {BOLD}Hashtags:{RESET} {' '.join(strategy['hashtags'])}")

    if "twitter" in posts:
        t = posts["twitter"]
        out.append(section("TWITTER / X", "🐦", BLUE))
        out.append(f"  {BOLD}Variation 1{RESET}")
        out.append(wrap(t["tweet_1"]))
        out.append(f"\n  {BOLD}Variation 2{RESET}")
        out.append(wrap(t["tweet_2"]))
        out.append(f"\n  {BOLD}Thread opener{RESET}")
        out.append(wrap(t["thread_opener"]))

    if "linkedin" in posts:
        out.append(section("LINKEDIN", "💼", GREEN))
        out.append(wrap(posts["linkedin"]))

    if "instagram" in posts:
        ig = posts["instagram"]
        out.append(section("INSTAGRAM", "📸", YELLOW))
        out.append(wrap(ig["caption"]))
        out.append(f"\n  {BOLD}Hashtags{RESET}")
        out.append(wrap(" ".join(ig["hashtags"])))

    out.append(f"\n{footer()}")
    return "\n".join(out)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _run(topic: str, brand: str, platforms: list[str], force_mock: bool = False) -> dict:
    """Core logic shared by the CLI and the web app. Returns a plain dict."""
    use_mock = force_mock or not os.getenv("GEMINI_API_KEY")

    if use_mock:
        strategy = mock_strategy(topic, brand, platforms)
        posts = mock_posts(topic, brand, platforms)
        return {
            "mode": "mock",
            "topic": topic,
            "brand": brand,
            "platforms": platforms,
            "strategy": strategy,
            "posts": posts,
        }
    else:
        content = generate_social_content_live(topic, brand, platforms)
        return {
            "mode": "live",
            "topic": topic,
            "brand": brand,
            "platforms": platforms,
            "content": content,
        }


# ----------------------------------------------------------------------
# Embedded frontend (single-file deployment — no templates/ folder needed)
# ----------------------------------------------------------------------

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dispatch — Social Media Content Agent</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --ink: #14171c;
    --panel: #1b2028;
    --panel-raised: #21272f;
    --line: #2a303a;
    --paper: #edeae4;
    --paper-dim: #9aa1ac;
    --brass: #c9972e;
    --brass-bright: #e3ac3c;
    --wire-blue: #4e9fe0;
    --signal-teal: #3e8e85;
    --transmission-rose: #c15b8c;
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    background: var(--ink);
    color: var(--paper);
    font-family: 'IBM Plex Sans', sans-serif;
    line-height: 1.5;
    min-height: 100vh;
  }

  .wrap {
    max-width: 1080px;
    margin: 0 auto;
    padding: 48px 24px 96px;
  }

  /* ---------- Header / Dispatch panel ---------- */

  .eyebrow {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--brass);
    margin-bottom: 18px;
  }

  .eyebrow .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--brass);
    box-shadow: 0 0 0 3px rgba(201, 151, 46, 0.2);
  }

  h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: clamp(32px, 5vw, 52px);
    line-height: 1.05;
    margin: 0 0 14px;
    letter-spacing: -0.01em;
  }

  .sub {
    color: var(--paper-dim);
    font-size: 16px;
    max-width: 46ch;
    margin: 0 0 32px;
  }

  /* Signature: waveform */
  .waveform {
    width: 100%;
    height: 40px;
    margin-bottom: 40px;
    overflow: hidden;
  }
  .waveform svg { display: block; width: 100%; height: 100%; }
  .waveform path {
    fill: none;
    stroke: var(--line);
    stroke-width: 1.5;
  }
  .waveform path.active {
    stroke: var(--brass);
  }
  .waveform.transmitting path.active {
    stroke-dasharray: 6 10;
    animation: pulse-flow 1.1s linear infinite;
  }
  @keyframes pulse-flow {
    to { stroke-dashoffset: -32; }
  }
  @media (prefers-reduced-motion: reduce) {
    .waveform.transmitting path.active { animation: none; }
  }

  /* ---------- Form ---------- */

  form {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 28px;
  }

  .field { margin-bottom: 20px; }

  .field label {
    display: block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--paper-dim);
    margin-bottom: 8px;
  }

  .field input[type="text"] {
    width: 100%;
    background: var(--ink);
    border: 1px solid var(--line);
    border-radius: 3px;
    padding: 12px 14px;
    color: var(--paper);
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 15px;
  }

  .field input[type="text"]:focus,
  .channel-toggle input:focus-visible + .channel-box {
    outline: 2px solid var(--brass);
    outline-offset: 1px;
  }

  .channels {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
  }

  .channel-toggle {
    position: relative;
  }
  .channel-toggle input {
    position: absolute;
    opacity: 0;
    width: 100%;
    height: 100%;
    cursor: pointer;
    margin: 0;
  }
  .channel-box {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    padding: 9px 16px;
    border: 1px solid var(--line);
    border-radius: 20px;
    color: var(--paper-dim);
    transition: border-color 0.15s, color 0.15s, background 0.15s;
  }
  .channel-toggle input:checked + .channel-box {
    color: var(--ink);
    background: var(--paper);
    border-color: var(--paper);
  }

  .row-end {
    display: flex;
    justify-content: flex-end;
    margin-top: 28px;
  }

  button[type="submit"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: var(--brass);
    color: var(--ink);
    border: none;
    border-radius: 3px;
    padding: 13px 26px;
    cursor: pointer;
    transition: background 0.15s;
  }
  button[type="submit"]:hover { background: var(--brass-bright); }
  button[type="submit"]:focus-visible { outline: 2px solid var(--paper); outline-offset: 2px; }
  button[type="submit"]:disabled { opacity: 0.5; cursor: not-allowed; }

  /* ---------- Status line ---------- */

  .status {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--paper-dim);
    margin-top: 16px;
    min-height: 18px;
  }
  .status.error { color: var(--transmission-rose); }

  /* ---------- Briefing note ---------- */

  .briefing {
    display: none;
    margin-top: 36px;
    border: 1px dashed var(--line);
    border-radius: 4px;
    padding: 20px 24px;
  }
  .briefing.show { display: block; }
  .briefing h3 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--brass);
    margin: 0 0 14px;
  }
  .briefing dl {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 8px 20px;
    margin: 0;
  }
  .briefing dt {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--paper-dim);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    white-space: nowrap;
    padding-top: 2px;
  }
  .briefing dd { margin: 0; font-size: 14px; }
  .briefing .tags { display: flex; flex-wrap: wrap; gap: 6px; }
  .briefing .tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--brass);
    background: rgba(201, 151, 46, 0.1);
    padding: 3px 9px;
    border-radius: 3px;
  }

  /* ---------- Channel cards ---------- */

  .channels-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 20px;
    margin-top: 28px;
  }

  .card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-left: 3px solid var(--line);
    border-radius: 4px;
    padding: 22px;
    opacity: 0;
    transform: translateY(8px);
    animation: rise 0.4s ease forwards;
  }
  @media (prefers-reduced-motion: reduce) {
    .card { animation: none; opacity: 1; transform: none; }
  }
  @keyframes rise {
    to { opacity: 1; transform: translateY(0); }
  }
  .card.twitter { border-left-color: var(--wire-blue); }
  .card.linkedin { border-left-color: var(--signal-teal); }
  .card.instagram { border-left-color: var(--transmission-rose); }

  .card-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    gap: 10px;
  }
  .card-head-labels {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 15px;
    letter-spacing: 0.01em;
  }
  .card-ch {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--paper-dim);
    letter-spacing: 0.08em;
  }

  .block { margin-bottom: 16px; }
  .block:last-child { margin-bottom: 0; }

  .block-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--paper-dim);
    margin-bottom: 6px;
  }

  .block-text {
    font-size: 14px;
    white-space: pre-wrap;
    color: var(--paper);
  }

  .copy-btn {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.05em;
    background: none;
    border: 1px solid var(--line);
    color: var(--paper-dim);
    border-radius: 3px;
    padding: 5px 10px;
    cursor: pointer;
    transition: border-color 0.15s, color 0.15s;
    white-space: nowrap;
    flex-shrink: 0;
  }
  .copy-btn:hover { border-color: var(--paper-dim); color: var(--paper); }
  .copy-btn:focus-visible { outline: 2px solid var(--brass); }
  .copy-btn.copied { color: var(--brass); border-color: var(--brass); }

  .hashtags {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--brass);
    line-height: 1.7;
    word-break: break-word;
  }

  /* ---------- Empty state ---------- */

  .empty {
    margin-top: 28px;
    padding: 40px 24px;
    text-align: center;
    color: var(--paper-dim);
    font-size: 14px;
    border: 1px dashed var(--line);
    border-radius: 4px;
  }
  .empty strong { color: var(--paper); font-weight: 500; }

  footer {
    margin-top: 56px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--paper-dim);
    text-align: center;
  }
</style>
</head>
<body>
<div class="wrap">

  <div class="eyebrow"><span class="dot"></span>DISPATCH</div>
  <h1>One idea.<br>Three transmissions.</h1>
  <p class="sub">Enter a topic and it goes out across Twitter/X, LinkedIn, and Instagram — each written for the channel it lands on.</p>

  <div class="waveform" id="waveform">
    <svg viewBox="0 0 400 40" preserveAspectRatio="none">
      <path class="track" d="M0,20 L400,20"></path>
      <path class="active" id="waveline" d="M0,20 L400,20"></path>
    </svg>
  </div>

  <form id="dispatch-form">
    <div class="field">
      <label for="topic">Topic</label>
      <input type="text" id="topic" name="topic" placeholder="e.g. The rise of AI agents in 2025" required>
    </div>
    <div class="field">
      <label for="brand">Brand <span style="opacity:0.6">(optional)</span></label>
      <input type="text" id="brand" name="brand" placeholder="e.g. CloudSync">
    </div>
    <div class="field">
      <label>Channels</label>
      <div class="channels">
        <label class="channel-toggle">
          <input type="checkbox" name="platform" value="twitter" checked>
          <span class="channel-box">Twitter / X</span>
        </label>
        <label class="channel-toggle">
          <input type="checkbox" name="platform" value="linkedin" checked>
          <span class="channel-box">LinkedIn</span>
        </label>
        <label class="channel-toggle">
          <input type="checkbox" name="platform" value="instagram" checked>
          <span class="channel-box">Instagram</span>
        </label>
      </div>
    </div>
    <div class="row-end">
      <button type="submit" id="submit-btn">Broadcast →</button>
    </div>
    <div class="status" id="status" role="status" aria-live="polite"></div>
  </form>

  <div class="briefing" id="briefing">
    <h3>Briefing note</h3>
    <dl>
      <dt>Message</dt><dd id="brief-message"></dd>
      <dt>Audience</dt><dd id="brief-audience"></dd>
      <dt>Hook</dt><dd id="brief-hook"></dd>
      <dt>Tags</dt><dd class="tags" id="brief-tags"></dd>
    </dl>
  </div>

  <div class="empty" id="empty-state">
    Nothing transmitted yet. Enter a topic above and hit <strong>Broadcast</strong>.
  </div>

  <div class="channels-grid" id="results" style="display:none"></div>

  <footer>social-media-content-agent · powered by gemini</footer>
</div>

<script>
const form = document.getElementById('dispatch-form');
const statusEl = document.getElementById('status');
const submitBtn = document.getElementById('submit-btn');
const waveform = document.getElementById('waveform');
const resultsEl = document.getElementById('results');
const emptyEl = document.getElementById('empty-state');
const briefingEl = document.getElementById('briefing');

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function copyBlock(btn, text) {
  navigator.clipboard.writeText(text).then(() => {
    const original = btn.textContent;
    btn.textContent = 'Copied';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = original; btn.classList.remove('copied'); }, 1600);
  });
}
window.copyBlock = copyBlock;

// Raw copy text is kept here (never re-serialized into HTML/attributes,
// which avoids quote-escaping bugs when content contains apostrophes
// like "isn't" or "they're").
const cardCopyText = [];

// Renders one card per platform with a SINGLE copy button in the card
// header that copies all of that platform's content at once.
function renderCard(kind, title, chLabel, blocks) {
  const combinedRaw = blocks.map(b => b.raw).join('\\n\\n');
  const cardIndex = cardCopyText.length;
  cardCopyText.push(combinedRaw);

  const blocksHtml = blocks.map(b => `
    <div class="block">
      <div class="block-label">${b.label}</div>
      <div class="${b.mono ? 'hashtags' : 'block-text'}">${escapeHtml(b.text)}</div>
    </div>
  `).join('');

  return `
    <div class="card ${kind}">
      <div class="card-head">
        <div class="card-head-labels">
          <span class="card-title">${title}</span>
          <span class="card-ch">${chLabel}</span>
        </div>
        <button type="button" class="copy-btn" data-card-index="${cardIndex}">Copy all</button>
      </div>
      ${blocksHtml}
    </div>
  `;
}

function wireCopyButtons(container) {
  container.querySelectorAll('.copy-btn[data-card-index]').forEach(btn => {
    btn.addEventListener('click', () => {
      const idx = Number(btn.getAttribute('data-card-index'));
      copyBlock(btn, cardCopyText[idx]);
    });
  });
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const topic = document.getElementById('topic').value.trim();
  const brand = document.getElementById('brand').value.trim();
  const platforms = Array.from(form.querySelectorAll('input[name="platform"]:checked')).map(el => el.value);

  if (!topic) return;
  if (platforms.length === 0) {
    statusEl.textContent = 'Select at least one channel.';
    statusEl.classList.add('error');
    return;
  }

  statusEl.classList.remove('error');
  statusEl.textContent = 'Transmitting…';
  submitBtn.disabled = true;
  waveform.classList.add('transmitting');
  emptyEl.style.display = 'none';
  cardCopyText.length = 0; // reset copy-text registry for this run

  try {
    const res = await fetch('/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic, brand, platforms })
    });

    if (!res.ok) throw new Error('Server returned ' + res.status);
    const data = await res.json();

    if (data.mode === 'mock' && data.strategy) {
      document.getElementById('brief-message').textContent = data.strategy.message;
      document.getElementById('brief-audience').textContent = data.strategy.audience;
      document.getElementById('brief-hook').textContent = data.strategy.hook;
      document.getElementById('brief-tags').innerHTML = data.strategy.hashtags.map(t => `<span class="tag">${escapeHtml(t)}</span>`).join('');
      briefingEl.classList.add('show');

      let html = '';
      if (data.posts.twitter) {
        const t = data.posts.twitter;
        html += renderCard('twitter', 'Twitter / X', 'CH.1', [
          { label: 'Variation 1', text: t.tweet_1, raw: t.tweet_1 },
          { label: 'Variation 2', text: t.tweet_2, raw: t.tweet_2 },
          { label: 'Thread opener', text: t.thread_opener, raw: t.thread_opener },
        ]);
      }
      if (data.posts.linkedin) {
        html += renderCard('linkedin', 'LinkedIn', 'CH.2', [
          { label: 'Post', text: data.posts.linkedin, raw: data.posts.linkedin },
        ]);
      }
      if (data.posts.instagram) {
        const ig = data.posts.instagram;
        html += renderCard('instagram', 'Instagram', 'CH.3', [
          { label: 'Caption', text: ig.caption, raw: ig.caption },
          { label: 'Hashtags', text: ig.hashtags.join(' '), raw: ig.hashtags.join(' '), mono: true },
        ]);
      }
      resultsEl.innerHTML = html;
      resultsEl.style.display = 'grid';
      wireCopyButtons(resultsEl);
    } else if (data.content) {
      // live CrewAI mode returns a single text blob
      briefingEl.classList.remove('show');
      resultsEl.innerHTML = renderCard('linkedin', 'Generated Content', 'LIVE', [
        { label: 'Output', text: data.content, raw: data.content },
      ]);
      resultsEl.style.display = 'grid';
      wireCopyButtons(resultsEl);
    }

    if (data.warning) {
      statusEl.textContent = data.warning;
      statusEl.classList.add('error');
    } else {
      statusEl.textContent = data.mode === 'mock'
        ? 'Transmission complete (offline demo content — set GEMINI_API_KEY for live generation).'
        : 'Transmission complete.';
    }
  } catch (err) {
    statusEl.textContent = 'Transmission failed: ' + err.message;
    statusEl.classList.add('error');
  } finally {
    submitBtn.disabled = false;
    waveform.classList.remove('transmitting');
  }
});
</script>
</body>
</html>
"""

# ----------------------------------------------------------------------
# Flask web app — lets this run as a Render "Web Service" via:
#   gunicorn agent:app
# ----------------------------------------------------------------------

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.get("/")
def home():
    return HTML_PAGE


@app.get("/api")
def api_info():
    return jsonify({
        "status": "ok",
        "service": "social-media-content-agent",
        "usage": "POST /generate with JSON: {\"topic\": \"...\", \"brand\": \"...\", \"platforms\": [\"twitter\",\"linkedin\",\"instagram\"]}",
    })


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


@app.post("/generate")
def generate():
    data = request.get_json(silent=True) or {}
    topic = data.get("topic", "How AI is transforming software development in 2025")
    brand = data.get("brand", "")
    platforms = data.get("platforms", ["twitter", "linkedin", "instagram"])
    platforms = [p.strip().lower() for p in platforms]
    force_mock = bool(data.get("mock", False))

    try:
        result = _run(topic, brand, platforms, force_mock=force_mock)
        return jsonify(result)
    except Exception as exc:
        # Surface the real error instead of a bare, unexplained 500.
        # Falls back to mock content so the UI still shows something useful.
        app.logger.exception("generate() failed, falling back to mock content")
        strategy = mock_strategy(topic, brand, platforms)
        posts = mock_posts(topic, brand, platforms)
        return jsonify({
            "mode": "mock",
            "topic": topic,
            "brand": brand,
            "platforms": platforms,
            "strategy": strategy,
            "posts": posts,
            "warning": f"Live generation failed ({type(exc).__name__}: {exc}); showing offline demo content instead.",
        }), 200


# ----------------------------------------------------------------------
# CLI entrypoint — still works locally: python agent.py --topic "..."
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Social Media Content Agent")
    parser.add_argument("--topic", default="How AI is transforming software development in 2025", help="Content topic")
    parser.add_argument("--brand", default="", help="Brand name (optional)")
    parser.add_argument("--platforms", default="twitter,linkedin,instagram", help="Comma-separated platforms")
    parser.add_argument("--mock", action="store_true", help="Run offline without calling any LLM API")
    parser.add_argument("--serve", action="store_true", help="Run as a local Flask web server instead of CLI")
    args = parser.parse_args()

    if args.serve:
        port = int(os.getenv("PORT", 5000))
        app.run(host="0.0.0.0", port=port)
        return

    platforms = [p.strip().lower() for p in args.platforms.split(",")]

    use_mock = args.mock or not os.getenv("GEMINI_API_KEY")

    if use_mock:
        strategy = mock_strategy(args.topic, args.brand, platforms)
        posts = mock_posts(args.topic, args.brand, platforms)
        print(render(args.topic, args.brand, platforms, strategy, posts))
        if not args.mock:
            print(f"\n{DIM}(no GEMINI_API_KEY found — showing offline mock output; "
                  f"set the key or pass --mock explicitly to silence this note){RESET}")
    else:
        content = generate_social_content_live(args.topic, args.brand, platforms)
        print(header("SOCIAL MEDIA CONTENT", "✍️"))
        print(content)
        print(footer())


if __name__ == "__main__":
    main()
