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
        model="gemini/gemini-1.5-flash",
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

def main():
    parser = argparse.ArgumentParser(description="Social Media Content Agent")
    parser.add_argument("--topic", default="How AI is transforming software development in 2025", help="Content topic")
    parser.add_argument("--brand", default="", help="Brand name (optional)")
    parser.add_argument("--platforms", default="twitter,linkedin,instagram", help="Comma-separated platforms")
    parser.add_argument("--mock", action="store_true", help="Run offline without calling any LLM API")
    args = parser.parse_args()

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
  
