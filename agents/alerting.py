"""
Alert Agent (A5)

Matches scored properties against user preference profiles and sends
alerts via configured channels (email, Telegram). In demo mode, all
alerts are logged but not actually sent.

Alert fatigue is real. If you send 20 alerts a day, people mute you.
So we have:
- Hard cap: 5 alerts per user per day
- Minimum threshold: value_ratio > 1.05 (at least 5% undervalued)
- Priority levels: >12% undervalued = high priority

The alert templates support Dutch and English — users set their
preferred language in their profile.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from agents.base import BaseAgent
from agents.schemas import PipelineState, ScoredProperty


# ── User preference model ─────────────────────────────────────────────

class UserPreference(BaseModel):
    """A user's search criteria and notification settings.

    In production, these live in Postgres. For demo, we use
    a hardcoded set of fictional users.
    """

    user_id: str
    name: str
    email: str = ""
    telegram_id: str = ""
    preferred_language: str = "nl"  # nl or en

    # Search criteria
    max_price: Optional[float] = None
    min_rooms: Optional[int] = None
    regions: list[str] = Field(default_factory=list)  # empty = any region

    # Alert settings
    channels: list[str] = Field(default_factory=lambda: ["log"])  # log, email, telegram
    min_value_ratio: float = 1.05  # only alert if this undervalued

    # Rate limiting (reset daily)
    alerts_today: int = 0
    last_alert_date: Optional[str] = None


# ── Demo users ────────────────────────────────────────────────────────

DEMO_USERS = [
    UserPreference(
        user_id="u-001",
        name="Jan de Vries",
        email="jan@example.nl",
        telegram_id="@jan_devries",
        preferred_language="nl",
        max_price=475000,
        min_rooms=2,
        regions=["amsterdam"],
        channels=["log", "telegram"],
    ),
    UserPreference(
        user_id="u-002",
        name="Lisa Bakker",
        email="lisa@example.nl",
        preferred_language="nl",
        max_price=450000,
        regions=["amsterdam", "utrecht"],
        channels=["log", "email"],
    ),
    UserPreference(
        user_id="u-003",
        name="Tom Kuipers",
        email="tom@example.nl",
        telegram_id="@tom_k",
        preferred_language="en",
        max_price=500000,
        min_rooms=3,
        regions=["amsterdam"],
        channels=["log", "telegram"],
        alerts_today=5,  # already at limit — should be skipped
        last_alert_date=datetime.utcnow().strftime("%Y-%m-%d"),
    ),
]


# ── Alert templates ───────────────────────────────────────────────────

TEMPLATES = {
    "nl": {
        "subject": "🏠 WoningScout: {address} - {undervalued_pct}% onder marktwaarde",
        "body": (
            "Hoi {user_name},\n\n"
            "WoningScout heeft een interessante woning gevonden:\n\n"
            "📍 {address}\n"
            "💰 Vraagprijs: €{asking_price:,.0f}\n"
            "📊 Geschatte waarde: €{predicted_price:,.0f} "
            "(€{ci_lower:,.0f} - €{ci_upper:,.0f})\n"
            "📈 {undervalued_pct:.1f}% onder marktwaarde\n"
            "🏘️ Buurt score: {livability}/10\n\n"
            "Vergelijkbare woningen:\n{comparables_text}\n\n"
            "Bekijk op Funda: https://funda.nl/koop/{listing_id}\n\n"
            "Groet,\nWoningScout"
        ),
    },
    "en": {
        "subject": "🏠 WoningScout: {address} - {undervalued_pct}% below market value",
        "body": (
            "Hi {user_name},\n\n"
            "WoningScout found a promising property:\n\n"
            "📍 {address}\n"
            "💰 Asking price: €{asking_price:,.0f}\n"
            "📊 Estimated value: €{predicted_price:,.0f} "
            "(€{ci_lower:,.0f} - €{ci_upper:,.0f})\n"
            "📈 {undervalued_pct:.1f}% below market value\n"
            "🏘️ Neighborhood score: {livability}/10\n\n"
            "Comparable sales:\n{comparables_text}\n\n"
            "View on Funda: https://funda.nl/koop/{listing_id}\n\n"
            "Cheers,\nWoningScout"
        ),
    },
}


def format_comparables(comparables: list, lang: str = "nl") -> str:
    """Format comparable properties for alert message."""
    if not comparables:
        return "  (geen vergelijkbare woningen gevonden)" if lang == "nl" else "  (no comparables found)"

    lines = []
    for i, comp in enumerate(comparables[:4], 1):
        lines.append(
            f"  {i}. {comp.address} — €{comp.sold_price:,.0f} "
            f"({comp.living_area_m2:.0f}m²)"
        )
    return "\n".join(lines)


def render_alert(
    prop: ScoredProperty,
    user: UserPreference,
    address: str = "",
) -> dict[str, str]:
    """Render alert message from template."""
    lang = user.preferred_language if user.preferred_language in TEMPLATES else "en"
    template = TEMPLATES[lang]

    addr = address or f"Listing {prop.listing_id}"
    comparables_text = format_comparables(prop.comparables, lang)

    context = {
        "user_name": user.name.split()[0],  # first name only
        "address": addr,
        "asking_price": prop.asking_price,
        "predicted_price": prop.predicted_price,
        "ci_lower": prop.ci_lower,
        "ci_upper": prop.ci_upper,
        "undervalued_pct": prop.undervalued_pct,
        "livability": prop.livability.composite,
        "comparables_text": comparables_text,
        "listing_id": prop.listing_id,
    }

    return {
        "subject": template["subject"].format(**context),
        "body": template["body"].format(**context),
    }


# ── Channel senders (demo = log only) ────────────────────────────────

class LogChannel:
    """Default channel — just logs the alert. Always available."""

    async def send(self, user: UserPreference, content: dict, priority: str = "normal"):
        import structlog
        logger = structlog.get_logger()
        logger.info(
            "alert_sent_log",
            user=user.user_id,
            channel="log",
            priority=priority,
            subject=content["subject"][:80],
        )


class EmailChannel:
    """Email channel — sends via SMTP in production, logs in demo."""

    def __init__(self, host: str = "", port: int = 587, user: str = "", password: str = ""):
        self.configured = bool(host and user)
        if not self.configured:
            import structlog
            structlog.get_logger().info("email_channel_not_configured_using_log_mode")

    async def send(self, user: UserPreference, content: dict, priority: str = "normal"):
        if not self.configured:
            # Demo mode: just log
            import structlog
            structlog.get_logger().info(
                "email_would_send",
                to=user.email,
                subject=content["subject"][:80],
                priority=priority,
            )
            return

        # Production SMTP sending would go here
        # import aiosmtplib
        # ...
        raise NotImplementedError("SMTP sending not yet implemented")


class TelegramChannel:
    """Telegram channel — sends via Bot API in production, logs in demo."""

    def __init__(self, token: str = ""):
        self.configured = bool(token)

    async def send(self, user: UserPreference, content: dict, priority: str = "normal"):
        if not self.configured:
            import structlog
            structlog.get_logger().info(
                "telegram_would_send",
                to=user.telegram_id,
                priority=priority,
            )
            return

        # Production Telegram Bot API call would go here
        # import httpx
        # async with httpx.AsyncClient() as client:
        #     await client.post(
        #         f"https://api.telegram.org/bot{self.token}/sendMessage",
        #         json={"chat_id": user.telegram_id, "text": content["body"]},
        #     )
        raise NotImplementedError("Telegram sending not yet implemented")


# ── Rate limiter ──────────────────────────────────────────────────────

class UserRateLimiter:
    """In-memory rate limiter. 5 alerts per user per day.

    In production this would use Redis with expiring keys.
    For demo, we track it in memory (resets when pipeline restarts).
    """

    MAX_PER_DAY = 5

    def __init__(self):
        self._counts: dict[str, int] = {}
        self._date: str = datetime.utcnow().strftime("%Y-%m-%d")

    def _maybe_reset(self):
        """Reset if it's a new day."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if today != self._date:
            self._counts.clear()
            self._date = today

    def is_exhausted(self, user_id: str) -> bool:
        self._maybe_reset()
        return self._counts.get(user_id, 0) >= self.MAX_PER_DAY

    def increment(self, user_id: str):
        self._maybe_reset()
        self._counts[user_id] = self._counts.get(user_id, 0) + 1


# ── The Agent ─────────────────────────────────────────────────────────

class AlertAgent(BaseAgent):
    """Agent A5: Match and alert users about undervalued properties.

    Only sends alerts for properties with value_ratio > 1.05 (5%+ undervalued).
    Rate-limited to 5 alerts per user per day.
    """

    MIN_VALUE_RATIO = 1.05

    def __init__(self):
        super().__init__()
        from agents.config import settings

        self.channels = {
            "log": LogChannel(),
            "email": EmailChannel(
                host=settings.smtp_host,
                port=settings.smtp_port,
                user=settings.smtp_user,
                password=settings.smtp_pass,
            ),
            "telegram": TelegramChannel(token=settings.telegram_bot_token),
        }
        self.rate_limiter = UserRateLimiter()
        self._users = DEMO_USERS  # In production: load from DB

    @property
    def name(self) -> str:
        return "alerting"

    def _matches_preferences(self, prop: ScoredProperty, user: UserPreference) -> bool:
        """Check if a property matches a user's search criteria."""
        # Price check
        if user.max_price and prop.predicted_price > user.max_price:
            return False

        # Room count check (we don't have this on ScoredProperty directly,
        # so we skip if not available — in production, we'd join with listings)
        # if user.min_rooms and prop.rooms < user.min_rooms:
        #     return False

        # Region check would require listing region — skip for demo
        # The real implementation joins with the listings table

        # Value ratio threshold
        if prop.value_ratio < max(self.MIN_VALUE_RATIO, user.min_value_ratio):
            return False

        return True

    async def _execute(self, state: PipelineState) -> PipelineState:
        if not state.scored_properties:
            self.logger.info("no_scored_properties")
            return state

        # Filter to undervalued only
        undervalued = [p for p in state.scored_properties if p.is_undervalued]
        if not undervalued:
            self.logger.info("no_undervalued_properties")
            state.alerts_sent = 0
            return state

        raw_store: dict = getattr(state, "_raw_listings", {})
        total_sent = 0
        total_skipped_rate_limit = 0

        for prop in undervalued:
            # Get address for alert template
            raw = raw_store.get(prop.listing_id)
            address = raw.address if raw else f"Listing {prop.listing_id}"

            for user in self._users:
                # Check match
                if not self._matches_preferences(prop, user):
                    continue

                # Check rate limit
                if self.rate_limiter.is_exhausted(user.user_id):
                    self.logger.debug(
                        "rate_limited",
                        user=user.user_id,
                        listing=prop.listing_id,
                    )
                    total_skipped_rate_limit += 1
                    continue

                # Render alert
                content = render_alert(prop, user, address)

                # Send via each configured channel
                priority = "high" if prop.value_ratio > 1.12 else "normal"

                for ch_name in user.channels:
                    channel = self.channels.get(ch_name)
                    if channel:
                        try:
                            await channel.send(user, content, priority)
                            total_sent += 1
                        except Exception as e:
                            self.logger.error(
                                "channel_send_failed",
                                channel=ch_name,
                                user=user.user_id,
                                error=str(e)[:200],
                            )

                self.rate_limiter.increment(user.user_id)

        state.alerts_sent = total_sent
        self._log_stats(
            state,
            alerts_sent=total_sent,
            undervalued_count=len(undervalued),
            rate_limited=total_skipped_rate_limit,
        )

        return state
