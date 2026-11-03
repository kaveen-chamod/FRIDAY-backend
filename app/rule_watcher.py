# app/rule_watcher.py
from __future__ import annotations
import asyncio
import logging
from app.telegram_bot import send_telegram_alert

_LOG = logging.getLogger("friday.rule_watcher")

async def notify_rule_change(speaker_name: str, speaker_role: str, change_description: str):
    """පවුලේ සාමාජිකයෙක් නීතියක් හෝ තත්ත්වයක් වෙනස් කළ විට Master ට Telegram Alert යැවීම"""
    alert_msg = (
        f"⚠️ [FAMILY RULE CHANGE ALERT]\n"
        f"👤 Member: {speaker_name} ({speaker_role})\n"
        f"🔄 Action/Change: {change_description}\n"
        f"🔒 Status: Logged & Audited by F.R.I.D.A.Y."
    )
    try:
        await send_telegram_alert(alert_msg)
        _LOG.info(f"Rule change alert sent for {speaker_name}: {change_description}")
    except Exception as exc:
        _LOG.error(f"Failed to send rule change alert via Telegram: {exc}")