import base64
import csv
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import (
    XIAOMI_API_KEY,
    XIAOMI_BASE_URL,
    XIAOMI_TIMEOUT_SECONDS,
    TTS_MODEL,
    TTS_VOICE,
    TTS_FORMAT,
    TTS_TIMEOUT_SECONDS,
    TTS_STYLE_PROMPT,
    STT_MODEL_SIZE,
    STT_DEVICE,
    AUDIO_OUTPUT_DIR,
)

logger = logging.getLogger(__name__)


class VoiceAgent:
    """Voice-enabled support agent using MiMo TTS and Whisper STT."""

    def __init__(
        self,
        voice: str = None,
        audio_format: str = TTS_FORMAT,
        output_dir: Path = None,
    ):
        self.voice = voice or TTS_VOICE
        self.audio_format = audio_format
        self.output_dir = Path(output_dir or AUDIO_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._tts_client = None
        self._stt_model = None
        self._retriever = None

    @property
    def tts_available(self) -> bool:
        return bool(XIAOMI_API_KEY)

    def _get_tts_client(self):
        if self._tts_client is None:
            from openai import OpenAI
            self._tts_client = OpenAI(
                api_key=XIAOMI_API_KEY,
                base_url=XIAOMI_BASE_URL,
                timeout=TTS_TIMEOUT_SECONDS,
                max_retries=2,
            )
        return self._tts_client

    def _get_stt_model(self):
        if self._stt_model is None:
            from faster_whisper import WhisperModel
            logger.info("Loading Whisper STT model: %s on %s", STT_MODEL_SIZE, STT_DEVICE)
            self._stt_model = WhisperModel(STT_MODEL_SIZE, device=STT_DEVICE)
        return self._stt_model

    def _get_retriever(self):
        if self._retriever is None:
            from pipeline import build_default_retriever
            self._retriever = build_default_retriever()
        return self._retriever

    def text_to_speech(
        self,
        text: str,
        voice: str = None,
        style: str = None,
    ) -> Optional[bytes]:
        """Convert text to speech using MiMo V2.5 TTS.

        MiMo TTS uses the chat completions endpoint with an audio parameter.
        The text to synthesize goes in the assistant role message.
        Style/emotion instructions go in the user role message (not spoken).

        Returns raw WAV bytes, or None on failure.
        """
        if not self.tts_available:
            logger.warning("TTS unavailable: no API key")
            return None

        if not text or not text.strip():
            return None

        voice = voice or self.voice
        style = style or TTS_STYLE_PROMPT

        if len(text) > 6000:
            text = text[:6000] + "... Response truncated."

        try:
            client = self._get_tts_client()
            completion = client.chat.completions.create(
                model=TTS_MODEL,
                messages=[
                    {"role": "user", "content": style},
                    {"role": "assistant", "content": text},
                ],
                audio={"format": self.audio_format, "voice": voice},
            )

            audio_data = completion.choices[0].message.audio.data
            if not audio_data:
                logger.error("TTS returned empty audio data")
                return None

            return base64.b64decode(audio_data)

        except Exception as exc:
            logger.error("TTS error: %s: %s", type(exc).__name__, exc)
            return None

    def speech_to_text(self, audio_path: str) -> Optional[str]:
        """Convert speech to text using Whisper.

        Args:
            audio_path: Path to audio file (wav, mp3, m4a, etc.)

        Returns transcribed text, or None on failure.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.error("Audio file not found: %s", audio_path)
            return None

        try:
            model = self._get_stt_model()
            segments, info = model.transcribe(str(audio_path))
            text = " ".join(seg.text for seg in segments).strip()
            logger.info("STT transcribed %d chars (language=%s, probability=%.2f)",
                       len(text), info.language, info.language_probability)
            return text if text else None

        except Exception as exc:
            logger.error("STT error: %s: %s", type(exc).__name__, exc)
            return None

    def save_audio(self, audio_bytes: bytes, filename: str) -> Path:
        """Save audio bytes to file."""
        path = self.output_dir / filename
        path.write_bytes(audio_bytes)
        return path

    def _generate_filename(self, text: str) -> str:
        """Generate a deterministic filename from text."""
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"response_{text_hash}.wav"

    def process_ticket_text(
        self,
        issue: str,
        subject: str = "",
        company: str = "None",
        voice: str = None,
    ) -> Dict:
        """Process a text ticket through RAG and generate voice response.

        Returns dict with text_response, audio_path, status, etc.
        """
        from pipeline import triage_decision

        retriever = self._get_retriever()
        start = time.time()

        decision = triage_decision(issue, subject, company, retriever)
        response_text = decision.response

        audio_bytes = self.text_to_speech(response_text, voice=voice)
        audio_path = None
        if audio_bytes:
            filename = self._generate_filename(issue)
            audio_path = self.save_audio(audio_bytes, filename)

        elapsed = time.time() - start
        return {
            "text_response": response_text,
            "audio_path": str(audio_path) if audio_path else None,
            "audio_generated": audio_bytes is not None,
            "status": decision.status,
            "product_area": decision.product_area,
            "request_type": decision.request_type,
            "confidence": round(decision.confidence, 3),
            "justification": decision.justification,
            "elapsed_seconds": round(elapsed, 2),
        }

    def process_voice_input(
        self,
        audio_path: str,
        company: str = "None",
        voice: str = None,
    ) -> Dict:
        """Process voice input: STT -> RAG -> TTS."""
        transcription = self.speech_to_text(audio_path)
        if not transcription:
            return {
                "error": "Could not transcribe audio",
                "transcription": None,
                "text_response": None,
                "audio_path": None,
            }

        result = self.process_ticket_text(transcription, company=company, voice=voice)
        result["transcription"] = transcription
        return result

    def run_interactive(self, voice: str = None):
        """Interactive REPL mode for the voice agent."""
        print("=" * 60)
        print("  Voice Support Agent - Interactive Mode")
        print("=" * 60)
        print(f"  Voice: {voice or self.voice}")
        print(f"  Audio output: {self.output_dir}")
        print()
        print("  Commands:")
        print("    /voice <name>     Change voice (Mia/Chloe/Milo/Dean)")
        print("    /audio <path>     Process voice input from file")
        print("    /quit             Exit")
        print("    <text>            Type a support ticket")
        print("=" * 60)

        while True:
            try:
                user_input = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "/quit", "/exit"):
                print("Goodbye!")
                break

            if user_input.startswith("/voice "):
                new_voice = user_input.split(" ", 1)[1].strip()
                self.voice = new_voice
                print(f"Voice set to: {self.voice}")
                continue

            if user_input.startswith("/audio "):
                audio_path = user_input.split(" ", 1)[1].strip()
                print(f"Processing voice input: {audio_path}")
                result = self.process_voice_input(audio_path, voice=voice)
                if result.get("error"):
                    print(f"  Error: {result['error']}")
                else:
                    print(f"  Transcription: {result['transcription']}")
                    print(f"  Status: {result['status']}")
                    print(f"  Response: {result['text_response'][:200]}...")
                    if result["audio_path"]:
                        print(f"  Audio: {result['audio_path']}")
                continue

            result = self.process_ticket_text(user_input, voice=voice)
            print(f"\n  Status: {result['status']} ({result['product_area']})")
            print(f"  Confidence: {result['confidence']}")
            print(f"\n  {result['text_response']}")
            if result["audio_path"]:
                print(f"\n  Audio saved: {result['audio_path']}")

    def run_batch(
        self,
        input_csv: Path,
        voice: str = None,
    ) -> List[Dict]:
        """Batch mode: process CSV and generate audio for each ticket."""
        print(f"Loading retriever...")
        self._get_retriever()

        print(f"Reading tickets from {input_csv}...")
        with open(input_csv, "r", encoding="utf-8") as f:
            tickets = list(csv.DictReader(f))

        print(f"Processing {len(tickets)} tickets with voice '{voice or self.voice}'...")
        results = []

        for i, ticket in enumerate(tickets):
            issue = (
                ticket.get("Issue")
                or ticket.get("issue")
                or ticket.get("Description")
                or ticket.get("description")
                or ""
            ).strip()
            subject = (ticket.get("Subject") or ticket.get("subject") or "").strip()
            company = (ticket.get("Company") or ticket.get("company") or "None").strip()

            if not issue:
                continue

            label = subject[:50] if subject else issue[:50]
            print(f"  [{i+1}/{len(tickets)}] {label}...")

            result = self.process_ticket_text(issue, subject, company, voice=voice)
            results.append(result)

            status_icon = "OK" if result["audio_generated"] else "NO AUDIO"
            print(f"    -> {result['status']} / {status_icon} ({result['elapsed_seconds']}s)")

        replied = sum(1 for r in results if r["status"] == "replied")
        escalated = sum(1 for r in results if r["status"] == "escalated")
        audio_count = sum(1 for r in results if r["audio_generated"])
        print(f"\nDone. {len(results)} tickets processed.")
        print(f"  Replied: {replied}, Escalated: {escalated}")
        print(f"  Audio generated: {audio_count}/{len(results)}")
        print(f"  Audio directory: {self.output_dir}")

        return results


def main():
    """CLI entry point for the voice agent."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Voice Support Agent - MiMo TTS + RAG Pipeline"
    )
    parser.add_argument(
        "--input", type=str, default="",
        help="Path to input CSV (batch mode) or empty for interactive"
    )
    parser.add_argument(
        "--voice", type=str, default=TTS_VOICE,
        help=f"TTS voice name (default: {TTS_VOICE})"
    )
    parser.add_argument(
        "--audio-input", type=str, default="",
        help="Path to audio file for STT input"
    )
    parser.add_argument(
        "--audio-dir", type=str, default=str(AUDIO_OUTPUT_DIR),
        help=f"Output directory for audio files (default: {AUDIO_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run in interactive REPL mode"
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Process only first 10 tickets (batch mode)"
    )
    args = parser.parse_args()

    agent = VoiceAgent(
        voice=args.voice,
        output_dir=args.audio_dir,
    )

    if args.audio_input:
        result = agent.process_voice_input(args.audio_input)
        if result.get("error"):
            print(f"Error: {result['error']}")
        else:
            print(f"Transcription: {result['transcription']}")
            print(f"Status: {result['status']}")
            print(f"Response: {result['text_response']}")
            if result["audio_path"]:
                print(f"Audio: {result['audio_path']}")

    elif args.interactive:
        agent.run_interactive(voice=args.voice)

    elif args.input:
        input_path = Path(args.input)
        if args.sample:
            import tempfile
            with open(input_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)[:10]
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8")
            writer = csv.DictWriter(tmp, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            tmp.close()
            input_path = Path(tmp.name)

        agent.run_batch(input_path, voice=args.voice)

    else:
        agent.run_interactive(voice=args.voice)


if __name__ == "__main__":
    main()
