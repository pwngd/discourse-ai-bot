# Discourse AI Bot

A small Python bot that polls Discourse notifications, decides whether to reply with Ollama, and posts replies back through the published Discourse API.

Use `BOT_OLLAMA_HOST` to point the bot at Ollama. The project intentionally does not read `OLLAMA_HOST`, so it will not collide with other tools in your shell.

## Features

- Supports Discourse admin API auth (`Api-Key`, `Api-Username`) and session-cookie auth
- Supports optional custom headers like `Cookie` and `User-Agent` on all Discourse API requests
- Polls notifications and reacts to mentions, replies, and private messages
- Can optionally scan latest topics and let Ollama choose one post for a proactive reply
- Lets Ollama choose whether to reply or skip
- Lets Ollama optionally include one approved GIF from `./gifs` when it replies
- Can optionally attach official Roblox Engine API docs context for Roblox/Luau coding questions
- Schedules randomized reply delays and persists jobs in SQLite
- Keeps working across restarts
- Offers a best-effort optional presence adapter for `/presence/update`

## Quick Start

1. Set environment variables from `.env.example`.
2. Start the interactive bot shell:

```powershell
python -m discourse_ai_bot
```

This starts the polling worker and opens an interactive prompt where slash commands are available.

3. Run a health check if you want a one-shot connectivity test:

```powershell
python -m discourse_ai_bot healthcheck
```

4. Start the worker without the interactive shell if you prefer:

```powershell
python -m discourse_ai_bot run
```

### Auth Modes

Admin API mode:

```env
DISCOURSE_AUTH_MODE=api_key
DISCOURSE_TOKEN=...
DISCOURSE_USERNAME=...
```

Session-cookie mode:

```env
DISCOURSE_AUTH_MODE=session_cookie
DISCOURSE_COOKIE_STRING=session=abc; _t=xyz
```

If `DISCOURSE_COOKIE_STRING` or `DISCOURSE_COOKIE` is set and `DISCOURSE_AUTH_MODE` is omitted, the bot automatically uses `session_cookie` mode. The cookie string is the source-of-truth auth input for that mode and is also sent as the `Cookie` header on every Discourse API request.

## CLI

- `python -m discourse_ai_bot`
- `python -m discourse_ai_bot run`
- `python -m discourse_ai_bot healthcheck`
- `python -m discourse_ai_bot list-notifications`
- `python -m discourse_ai_bot list-manual-commands`
- `python -m discourse_ai_bot queue-ai-reply --post-url "https://forum.example.com/p/123" --request "Reply briefly and confirm we'll follow up."`
- `python -m discourse_ai_bot reply --topic-id 123 --reply-to-post-number 4 --raw "Hello"`
- `python -m discourse_ai_bot post-topic --title "Test" --raw "First post"`

## Manual AI Replies

While `run` is active, you can enqueue a manual AI-generated reply without interrupting notification polling:

```powershell
python -m discourse_ai_bot queue-ai-reply --post-url "https://forum.example.com/p/123" --request "Reply briefly and ask for their latest logs."
```

The worker will pick up the queued command on its next polling cycle, fetch the target post context, send your request plus the discussion to Ollama, apply the normal randomized delay, simulate typing if enabled, and then post the reply back to that Discourse post.

## Autonomous Latest-Post Replies

Set `BOT_AUTONOMOUS_REPLY_ENABLED=true` to let the worker occasionally inspect the latest Discourse topics. The scanner asks Ollama to choose at most one recent post that is worth a proactive reply. If Ollama picks a target above the configured confidence threshold, the bot queues a normal manual AI reply command for that post, so the existing delay, presence, retry, GIF, and posting flow is reused.

```env
BOT_AUTONOMOUS_REPLY_ENABLED=true
BOT_AUTONOMOUS_REPLY_INTERVAL=5m
BOT_AUTONOMOUS_REPLY_LATEST_COUNT=5
BOT_AUTONOMOUS_REPLY_MIN_CONFIDENCE=0.75
BOT_AUTONOMOUS_REPLY_BLOCKED_CATEGORY_URLS=
```

Posts the scanner already queued or skipped are remembered in SQLite so the same latest post is not reconsidered every polling cycle. Leave `BOT_AUTONOMOUS_REPLY_BLOCKED_CATEGORY_URLS` empty to allow all categories. To block categories, set it to a comma-separated list of category URLs, for example `https://forum.example.com/c/staff/4,https://forum.example.com/c/private/5`. Topics in child categories are skipped when their parent category is blocked, and the scanner will keep paging through latest topics until it fills the candidate set or runs out of pages.

## Optional Roblox API Docs Verification

Set `BOT_ROBLOX_DOCS_ENABLED=true` to let the bot fetch compact snippets from the official Roblox Creator Docs Engine API reference when a notification or manual request looks like a Roblox/Luau coding question. The lookup is gated by coding/API heuristics, so general Roblox discussion does not trigger docs requests.

- `BOT_ROBLOX_DOCS_SOURCE` chooses `auto`, `local`, or `remote`, default `auto`
- `BOT_ROBLOX_DOCS_LOCAL_PATH` points to a local sparse checkout, default `vendor/creator-docs`
- `BOT_ROBLOX_DOCS_REF` chooses the `Roblox/creator-docs` Git ref, default `main`
- `BOT_ROBLOX_DOCS_TIMEOUT_SECONDS` controls per-request timeout
- `BOT_ROBLOX_DOCS_CACHE_TTL` controls the in-memory docs cache lifetime, default `24h`
- `BOT_ROBLOX_DOCS_MAX_TERMS`, `BOT_ROBLOX_DOCS_MAX_RESULTS`, and `BOT_ROBLOX_DOCS_MAX_CONTEXT_CHARS` bound lookup and prompt size

For the fastest path, keep a local sparse checkout:

```powershell
git clone --depth 1 --filter=blob:none --sparse https://github.com/Roblox/creator-docs vendor/creator-docs
git -C vendor/creator-docs sparse-checkout set content/en-us/reference/engine
```

The `vendor/creator-docs/` directory is ignored by this repository. If the folder is missing in `auto` or `local` mode, the bot falls back to GitHub.

## Optional GIF Replies

If you create a local `gifs` folder in the project root and place `.gif` files inside it, the bot can optionally choose one GIF when it generates a reply.

- GIF choices are derived from filenames such as `friendly_wave.gif`
- The filename becomes the GIF id and description shown to Ollama
- The bot uploads the chosen GIF through the documented Discourse uploads API and appends it to the reply
- If no GIF fits, the bot sends a normal text reply

## Interactive Shell

Running `python -m discourse_ai_bot` opens an interactive prompt after the bot boots:

- `/help`
- `/health`
- `/notifications`
- `/manual`
- `/send "<post_url>" "<request>"`
- `/quit`

Plain text is also supported:

- Type a message and the shell will ask for the target post URL.
- Type a post URL and the shell will ask for the message to send.
- Type `<post_url> | <message>` to queue a manual AI reply in one line.

### AutoRead Timing

You can control how much reading time each AutoRead post reports to Discourse:

- Env default: `BOT_AUTOREAD_POST_TIME=2m`
- Runtime command: `/config autoread-time 1m`

Supported duration formats include `30s`, `1m`, and `1h`.

## Tests

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## Presence Configuration

If you enable `BOT_TYPING_MODE=presence_update`, you can configure the browser-style presence request headers with env vars:

- `DISCOURSE_PRESENCE_COOKIE` or `DISCOURSE_PRESENCE_COOKIE_STRING` for the full `Cookie` header value
- `DISCOURSE_PRESENCE_USER_AGENT` for a custom user agent
- `DISCOURSE_PRESENCE_EXTRA_HEADERS_JSON` for any additional headers, for example `{"Accept-Language":"en-US,en;q=0.9"}`

General Discourse API requests can also include browser-style headers:

- `DISCOURSE_COOKIE` or `DISCOURSE_COOKIE_STRING`
- `DISCOURSE_USER_AGENT`
- `DISCOURSE_EXTRA_HEADERS_JSON`

When both `DISCOURSE_COOKIE` and `DISCOURSE_COOKIE_STRING` are set, `DISCOURSE_COOKIE_STRING` wins.

Presence requests inherit from the general Discourse header settings unless you override them with the `DISCOURSE_PRESENCE_*` variants.
