# justagent model-relay (self-hosted)

A tiny, dependency-free HTTP relay that lets JustAgent-CLI talk to a local
model backend (such as [Ollama](https://ollama.com/)) through an
OpenAI-compatible endpoint, without ever exposing a port to the network or
calling out to a third-party API.

## Why this preserves local-first

JustAgent's local-first promise means your code, prompts, and error snippets
should never leave your machine. This relay is a **thin localhost-only proxy**:

- It binds to `127.0.0.1` exclusively -- never `0.0.0.0`.
- It forwards requests only to an allowlisted upstream (default
  `{"localhost", "127.0.0.1"}`), refusing anything else with HTTP 403.
- It uses only the Python standard library, so there is nothing extra to
  audit or supply-chain.

Your data stays on the loopback interface between JustAgent, the relay, and
your local model server. Nothing is sent to a public endpoint.

## Run it

Start a local model backend first, for example Ollama:

```bash
ollama serve
ollama pull qwen2.5:7b
```

Then start the relay (defaults shown explicitly):

```bash
python relay.py --port 8787 --upstream http://localhost:11434
```

Check that it is healthy:

```bash
curl http://127.0.0.1:8787/healthz
# {"status": "ok", "upstream": "http://localhost:11434"}
```

## Point JustAgent at the relay

In your `.justagent.toml`, add an Ollama backend whose `base_url` is the
relay. You can think of this as setting
`model.ollama_base_url = "http://localhost:8787"`; the canonical, working
form is:

```toml
[[model.backends]]
provider = "ollama"
base_url = "http://127.0.0.1:8787/v1"
model = "qwen2.5:7b"
timeout = 30.0
```

Because the relay speaks OpenAI's `/v1/chat/completions` dialect, JustAgent
addresses it exactly like any other OpenAI-compatible backend. Run
`justagent doctor` to confirm reachability.

## Security model

- **Bind address**: `127.0.0.1` only. The relay is unreachable from other
  hosts on the LAN and from the public internet.
- **Upstream allowlist**: only hosts in `ALLOWED_HOSTS` (default `localhost`
  and `127.0.0.1`) may be proxied to. Any other upstream host yields HTTP
  403 for every `/v1/chat/completions` request, and a warning is logged at
  startup.
- **Credential handling**: `Authorization` and other sensitive headers are
  redacted (`***REDACTED***`) from every audit and error log line. They are
  still forwarded to the upstream when present, so bearer auth continues to
  work against your local backend.
- **Hop-by-hop headers**: standard hop-by-hop headers are stripped in both
  directions to avoid ambiguous connection state.
- **No telemetry**: the relay makes no outbound calls except to the
  allowlisted upstream, and it writes no metrics anywhere.

## Endpoints

| Method | Path                    | Purpose                                          |
|--------|-------------------------|--------------------------------------------------|
| GET    | `/healthz`              | Liveness probe; echoes the configured upstream.  |
| GET    | `/`                     | Tiny service descriptor.                         |
| POST   | `/v1/chat/completions`  | Forwarded verbatim to `<upstream>/v1/chat/completions`. |

## Notes

- This is an example relay. For production hardening consider adding TLS
  termination on the loopback interface and request size limits.
- The relay does not cache responses; each request is proxied verbatim.
- Source `/v1/chat/completions` requests should include the model name in
  the JSON body, as expected by Ollama's OpenAI-compatible API.
