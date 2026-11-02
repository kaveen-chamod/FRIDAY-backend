# Ambient Presence Lifecycle

The presence service gates **only** authenticated `esp32_satellite` audio on
`/ws/live`. It does not stop the FastAPI process, browser clients, Telegram,
REST endpoints, or the defense orchestrator. Startup is `SLEEP_STANDBY`, so
microphone frames are discarded until a sensor or wake event activates the
home edge.

```text
SLEEP_STANDBY -- PIR / person / wake signal --> ACTIVE_PRESENCE
ACTIVE_PRESENCE -- no PIR and zero persons --> VACANT_COUNTDOWN (900 s)
VACANT_COUNTDOWN -- PIR or person detected --> ACTIVE_PRESENCE
VACANT_COUNTDOWN -- 900 s elapsed ----------> SLEEP_STANDBY
Master override -- verified passphrase -----> requested state
```

`ACTIVE_PRESENCE` and `VACANT_COUNTDOWN` both permit microphone streaming.
At countdown expiry, `/ws/live` refuses satellite PCM before Gemini Live sees
it, and notifies the satellite that its microphone is muted. Wake signals are
recorded as `wake_override:<source>` and transition directly to
`ACTIVE_PRESENCE`.

## Local API exercise

Start the backend with its normal environment, including `FRIDAY_API_TOKEN`.
These PowerShell commands use the authenticated REST API:

```powershell
$headers = @{ Authorization = "Bearer $env:FRIDAY_API_TOKEN" }
$base = "http://127.0.0.1:8000"

Invoke-RestMethod -Method Post -Uri "$base/api/presence/telemetry" `
  -Headers $headers -ContentType "application/json" `
  -Body '{"pir_active":true,"person_count":0,"light_lux":12.5}'

Invoke-RestMethod -Method Post -Uri "$base/api/presence/telemetry" `
  -Headers $headers -ContentType "application/json" `
  -Body '{"pir_active":false,"person_count":0,"light_lux":12.5}'

Invoke-RestMethod -Method Get -Uri "$base/api/presence/state" -Headers $headers
```

The second request begins the 900-second vacancy countdown. Poll the state
endpoint to observe `remaining_seconds`; after 15 minutes without presence it
reports `SLEEP_STANDBY` and `mic_active: false`. To test the sleep transition
without waiting, the unit tests inject a monotonic clock and advance it by 900
seconds; production timeout is fixed at 900 seconds.

Wake from a satellite event or an authenticated API request:

```powershell
Invoke-RestMethod -Method Post -Uri "$base/api/presence/wake-word" `
  -Headers $headers -ContentType "application/json" `
  -Body '{"source":"energy_threshold"}'
```

The energy threshold is a local amplitude trigger, not speech recognition.
The satellite sends no PCM while asleep and resumes only after a backend state
message reports that microphone input is active.
