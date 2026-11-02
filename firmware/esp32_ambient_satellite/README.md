# ESP32-S3 Ambient Voice Satellite

The satellite uses a TLS-verified WebSocket connection to `/ws/live`. It starts
with its microphone muted, streams mono 16 kHz signed 16-bit PCM only after the
backend reports `mic_active: true` (`ACTIVE_PRESENCE` or `VACANT_COUNTDOWN`),
and discards local microphone samples while the backend is asleep. In standby,
an energy threshold can request a wake; raw audio is not sent as part of that
request. A PIR event sends telemetry and wakes the backend. The backend remains
responsible for the authoritative state.

## Pinout

| Peripheral | Signal | ESP32-S3 GPIO |
|---|---|---:|
| PIR sensor | Digital output | 13 |
| INMP441 microphone | SCK / BCLK | 14 |
| INMP441 microphone | WS / LRCL | 15 |
| INMP441 microphone | SD / DOUT | 32 |
| MAX98357A amplifier | BCLK | 26 |
| MAX98357A amplifier | LRC | 16 |
| MAX98357A amplifier | DIN | 17 |
| Status NeoPixel | Data | 21 |

GPIO 22 and GPIO 25 are not available on the ESP32-S3 target selected here;
the amplifier LRC and DIN signals therefore use GPIO 16 and GPIO 17.
Connect grounds together. Power the amplifier from an appropriate supply for
the speaker and board; do not power the speaker from an ESP32 GPIO. Connect the
INMP441 L/R select to ground for the configured left-channel capture. Verify
module supply and logic-level requirements before wiring.

```text
ESP32-S3 GPIO 14 (BCLK)  ---> INMP441 SCK
ESP32-S3 GPIO 15 (WS)    ---> INMP441 WS
ESP32-S3 GPIO 32 (DATA)  <--- INMP441 SD
ESP32-S3 GND             ---> INMP441 GND, MAX98357A GND, PIR GND
ESP32-S3 GPIO 26 (BCLK)  ---> MAX98357A BCLK
ESP32-S3 GPIO 16 (LRC)   ---> MAX98357A LRC
ESP32-S3 GPIO 17 (DATA)  ---> MAX98357A DIN
ESP32-S3 GPIO 13         <--- PIR OUT
ESP32-S3 GPIO 21         ---> NeoPixel DIN
External suitable supply ---> MAX98357A VIN / speaker supply
```

## Configure and flash

1. Install PlatformIO Core and the ESP32-S3 USB drivers required by your board.
2. Copy `include/secrets.example.h` to `include/secrets.h`.
3. Set the Wi-Fi SSID/password, backend DNS name, TLS port, API token, and
   backend CA certificate PEM in `include/secrets.h`. The token must have the
   backend's configured API-token value; do not commit this file.
4. Set `FRIDAY_WS_PATH` to `/ws/live`. Use a valid certificate chain for the
   configured backend host. The firmware does not use insecure TLS.
5. From this directory run `pio run -t upload`, then `pio device monitor`.

The firmware fails closed and leaves the LED off if credentials, a CA
certificate, or either I2S pipeline is unavailable. The default example
credentials are deliberately blank and will not connect.

## Audio and wake behavior

The INMP441 delivers 24-bit samples in 32-bit I2S slots; firmware shifts and
converts them to signed 16-bit samples before transmission. Backend playback
is 24 kHz mono PCM, duplicated to both MAX98357A channels. The NeoPixel is off
when disconnected or muted, amber when the backend permits microphone input,
and cyan during speaker playback.

The standby energy detector is a configurable amplitude threshold, not a
lexical "Hey Friday" recognizer. It transmits only a `wake_word` control
message (`source: energy_threshold`) and waits for backend confirmation before
streaming PCM. For lexical wake-word detection, attach a separately validated
on-device wake-word model; do not substitute continuous audio forwarding.

The ESP32-S3 firmware does not include an ESP32-CAM. A camera service can send
`person_count` through authenticated `POST /api/presence/telemetry`; omitted
person counts in satellite PIR updates preserve the last camera value.
