from __future__ import annotations

import json
import os
import time
import uuid


_MOVEMENTS = {"FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"}
_SWITCH_STATES = {"ON", "OFF"}


def _publish(payload: dict) -> str:
    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:
        raise RuntimeError("MQTT support is not installed; install paho-mqtt.") from exc

    host = os.environ.get("FRIDAY_MQTT_HOST", "").strip()
    username = os.environ.get("FRIDAY_MQTT_USERNAME", "").strip()
    password = os.environ.get("FRIDAY_MQTT_PASSWORD", "")
    topic = os.environ.get("FRIDAY_MQTT_TOPIC", "").strip()
    if not all((host, username, password, topic)):
        raise RuntimeError(
            "Configure FRIDAY_MQTT_HOST, FRIDAY_MQTT_USERNAME, "
            "FRIDAY_MQTT_PASSWORD, and FRIDAY_MQTT_TOPIC first."
        )

    port = int(os.environ.get("FRIDAY_MQTT_PORT", "8883"))
    if port != 8883:
        raise RuntimeError("Secure HiveMQ MQTT requires FRIDAY_MQTT_PORT=8883.")
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"friday-{uuid.uuid4().hex[:12]}",
        protocol=mqtt.MQTTv311,
    )
    client.username_pw_set(username, password)
    client.tls_set()
    loop_started = False
    try:
        client.connect(host, port=port, keepalive=15)
        client.loop_start()
        loop_started = True
        result = client.publish(topic, json.dumps(payload), qos=1, retain=False)
        result.wait_for_publish(timeout=5)
        if not result.is_published() or result.rc != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"MQTT publish failed with result code {result.rc}.")
    finally:
        if loop_started:
            client.loop_stop()
        client.disconnect()
    return f"Published the confirmed command to MQTT topic '{topic}'."


def control_iot(parameters: dict, **_context) -> str:
    action = str(parameters.get("action") or "").strip().lower()
    device_id = str(parameters.get("device_id") or "").strip()
    if not device_id or len(device_id) > 80:
        return "A valid device_id is required."
    if action == "move":
        direction = str(parameters.get("direction") or "").strip().upper()
        if direction not in _MOVEMENTS:
            return "Movement must be FORWARD, BACKWARD, LEFT, RIGHT, or STOP."
        command = {"device_id": device_id, "action": "move", "direction": direction}
    elif action == "switch":
        state = str(parameters.get("state") or "").strip().upper()
        if state not in _SWITCH_STATES:
            return "Switch state must be ON or OFF."
        command = {"device_id": device_id, "action": "switch", "state": state}
    else:
        return "Action must be 'move' or 'switch'."
    command.update({"source": "F.R.I.D.A.Y.", "timestamp": int(time.time())})
    try:
        emergency_stop = action == "move" and command.get("direction") == "STOP"
        if not emergency_stop:
            from app.digital_twin import get_default_engine

            twin = get_default_engine()
            if twin is None:
                return (
                    "NOT_SUPPORTED: No digital twin is available; physical actuation "
                    "was not sent."
                )
            device_configuration = getattr(twin, "device_configuration", None)
            if callable(device_configuration):
                configuration = device_configuration(device_id)
                specification = (
                    configuration.get("specification")
                    if isinstance(configuration, dict)
                    else None
                )
                if (
                    isinstance(configuration, dict)
                    and configuration.get("domain") == "robotics"
                    and (
                        not isinstance(specification, dict)
                        or specification.get("vehicle_type") != "rc_car"
                    )
                ):
                    return (
                        "NOT_SUPPORTED: This device is not an enabled RC car; "
                        "physical actuation was not sent."
                    )
            preflight = twin.preflight_mqtt_command(device_id, command)
            if preflight.get("dispatch_allowed") is not True:
                return (
                    f"Physical actuation rejected by digital-twin preflight "
                    f"({preflight.get('status', 'FAILED')}): "
                    f"{preflight.get('detail', 'No validated safe result.')}"
                )
        return _publish(command)
    except (OSError, RuntimeError, ValueError) as exc:
        return f"MQTT command was not delivered: {exc}"


TOOL = {
    "name": "iot_control",
    "description": (
        "For user-requested RC car movement or switch control, preserve the "
        "provided device_id and discrete direction/state. Non-STOP physical "
        "commands require the validated Digital Twin preflight and explicit "
        "user confirmation before authenticated TLS MQTT publishing. STOP is "
        "an immediate fail-safe and does not wait for inference. Understand "
        "English, spoken Sinhala, and Singlish; ask for missing device/action "
        "details instead of guessing. This tool does not control drones."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {"type": "STRING", "enum": ["move", "switch"]},
            "device_id": {"type": "STRING", "description": "Configured device identifier."},
            "direction": {
                "type": "STRING",
                "enum": ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"],
            },
            "state": {"type": "STRING", "enum": ["ON", "OFF"]},
        },
        "required": ["action", "device_id"],
    },
    "handler": control_iot,
}
