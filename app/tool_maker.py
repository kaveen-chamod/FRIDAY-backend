from __future__ import annotations

import ast
import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.access_control import assert_safe_write_target
from core.sandbox import scan_for_danger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGING_DIR = Path(
    os.environ.get("FRIDAY_STAGING_DIR", PROJECT_ROOT / "staging_tools")
).expanduser().resolve()
_FORBIDDEN_CALLS = {"__import__", "compile", "eval", "exec", "open"}
_FORBIDDEN_IMPORTS = {
    "builtins", "ctypes", "importlib", "multiprocessing", "os", "pathlib",
    "pickle", "pty", "requests", "shutil", "socket", "subprocess", "sys",
    "tempfile", "urllib",
}
_FORBIDDEN_ATTRIBUTES = {
    "__builtins__", "__globals__", "__subclasses__", "connect", "kill",
    "link", "open", "popen", "remove", "rename", "rmdir", "system",
    "unlink", "write_bytes", "write_text",
}
_ALLOWED_IMPORTS = {
    "app.modules.base", "collections", "datetime", "decimal", "enum",
    "json", "math", "re", "statistics", "typing",
}
MANIFEST_FILE = PROJECT_ROOT / "memory" / "tool_manifest.json"


class ToolMaker:
    """Analyze and stage statically validated, review-only tool proposals."""

    def __init__(self, generate_text: Callable[[str], str]) -> None:
        self._generate_text = generate_text

    def stage(
        self, capability: str, action_registry=None, plugin_registry=None
    ) -> Path:
        capability = capability.strip()
        if not capability or len(capability) > 500:
            raise ValueError("Describe one capability in 1 to 500 characters.")
        specification = self.analyze(
            capability, action_registry=action_registry,
            plugin_registry=plugin_registry,
        )
        spec_prompt = (
            "Create one small Python BaseModule proposal from this validated "
            "JSON specification. Use the absolute import "
            "'from app.modules.base import BaseModule'. Return only Python "
            "source. Use type hints and validate all parameters. Do not import "
            "or perform filesystem, network, process, OS, or dynamic-code "
            "operations. Do not add dependencies.\n"
            f"Specification:\n{json.dumps(specification, ensure_ascii=False)}"
        )
        source = ""
        diagnostic = ""
        verification = {}
        for attempt in range(4):
            prompt = spec_prompt if attempt == 0 else (
                f"{spec_prompt}\n\nThe previous proposal failed static "
                f"or sandbox verification: {diagnostic}\n"
                f"Previous candidate:\n{source}\nReturn a corrected complete module."
            )
            source = self._clean_source(self._generate_text(prompt))
            try:
                self._validate(source)
                tool_metadata = self._tool_metadata(source)
                candidate_name = tool_metadata["name"]
                if candidate_name != specification["name"]:
                    raise ValueError(
                        "Generated TOOL name does not match the analyzed specification."
                    )
                if tool_metadata["description"] != specification["description"].strip():
                    raise ValueError(
                        "Generated TOOL description does not match its specification."
                    )
                if tool_metadata["parameters"] != specification["parameters"]:
                    raise ValueError(
                        "Generated TOOL parameters do not match their specification."
                    )
                verification = self._sandbox_test(source, specification)
                break
            except (ValueError, RuntimeError) as exc:
                diagnostic = str(exc)
        else:
            raise ValueError(
                "Proposal failed static or sandbox validation after 3 repair attempts: "
                + diagnostic
            )
        name = self._tool_name(source)
        assert_safe_write_target(STAGING_DIR)
        STAGING_DIR.mkdir(parents=True, exist_ok=True)
        destination = STAGING_DIR / f"{name}.py"
        assert_safe_write_target(destination)
        if destination.exists():
            raise FileExistsError(f"A staged proposal named '{name}' already exists.")
        try:
            with destination.open("x", encoding="utf-8") as stream:
                stream.write(source + "\n")
            destination.chmod(0o600)
            self._record_staged_tool(
                name, capability, specification, destination, verification
            )
        except Exception:
            assert_safe_write_target(destination)
            destination.unlink(missing_ok=True)
            raise
        return destination

    def analyze(
        self, capability: str, action_registry=None, plugin_registry=None
    ) -> dict:
        """Create a tool specification before asking for source code."""
        existing: list[dict] = []
        if action_registry is not None:
            existing.extend(action_registry.get_tool_declarations())
        if plugin_registry is not None:
            existing.extend(plugin_registry.get_tool_declarations())
        existing = [
            {"name": str(item.get("name", "")),
             "description": str(item.get("description", ""))}
            for item in existing if isinstance(item, dict)
        ]
        words = set(re.findall(r"\w+", capability.casefold()))
        possible_matches = []
        for tool in existing:
            tool_words = set(re.findall(
                r"\w+", f"{tool['name']} {tool['description']}".casefold()
            ))
            overlap = len(words & tool_words) / max(1, len(words | tool_words))
            if overlap >= 0.18:
                possible_matches.append(tool)
        inventory = {
            "request": capability,
            "language_note": "Interpret English, Sinhala, or Singlish as written.",
            "candidate_dependencies": [],
            "existing_capability_matches": possible_matches[:10],
            "existing_tool_names": sorted(
                {tool["name"] for tool in existing if tool["name"]}
            ),
        }
        analysis_prompt = (
            "Analyze the requested capability before writing code. Return only "
            "one JSON object with keys name, description, parameters, "
            "expected_output, candidate_dependencies, duplicate_of. `name` "
            "must be a Python identifier. `parameters` must be a JSON object "
            "schema with type OBJECT and properties. Dependencies must be "
            "standard-library modules only. If this duplicates an existing "
            "capability, set duplicate_of to its name; otherwise null. "
            "Understand English, Sinhala, and Singlish. Do not write code.\n"
            f"Request and available registry inventory:\n"
            f"{json.dumps(inventory, ensure_ascii=False)}"
        )
        raw_spec = self._clean_source(self._generate_text(analysis_prompt))
        try:
            specification = json.loads(raw_spec)
        except json.JSONDecodeError as exc:
            raise ValueError("Requirement analysis did not return valid JSON.") from exc
        if not isinstance(specification, dict):
            raise ValueError("Requirement analysis must return one JSON object.")
        name = specification.get("name")
        description = specification.get("description")
        parameters = specification.get("parameters")
        expected_output = specification.get("expected_output")
        dependencies = specification.get("candidate_dependencies")
        duplicate = specification.get("duplicate_of")
        if not isinstance(name, str) or not re.fullmatch(
            r"[a-zA-Z_][a-zA-Z0-9_]{0,63}", name
        ):
            raise ValueError("Requirement analysis returned an invalid tool name.")
        if name in inventory["existing_tool_names"]:
            raise ValueError(f"Capability already exists as '{name}'.")
        if duplicate:
            if not isinstance(duplicate, str):
                raise ValueError("Requirement analysis returned an invalid duplicate reference.")
            raise ValueError(f"Capability duplicates existing tool '{duplicate}'.")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Requirement analysis returned no tool description.")
        if (
            not isinstance(parameters, dict)
            or parameters.get("type") != "OBJECT"
            or not isinstance(parameters.get("properties"), dict)
        ):
            raise ValueError("Requirement analysis returned an invalid parameter schema.")
        if not isinstance(expected_output, str) or not expected_output.strip():
            raise ValueError("Requirement analysis returned no expected output.")
        if not isinstance(dependencies, list) or any(
            not isinstance(value, str) for value in dependencies
        ):
            raise ValueError("Requirement analysis returned invalid dependencies.")
        if any(
            dependency not in (_ALLOWED_IMPORTS - {"app.modules.base"})
            for dependency in dependencies
        ):
            raise ValueError("Requirement analysis proposed an unsupported dependency.")
        specification.update({
            "request": capability,
            "existing_capability_matches": possible_matches[:10],
            "existing_tool_names": inventory["existing_tool_names"],
        })
        return specification

    @staticmethod
    def _clean_source(source: str) -> str:
        cleaned = source.strip()
        cleaned = re.sub(r"^```(?:python|json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

    @staticmethod
    def _record_staged_tool(
        name: str,
        capability: str,
        specification: dict,
        destination: Path,
        verification: dict,
    ) -> None:
        assert_safe_write_target(MANIFEST_FILE)
        MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        manifest = {"version": 1, "tools": []}
        if MANIFEST_FILE.exists():
            try:
                manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError("The tool manifest is unreadable; refusing to overwrite it.") from exc
            if manifest.get("version") != 1 or not isinstance(manifest.get("tools"), list):
                raise RuntimeError("The tool manifest has an unsupported format.")

        previous = [
            item for item in manifest["tools"]
            if isinstance(item, dict) and item.get("name") == name
        ]
        if any(item.get("status") == "staged" for item in previous):
            raise FileExistsError(f"A staged manifest entry named '{name}' already exists.")
        major = minor = patch = 0
        if previous:
            match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", str(previous[-1].get("version", "")))
            if match:
                major, minor, patch = (int(value) for value in match.groups())
            patch += 1
        entry = {
            "name": name,
            "version": f"{major}.{minor}.{patch}",
            "status": "staged",
            "requested_capability": capability,
            "specification": specification,
            "path": (
                str(destination.relative_to(PROJECT_ROOT))
                if destination.is_relative_to(PROJECT_ROOT)
                else str(destination)
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "validation": {
                "ast": "passed",
                "static_security": "passed",
                "unit_tests": "passed",
                "sandbox_execution": "passed",
                "sandbox_diagnostic": verification.get("output", ""),
                "sandbox_duration_ms": verification.get("duration_ms", 0),
                "deployment": "disabled_pending_os_sandbox",
            },
        }
        manifest["tools"].append(entry)
        temporary = None
        try:
            assert_safe_write_target(MANIFEST_FILE.parent)
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=MANIFEST_FILE.parent,
                prefix=MANIFEST_FILE.name + ".", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                assert_safe_write_target(temporary)
                json.dump(manifest, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            if os.name != "nt":
                temporary.chmod(0o600)
            os.replace(temporary, MANIFEST_FILE)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _sandbox_test(source: str, specification: dict) -> dict:
        from core.sandbox import run_sandboxed

        parameters = _mock_parameters(specification["parameters"])
        runner = (
            "import json, sys, types, unittest\n"
            "app = types.ModuleType('app'); app.__path__ = []\n"
            "modules = types.ModuleType('app.modules'); modules.__path__ = []\n"
            "base = types.ModuleType('app.modules.base')\n"
            "class BaseModule: pass\n"
            "base.BaseModule = BaseModule\n"
            "sys.modules.update({'app': app, 'app.modules': modules, "
            "'app.modules.base': base})\n"
            f"candidate_source = {source!r}\n"
            f"mock_parameters = json.loads({json.dumps(json.dumps(parameters))})\n"
            "namespace = {'__name__': 'candidate_tool'}\n"
            "exec(compile(candidate_source, '<candidate-tool>', 'exec'), namespace)\n"
            "candidates = [value for value in namespace.values() "
            "if isinstance(value, type) and issubclass(value, BaseModule) "
            "and value is not BaseModule]\n"
            "if len(candidates) != 1: raise RuntimeError('Expected exactly one tool class.')\n"
            "tool = candidates[0]()\n"
            "class CandidateToolTests(unittest.TestCase):\n"
            "    def test_mock_parameters_return_nonempty_text(self):\n"
            "        result = tool.run(mock_parameters)\n"
            "        self.assertIsInstance(result, str)\n"
            "        self.assertTrue(result.strip())\n"
            "unittest.TextTestRunner(verbosity=0).run("
            "unittest.defaultTestLoader.loadTestsFromTestCase(CandidateToolTests))\n"
            "result = tool.run(mock_parameters)\n"
            "if not isinstance(result, str) or not result.strip(): "
            "raise RuntimeError('Tool returned an invalid result.')\n"
            "print('TOOL_TESTS_PASSED')\n"
        )
        with tempfile.TemporaryDirectory(prefix="friday-tool-test-") as directory:
            runner_path = Path(directory) / "candidate_test_runner.py"
            assert_safe_write_target(runner_path)
            runner_path.write_text(runner, encoding="utf-8")
            started = time.monotonic()
            output = run_sandboxed(runner_path, [], timeout=10)
            elapsed_ms = int((time.monotonic() - started) * 1000)
        if "TOOL_TESTS_PASSED" not in output:
            raise RuntimeError(
                "Sandbox-synthesized unit tests failed: " + output[:2000]
            )
        return {"output": output[:1000], "duration_ms": elapsed_ms}

    @staticmethod
    def _tool_name(source: str) -> str:
        return ToolMaker._tool_metadata(source)["name"]

    @staticmethod
    def _tool_metadata(source: str) -> dict:
        module = ast.parse(source)
        for node in module.body:
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if not (
                        isinstance(item, ast.Assign)
                        and any(
                            isinstance(target, ast.Name) and target.id == "TOOL"
                            for target in item.targets
                        )
                    ):
                        continue
                    try:
                        metadata = ast.literal_eval(item.value)
                    except (ValueError, TypeError) as exc:
                        raise ValueError("Proposal must declare a literal TOOL dictionary.") from exc
                    if not isinstance(metadata, dict):
                        raise ValueError("Proposal must declare a literal TOOL dictionary.")
                    name = metadata.get("name")
                    description = metadata.get("description")
                    parameters = metadata.get("parameters")
                    if not isinstance(name, str) or not re.fullmatch(
                        r"[a-zA-Z_][a-zA-Z0-9_]{0,63}", name
                    ):
                        raise ValueError("Proposal must declare a valid literal TOOL name.")
                    if not isinstance(description, str) or not description.strip():
                        raise ValueError("Proposal must declare a non-empty TOOL description.")
                    if (
                        not isinstance(parameters, dict)
                        or parameters.get("type") != "OBJECT"
                        or not isinstance(parameters.get("properties"), dict)
                    ):
                        raise ValueError("Proposal must declare a literal OBJECT parameter schema.")
                    return metadata
        raise ValueError("Proposal must declare a literal TOOL dictionary.")

    @staticmethod
    def _validate(source: str) -> None:
        try:
            tree = ast.parse(source, filename="<staged-tool>")
            compile(tree, "<staged-tool>", "exec")
        except SyntaxError as exc:
            raise ValueError(f"Generated proposal has invalid syntax: {exc}") from exc

        base_import = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "app.modules.base"
            and any(alias.name == "BaseModule" for alias in node.names)
            for node in tree.body
        )
        base_classes = [
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and any(isinstance(base, ast.Name) and base.id == "BaseModule" for base in node.bases)
        ]
        if not base_import or len(base_classes) != 1:
            raise ValueError("Proposal must import and subclass app.modules.base.BaseModule.")
        ToolMaker._tool_metadata(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = [alias.name.split(".", 1)[0] for alias in node.names]
                if any(name in _FORBIDDEN_IMPORTS for name in imports):
                    raise ValueError("Proposal imports a prohibited system or network module.")
                raise ValueError("Proposal must not use direct module imports.")
            if isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if module_name.split(".", 1)[0] in _FORBIDDEN_IMPORTS:
                    raise ValueError("Proposal imports a prohibited system or network module.")
                if module_name not in _ALLOWED_IMPORTS:
                    raise ValueError(f"Proposal imports an unsupported module: {module_name}.")
            if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRIBUTES:
                raise ValueError(f"Proposal accesses a prohibited operation: {node.attr}.")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in _FORBIDDEN_CALLS or node.func.id in {
                    "getattr", "setattr", "globals", "locals", "vars", "breakpoint",
                }:
                    raise ValueError(
                        "Proposal contains a prohibited dynamic-code or file operation."
                    )
        if not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
            for node in base_classes[0].body
        ):
            raise ValueError("The BaseModule subclass must implement run().")
        if any(
            isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS
                or isinstance(node.func, ast.Attribute) and node.func.attr in _FORBIDDEN_CALLS
            )
            for node in ast.walk(tree)
        ):
            raise ValueError("Proposal contains a prohibited dynamic-code or file operation.")
        hazards = scan_for_danger(source)
        if hazards:
            raise ValueError("Proposal flagged by the static safety scan: " + ", ".join(hazards))


def _mock_parameters(schema: dict) -> dict:
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        raise ValueError("Tool input schema is malformed.")
    parameters: dict = {}
    for name in required:
        if not isinstance(name, str) or name not in properties:
            raise ValueError("Tool input schema has an unknown required parameter.")
    for name, definition in properties.items():
        if not isinstance(name, str) or not isinstance(definition, dict):
            raise ValueError("Tool input schema contains an invalid property.")
        enum_values = definition.get("enum")
        if isinstance(enum_values, list) and enum_values:
            parameters[name] = enum_values[0]
            continue
        kind = str(definition.get("type", "STRING")).upper()
        if kind == "STRING":
            parameters[name] = "test"
        elif kind in {"INTEGER", "INT"}:
            parameters[name] = 1
        elif kind in {"NUMBER", "FLOAT"}:
            parameters[name] = 1.0
        elif kind in {"BOOLEAN", "BOOL"}:
            parameters[name] = True
        elif kind == "ARRAY":
            parameters[name] = []
        elif kind in {"OBJECT", "DICT"}:
            parameters[name] = {}
        else:
            raise ValueError(f"Unsupported mock parameter type: {kind}.")
    return parameters
